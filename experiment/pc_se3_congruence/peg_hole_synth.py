"""Peg-and-hole synthetic point-cloud -> stiffness data.

Replaces the abstract Gaussian-blob clouds of :mod:`data_synth` with scenes
that look like a real insertion task: a prismatic peg (circle / triangle /
square / hexagon / rectangle / ellipse / D-profile cross-section, optional tip
chamfer) and a plate with a matching through-hole (peg profile scaled by
1 + clearance).  Poses are drawn from a three-stage mixture that mirrors an
assembly sequence:

  free    peg hovering above the plate (no contact),
  search  peg tip touching / skimming the plate surface near the hole rim,
  insert  peg partially-to-deeply inserted; lateral offset, axial yaw mismatch
          and tilt are sampled inside the clearance budget and validated by an
          explicit no-penetration test.

Point clouds are uniform area-weighted surface samples of both parts with
additive sensor noise, placed at a random global SE(3) pose (trans_scale O(1),
matching the rest of the pipeline).

Two label terms are produced, both exactly congruence-equivariant by the same
argument as ``data_synth.contact_spring_K`` (invariant weights x coadjoint
wrench outer products), and both written as MONTE-CARLO SURFACE INTEGRALS so
that they are properties of the geometry rather than of the sampling:

  K_contact  cross-part contact springs.  Physically the contact stiffness is
             the double surface integral

               K = int_{peg} int_{plate} kappa(|x-y|) w w^T dA(y) dA(x),

             so a point sampled uniformly from a surface of area A out of N
             points carries area weight A/N and the estimator is

               K = (A_peg/N_peg)(A_plate/N_plate) sum_{i,j} kappa w w^T.

             Every cross pair inside the hard contact radius contributes --
             there is no k-th-neighbour cut, which would otherwise re-introduce
             a density dependence of its own.  With no contact K vanishes
             exactly; under wall contact its range spans the locked directions.
  K_body     the same estimator over ALL pairs (not just cross-part) with its
             own radius -- a full-rank background term that keeps the composed
             target SPD so the AIRM distance is defined.

The training target is K_contact + lambda_body * K_body (composed in the
loader so lambda can be swept without regenerating data).

WHY THE AREA WEIGHTS MATTER.  The earlier 1/(N_peg k) normalisation made the
label drift with resolution -- the same physical contact gave |K_contact|
9.6e-3 / 7.5e-3 / 4.8e-3 at N = 2048 / 1024 / 512 (AIRM 1.6-2.8 apart), i.e.
part of the "answer" was the discretisation rather than the geometry.  With
area weights the same scenes agree to within +-5% over an 8x density range,
which is Monte-Carlo noise.

Frames.  Canonical (hole) frame: plate top surface z = 0, plate occupies
z in [-T, 0], hole axis = +z through the origin.  Peg frame: extrusion axis
+z, tip cross-section at z = 0, top at z = H; posed by (R_peg, p_peg) with
p_peg the tip-center position in the hole frame.  Cross-sections are convex
polygons (a 128-gon for smooth profiles; chord error ~5e-4 * r, below the
sensor noise floor).
"""
import math

import torch

from experiment.pc_se3_congruence.data_synth import contact_spring_kernel_K
from experiment.pc_se3_congruence.encoders import compact_wendland_weights
from experiment.pc_se3_congruence.se3_utils import random_SO3

PROFILE_TYPES = ('circle', 'triangle', 'square', 'hexagon', 'rect',
                 'ellipse', 'dee')
STAGES = ('free', 'search', 'insert')

DEFAULT_CFG = {
    # profile mixture (indices into PROFILE_TYPES)
    'profile_weights': (0.25, 0.10, 0.15, 0.15, 0.15, 0.10, 0.10),
    'stage_probs': (0.15, 0.25, 0.60),
    # scene geometry (scene units are O(1); sigma(P) ~ 1 like data_synth)
    'peg_radius': (0.22, 0.45),          # profile circumradius
    'peg_height': (0.9, 1.6),
    'plate_width': (1.8, 3.0),           # per-axis full width
    'plate_thickness': (0.25, 0.6),
    'clearance': (0.02, 0.08),           # hole = profile scaled by (1 + c)
    'chamfer_prob': 0.6,
    'chamfer_height': (0.05, 0.12),      # fraction of peg height
    'chamfer_scale': (0.75, 0.9),        # tip cross-section scale s0
    # stage parameters
    'free_gap': (0.33, 0.85),            # lowest peg point above plate top
                                         # (min > contact_radius + 3*noise
                                         #  => K_contact is EXACTLY zero)
    'free_tilt_deg': 20.0,
    'search_gap': (0.002, 0.02),
    'search_tilt_deg': 8.0,
    'search_offset': (0.3, 1.3),         # fraction of hole circumradius
    'insert_depth': (0.06, 1.0),         # fraction of d_max
    'insert_margin': 0.1,                # safety margin, fraction of clearance
    # sampling
    'min_peg_frac': 0.30,                # floor on the peg's point share
    'max_peg_frac': 0.55,
    'noise': (1e-3, 5e-3),               # isotropic sensor noise sigma
    'trans_scale': 1.0,                  # global pose translation scale
    # labels -- all radii are PHYSICAL lengths, never order statistics, so the
    # estimators converge as the sampling is refined
    'sigma_c': 0.09,                     # contact spring radial scale
    'contact_radius': None,              # hard support; None -> 3 * sigma_c
    'contact_candidates': 160,           # per-anchor budget; must cover r_c
    'body_sigma': 0.05,                  # background radial scale
    'body_radius': None,                 # hard support; None -> 3 * body_sigma
    'body_candidates': 256,              # per-anchor budget; must cover r_b
    'lambda_body': 0.005,                # contact ~17x body when touching
}


def make_cfg(**overrides):
    cfg = dict(DEFAULT_CFG)
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise KeyError(f'unknown peg-hole cfg keys: {sorted(unknown)}')
    cfg.update(overrides)
    return cfg


# ------------------------------------------------------------------ profiles
class ConvexProfile:
    """Convex polygon cross-section, CCW vertices [M, 2], origin inside."""

    def __init__(self, V):
        self.V = V
        e = torch.roll(V, -1, dims=0) - V
        self.edge_start = V
        self.edge_vec = e
        self.edge_len = e.norm(dim=-1)
        self.perimeter = self.edge_len.sum()
        # CCW boundary -> outward normal is the edge vector rotated by -90 deg
        self.edge_normal = torch.stack([e[:, 1], -e[:, 0]], dim=-1) \
            / self.edge_len[:, None]
        x, y = V[:, 0], V[:, 1]
        self.area = 0.5 * (x * torch.roll(y, -1) - torch.roll(x, -1) * y).sum()

    def scaled(self, s):
        return ConvexProfile(self.V * s)

    def rotated(self, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        R = torch.tensor([[c, -s], [s, c]], dtype=self.V.dtype)
        return ConvexProfile(self.V @ R.T)

    def signed_margin(self, pts):
        """[K, 2] -> [K]: distance to the boundary, positive inside (convex:
        min over edge planes; sign is exact, magnitude exact for interior)."""
        d = (pts[:, None, :] - self.edge_start[None]) \
            * self.edge_normal[None]
        return -(d.sum(-1)).amax(dim=-1)

    def contains(self, pts, margin=0.0):
        return self.signed_margin(pts) >= margin

    def inradius(self):
        return self.signed_margin(self.V.new_zeros(1, 2))[0]

    def sample_boundary(self, n, gen):
        if n == 0:
            return self.V.new_zeros(0, 2)
        idx = torch.multinomial(self.edge_len, n, replacement=True,
                                generator=gen)
        t = torch.rand(n, 1, generator=gen, dtype=self.V.dtype)
        return self.edge_start[idx] + t * self.edge_vec[idx]

    def sample_interior(self, n, gen):
        """Uniform in the polygon via a triangle fan from the vertex mean."""
        if n == 0:
            return self.V.new_zeros(0, 2)
        c = self.V.mean(dim=0)
        A, B = self.V, torch.roll(self.V, -1, dims=0)
        tri_area = 0.5 * ((A[:, 0] - c[0]) * (B[:, 1] - c[1])
                          - (B[:, 0] - c[0]) * (A[:, 1] - c[1])).abs()
        idx = torch.multinomial(tri_area, n, replacement=True, generator=gen)
        u = torch.rand(n, 1, generator=gen, dtype=self.V.dtype).sqrt()
        v = torch.rand(n, 1, generator=gen, dtype=self.V.dtype)
        return (1 - u) * c + u * ((1 - v) * A[idx] + v * B[idx])


def _regular_poly(r, n, dtype):
    th = torch.arange(n, dtype=dtype) * (2 * math.pi / n)
    return torch.stack([r * th.cos(), r * th.sin()], dim=-1)


def _clip_halfplane(V, normal, offset):
    """Keep the part of convex polygon V with <p, normal> <= offset."""
    d = V @ normal - offset
    keep = d <= 0
    out, M = [], V.shape[0]
    for i in range(M):
        j = (i + 1) % M
        if keep[i]:
            out.append(V[i])
        if keep[i] != keep[j]:
            t = d[i] / (d[i] - d[j])
            out.append(V[i] + t * (V[j] - V[i]))
    return torch.stack(out)


def make_profile(kind, r, aux, dtype=torch.float64, m=128):
    """Profile factory.  ``r`` is the circumradius scale, ``aux`` in [0, 1]
    controls the shape-internal aspect (rect/ellipse ratio, dee flat size)."""
    if kind == 'circle':
        return ConvexProfile(_regular_poly(r, m, dtype))
    if kind == 'triangle':
        return ConvexProfile(_regular_poly(r, 3, dtype))
    if kind == 'square':
        return ConvexProfile(_regular_poly(r, 4, dtype))
    if kind == 'hexagon':
        return ConvexProfile(_regular_poly(r, 6, dtype))
    if kind == 'rect':
        b = r * (0.45 + 0.35 * aux)
        V = torch.tensor([[r, -b], [r, b], [-r, b], [-r, -b]], dtype=dtype)
        return ConvexProfile(V)
    if kind == 'ellipse':
        b = r * (0.5 + 0.35 * aux)
        V = _regular_poly(1.0, m, dtype)
        return ConvexProfile(V * torch.tensor([r, b], dtype=dtype))
    if kind == 'dee':
        cut = r * (0.4 + 0.3 * aux)
        V = _clip_halfplane(_regular_poly(r, m, dtype),
                            torch.tensor([1.0, 0.0], dtype=dtype), cut)
        return ConvexProfile(V - V.mean(dim=0, keepdim=True))
    raise ValueError(f'unknown profile kind {kind!r}')


# ------------------------------------------------------------------ rotations
def _rotz(a, dtype):
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], dtype=dtype)


def _rot_axis(axis, angle, dtype):
    """Rodrigues rotation about a unit 3-vector."""
    ax = torch.tensor(axis, dtype=dtype)
    K = torch.tensor([[0., -ax[2], ax[1]],
                      [ax[2], 0., -ax[0]],
                      [-ax[1], ax[0], 0.]], dtype=dtype)
    return torch.eye(3, dtype=dtype) + math.sin(angle) * K \
        + (1 - math.cos(angle)) * (K @ K)


def _peg_rotation(yaw, tilt, tilt_dir, dtype):
    """Profile yaw about +z, then a tilt about a horizontal axis."""
    Rt = _rot_axis((math.cos(tilt_dir), math.sin(tilt_dir), 0.), tilt, dtype)
    return Rt @ _rotz(yaw, dtype)


# ------------------------------------------------------------- scene sampling
def _u(gen, lo, hi):
    return lo + (hi - lo) * torch.rand((), generator=gen).item()


def _chamfer_scale(zeta, ch, s0):
    """Cross-section scale of the peg at local height zeta (tip at 0)."""
    if ch <= 0:
        return 1.0
    t = min(max(zeta / ch, 0.0), 1.0)
    return s0 + (1.0 - s0) * t


def _peg_rings(profile, spec, zetas, dtype):
    """World-frame xy of the peg outline at local heights ``zetas``: used for
    the no-penetration test.  Returns [len(zetas) * M, 3] world points."""
    R, p = spec['R_peg'], spec['p_peg']
    pts = []
    for z in zetas:
        s = _chamfer_scale(z, spec['chamfer_h'], spec['chamfer_s0'])
        V = profile.V * s
        ring = torch.cat([V, V.new_full((V.shape[0], 1), z)], dim=-1)
        pts.append(ring)
    pts = torch.cat(pts)
    return pts @ R.T + p


def _lowest_ring_zetas(spec):
    zs = [0.0, spec['chamfer_h'], spec['H']]
    return [z for i, z in enumerate(zs) if z <= spec['H'] and z >= 0
            and (i == 0 or z > 0)]


def _sample_pose(spec, hole, gen, cfg, dtype):
    """Fill spec with (R_peg, p_peg) for its stage; explicit feasibility."""
    stage = spec['stage']
    r_hole = (1.0 + spec['clearance']) * spec['r_peg']
    reach = spec['half_w'] - 1.2 * spec['r_peg']

    if stage in ('free', 'search'):
        if stage == 'free':
            tilt = math.radians(_u(gen, 0.0, cfg['free_tilt_deg']))
            gap = _u(gen, *cfg['free_gap'])
            off_mag = abs(torch.randn((), generator=gen).item()) \
                * 1.5 * r_hole
        else:
            tilt = math.radians(_u(gen, 0.0, cfg['search_tilt_deg']))
            gap = _u(gen, *cfg['search_gap'])
            off_mag = _u(gen, *cfg['search_offset']) * r_hole
        off_mag = min(off_mag, max(reach, 0.2))
        yaw = _u(gen, 0.0, 2 * math.pi)
        odir = _u(gen, 0.0, 2 * math.pi)
        spec['R_peg'] = _peg_rotation(yaw, tilt, _u(gen, 0, 2 * math.pi),
                                      dtype)
        spec['p_peg'] = torch.tensor([off_mag * math.cos(odir),
                                      off_mag * math.sin(odir), 0.0],
                                     dtype=dtype)
        # place the lowest peg point exactly ``gap`` above the plate top
        rings = _peg_rings(spec['profile'], spec,
                           _lowest_ring_zetas(spec), dtype)
        spec['p_peg'][2] = gap - rings[:, 2].min().item()
        spec['depth'] = 0.0
        spec['tilt'] = tilt
        spec['yaw_mismatch'] = float('nan')
        return spec

    # ---- insert: rejection sampling inside the clearance budget
    d_max = min(0.9 * spec['H'], 1.15 * spec['T'])
    depth = _u(gen, *cfg['insert_depth']) * d_max
    c_abs = spec['clearance'] * spec['profile'].inradius().item()
    margin = cfg['insert_margin'] * c_abs
    shrink = 1.0
    for _ in range(40):
        off_mag = 0.95 * c_abs * _u(gen, 0.2, 1.0) * shrink
        odir = _u(gen, 0.0, 2 * math.pi)
        if spec['kind'] == 'circle':
            dyaw = _u(gen, 0.0, 2 * math.pi)
        else:
            dyaw = 0.5 * (spec['clearance'] * shrink) \
                * torch.randn((), generator=gen).item()
        tilt = abs(torch.randn((), generator=gen).item()) \
            * 0.4 * (c_abs / max(depth, 0.1)) * shrink
        tilt = min(tilt, math.radians(3.0))
        spec['R_peg'] = _peg_rotation(spec['hole_yaw'] + dyaw, tilt,
                                      _u(gen, 0, 2 * math.pi), dtype)
        spec['p_peg'] = torch.tensor([off_mag * math.cos(odir),
                                      off_mag * math.sin(odir), -depth],
                                     dtype=dtype)
        # Outline must stay inside the hole wherever it crosses the plate
        # band z in [-T, 0].  The surface between two outline rings is ruled,
        # so convexity reduces the test to rings that BRACKET the band; the
        # 0.08 / 0.05 slack keeps the bracket valid under the <= 3 deg tilt.
        z_lo = max(0.0, depth - spec['T'] - 0.05)
        z_hi = min(depth + 0.08, spec['H'])
        zetas = sorted({z_lo, min(max(spec['chamfer_h'], z_lo), z_hi), z_hi})
        rings = _peg_rings(spec['profile'], spec, zetas, dtype)
        if hole.contains(rings[:, :2], margin=margin).all():
            spec['depth'] = depth
            spec['tilt'] = tilt
            spec['yaw_mismatch'] = dyaw
            return spec
        shrink *= 0.7
    # fall back to the always-feasible centered pose
    spec['R_peg'] = _rotz(spec['hole_yaw'], dtype)
    spec['p_peg'] = torch.tensor([0.0, 0.0, -depth], dtype=dtype)
    spec['depth'], spec['tilt'], spec['yaw_mismatch'] = depth, 0.0, 0.0
    return spec


def sample_scene_spec(gen, cfg, dtype=torch.float64):
    """Randomize one scene: profile, plate, clearance, chamfer, stage, pose."""
    pw = torch.tensor(cfg['profile_weights'], dtype=torch.float64)
    kind_i = torch.multinomial(pw, 1, generator=gen).item()
    kind = PROFILE_TYPES[kind_i]
    r_peg = _u(gen, *cfg['peg_radius'])
    profile = make_profile(kind, r_peg, _u(gen, 0.0, 1.0), dtype)
    clearance = _u(gen, *cfg['clearance'])
    hole_yaw = _u(gen, 0.0, 2 * math.pi)

    spec = {
        'kind': kind, 'kind_i': kind_i, 'r_peg': r_peg,
        # base axes: the peg profile is NOT pre-rotated -- all yaw lives in
        # R_peg, while the hole polygon below carries hole_yaw explicitly.
        'profile': profile,
        'clearance': clearance, 'hole_yaw': hole_yaw,
        'H': _u(gen, *cfg['peg_height']),
        'Wx': _u(gen, *cfg['plate_width']),
        'Wy': _u(gen, *cfg['plate_width']),
        'T': _u(gen, *cfg['plate_thickness']),
        'noise': _u(gen, *cfg['noise']),
    }
    spec['half_w'] = 0.5 * min(spec['Wx'], spec['Wy'])
    if torch.rand((), generator=gen).item() < cfg['chamfer_prob']:
        spec['chamfer_h'] = _u(gen, *cfg['chamfer_height']) * spec['H']
        spec['chamfer_s0'] = _u(gen, *cfg['chamfer_scale'])
    else:
        spec['chamfer_h'], spec['chamfer_s0'] = 0.0, 1.0
    sp = torch.tensor(cfg['stage_probs'], dtype=torch.float64)
    spec['stage'] = STAGES[torch.multinomial(sp, 1, generator=gen).item()]
    # the hole profile shares the peg's base axes, rotated by hole_yaw
    hole = profile.scaled(1.0 + clearance).rotated(hole_yaw)
    spec['hole'] = hole
    return _sample_pose(spec, hole, gen, cfg, dtype)


def _sample_counts(weights, n, gen):
    idx = torch.multinomial(weights, n, replacement=True, generator=gen)
    return torch.bincount(idx, minlength=weights.shape[0])


def _plate_points(spec, n, gen, dtype):
    """Area-weighted uniform samples of the plate surface (canonical frame)."""
    Wx, Wy, T = spec['Wx'], spec['Wy'], spec['T']
    hole = spec['hole']
    a_rect = Wx * Wy
    a_face = a_rect - hole.area.item()
    areas = torch.tensor([a_face, a_face,                     # top, bottom
                          Wx * T, Wx * T, Wy * T, Wy * T,     # outer sides
                          hole.perimeter.item() * T],         # hole wall
                         dtype=torch.float64)
    cnt = _sample_counts(areas, n, gen)
    out = []

    def rect_face(m, z):
        if m == 0:
            return torch.zeros(0, 3, dtype=dtype)
        acc = []
        need = m
        while need > 0:
            draw = max(2 * need, 64)
            xy = (torch.rand(draw, 2, generator=gen, dtype=dtype) - 0.5) \
                * torch.tensor([Wx, Wy], dtype=dtype)
            xy = xy[~hole.contains(xy, margin=0.0)]
            acc.append(xy[:need])
            need -= acc[-1].shape[0]
        xy = torch.cat(acc)
        return torch.cat([xy, xy.new_full((m, 1), z)], dim=-1)

    out.append(rect_face(int(cnt[0]), 0.0))
    out.append(rect_face(int(cnt[1]), -T))
    for i, (axis, sign) in enumerate([(1, 1), (1, -1), (0, 1), (0, -1)]):
        m = int(cnt[2 + i])
        p = torch.empty(m, 3, dtype=dtype)
        free = torch.rand(m, generator=gen, dtype=dtype)
        p[:, 1 - axis] = (free - 0.5) * (Wx if axis == 1 else Wy)
        p[:, axis] = sign * 0.5 * (Wy if axis == 1 else Wx)
        p[:, 2] = -T * torch.rand(m, generator=gen, dtype=dtype)
        out.append(p)
    m = int(cnt[6])
    b = hole.sample_boundary(m, gen)
    z = -T * torch.rand(m, 1, generator=gen, dtype=dtype)
    out.append(torch.cat([b, z], dim=-1))
    return torch.cat(out)


def _peg_points(spec, n, gen, dtype):
    """Area-weighted uniform samples of the peg surface (canonical frame)."""
    prof, H = spec['profile'], spec['H']
    ch, s0 = spec['chamfer_h'], spec['chamfer_s0']
    A = prof.area.item()
    per = prof.perimeter.item()
    r_mean = prof.V.norm(dim=-1).mean().item()
    slant = math.sqrt(ch ** 2 + ((1 - s0) * r_mean) ** 2)
    areas = torch.tensor([A * s0 ** 2,                        # tip cap
                          0.5 * (1 + s0) * per * slant,       # chamfer band
                          per * (H - ch),                     # lateral wall
                          A],                                 # top cap
                         dtype=torch.float64).clamp_min(0.0)
    cnt = _sample_counts(areas.clamp_min(1e-12), n, gen)
    parts = []
    m = int(cnt[0])
    tip = prof.sample_interior(m, gen) * s0
    parts.append(torch.cat([tip, tip.new_zeros(m, 1)], dim=-1))
    m = int(cnt[1])
    if m > 0:
        # rejection on z so density tracks the local perimeter ~ scale(z)
        zs = []
        need = m
        while need > 0:
            t = torch.rand(2 * need + 32, generator=gen, dtype=dtype)
            acc = torch.rand(t.shape[0], generator=gen, dtype=dtype) \
                <= (s0 + (1 - s0) * t)
            zs.append(t[acc][:need])
            need -= zs[-1].shape[0]
        t = torch.cat(zs)
        b = prof.sample_boundary(m, gen)
        scale = (s0 + (1 - s0) * t)[:, None]
        parts.append(torch.cat([b * scale, (t * ch)[:, None]], dim=-1))
    m = int(cnt[2])
    b = prof.sample_boundary(m, gen)
    z = ch + (H - ch) * torch.rand(m, 1, generator=gen, dtype=dtype)
    parts.append(torch.cat([b, z], dim=-1))
    m = int(cnt[3])
    top = prof.sample_interior(m, gen)
    parts.append(torch.cat([top, top.new_full((m, 1), H)], dim=-1))
    pts = torch.cat(parts)
    return pts @ spec['R_peg'].T + spec['p_peg']


def build_scene_cloud(spec, n_points, gen, cfg, dtype=torch.float64):
    """One canonical-frame cloud: [N, 3] points and [N] part mask (1=peg)."""
    prof, H = spec['profile'], spec['H']
    ch, s0 = spec['chamfer_h'], spec['chamfer_s0']
    a_peg = (prof.area.item() * (1 + s0 ** 2)
             + prof.perimeter.item() * (H - ch)
             + 0.5 * (1 + s0) * prof.perimeter.item()
             * math.sqrt(ch ** 2 + ((1 - s0) * prof.V.norm(dim=-1)
                                    .mean().item()) ** 2))
    a_plate = 2 * (spec['Wx'] * spec['Wy'] - spec['hole'].area.item()) \
        + 2 * (spec['Wx'] + spec['Wy']) * spec['T'] \
        + spec['hole'].perimeter.item() * spec['T']
    frac = a_peg / (a_peg + a_plate)
    frac = min(max(frac, cfg['min_peg_frac']), cfg['max_peg_frac'])
    n_peg = int(round(n_points * frac))
    n_plate = n_points - n_peg
    P = torch.cat([_plate_points(spec, n_plate, gen, dtype),
                   _peg_points(spec, n_peg, gen, dtype)])
    part = torch.cat([torch.zeros(n_plate, dtype=torch.uint8),
                      torch.ones(n_peg, dtype=torch.uint8)])
    perm = torch.randperm(n_points, generator=gen)
    # the TRUE surface areas -- the label's Monte-Carlo weights, not derived
    # from the sampled points (which would reintroduce a sampling dependence)
    return P[perm], part[perm], a_peg, a_plate


# ------------------------------------------------------------------- labels
def _pair_second_moment(P, mask, sigma, radius, candidate_k, chunk, label):
    """(1/pairs) sum over admissible pairs inside ``radius`` of  w w^T.

    ``mask``: [S, N, N] bool, which ordered pairs (anchor i, partner j) may
    contribute.  Returns the UNNORMALISED sum [S, 6, 6]; the caller supplies
    the area weights.  Every pair inside the hard radius is counted -- the
    per-anchor ``candidate_k`` is only a materialisation budget and a
    :class:`ValueError` is raised if it is too small, because a truncated
    label would silently depend on the sampling again.
    """
    S, N, _ = P.shape
    cap = int(min(candidate_k, N - 1))
    out = P.new_zeros(S, 6, 6)
    worst = 0
    for s0 in range(0, S, chunk):
        sl = slice(s0, min(s0 + chunk, S))
        Pc, mc = P[sl], mask[sl]
        dist = torch.cdist(Pc, Pc,
                           compute_mode='donot_use_mm_for_euclid_dist')
        adm = mc & (dist < radius)
        # The radius is a FIXED physical length, so a high in-radius count is
        # the geometry talking, not a misconfiguration: materialise exactly
        # what is there (bounded by candidate_k, which then errors below).
        need = int(adm.sum(-1).max().item())
        worst = max(worst, need)
        k = max(1, min(need, cap))
        dist = dist.masked_fill(~adm, 1e12)
        d, idx = dist.topk(k, dim=-1, largest=False)             # [b, N, k]
        nbr = torch.gather(Pc.unsqueeze(2).expand(-1, -1, k, -1), 1,
                           idx.unsqueeze(-1).expand(-1, -1, k, 3))
        vec = nbr - Pc.unsqueeze(2)
        sq = vec.square().sum(-1)
        u = vec / sq.clamp_min(1e-18).sqrt().unsqueeze(-1)
        m = torch.cross(Pc.unsqueeze(2).expand_as(u), u, dim=-1)
        w = torch.cat([u, m], dim=-1)                            # [b, N, k, 6]
        q = (d / radius).clamp(max=1.0)
        window = (1.0 - q).clamp_min(0.0).pow(4) * (1.0 + 4.0 * q)
        kw = torch.exp(-sq / (2.0 * sigma ** 2)) * window * (d < radius)
        out[sl] = torch.einsum('bnk,bnki,bnkj->bij', kw, w, w)
    if worst > cap:
        raise ValueError(
            f'{label}: up to {worst} partners lie inside radius {radius:.3f} '
            f'but the budget caps materialisation at {cap}.  Raise it -- a '
            'truncated label is a function of the sampling, not the geometry.')
    return 0.5 * (out + out.transpose(-1, -2))


def peg_contact_K(P, part, area_peg, area_plate, sigma_c=0.09,
                  contact_radius=None, candidate_k=160, chunk=8):
    """Area-weighted cross-part contact stiffness, [S, 6, 6] in [f; m] order.

    ``area_peg`` / ``area_plate``: [S] true surface areas of the two parts, so
    that a point stands for A / N of surface and the sum is the Monte-Carlo
    estimate of the double surface integral (module docstring).  This is what
    makes the label a property of the SCENE rather than of its sampling.
    """
    if contact_radius is None:
        contact_radius = 3.0 * sigma_c
    peg = part.bool()
    n_peg = peg.sum(-1).to(P.dtype)
    n_plate = (~peg).sum(-1).to(P.dtype)
    if (n_peg < 1).any() or (n_plate < 1).any():
        raise ValueError('each sample needs at least one point of each part')
    mask = peg.unsqueeze(-1) & ~peg.unsqueeze(-2)         # peg anchor -> plate
    K = _pair_second_moment(P, mask, sigma_c, contact_radius, candidate_k,
                            chunk, 'peg_contact_K')
    dA = (area_peg.to(P) / n_peg) * (area_plate.to(P) / n_plate)
    return K * dA[:, None, None]


def body_K_areas(P, area_total, sigma_b=0.05, body_radius=None,
                 candidate_k=256, chunk=8):
    """Area-weighted all-pairs background second moment, [S, 6, 6].

    Same estimator with the cross-part restriction dropped, so it is full rank
    for any cloud and keeps ``K_contact + lambda * K_body`` SPD.
    """
    if body_radius is None:
        body_radius = 3.0 * sigma_b
    S, N, _ = P.shape
    mask = ~torch.eye(N, dtype=torch.bool, device=P.device).expand(S, N, N)
    K = _pair_second_moment(P, mask, sigma_b, body_radius, candidate_k,
                            chunk, 'body_K')
    dA = (area_total.to(P) / N) ** 2
    return K * dA[:, None, None]


def compose_K(K_contact, K_body, lambda_body):
    return K_contact + lambda_body * K_body


# ------------------------------------------------------------------ pipeline
_SPEC_SCALARS = ('r_peg', 'H', 'T', 'clearance', 'depth', 'tilt',
                 'yaw_mismatch', 'noise')


def generate_batch(n_samples, n_points, seed, cfg=None, device='cpu',
                   dtype=torch.float64, return_canonical=False):
    """Generate a labelled batch.

    Returns a dict of CPU tensors:
      points     [S, N, 3] float32 -- posed, noisy cloud (storage precision)
      part       [S, N]    uint8   -- 1 = peg, 0 = plate
      K_contact  [S, 6, 6] float64 -- computed FROM the float32-rounded cloud
      K_body     [S, 6, 6] float64
      area_peg / area_plate [S] float32 -- the label's Monte-Carlo weights
      stage / profile [S] uint8, per-sample scalars [S] float32

    The float32 rounding happens BEFORE the labels so that stored labels are
    exactly the label function of the stored points.  With
    ``return_canonical`` the pre-pose, pre-noise float64 clouds and the spec
    list are attached (used by the no-penetration tests).
    """
    cfg = cfg or DEFAULT_CFG
    gen = torch.Generator().manual_seed(seed)
    pts, parts, specs, areas = [], [], [], []
    for _ in range(n_samples):
        spec = sample_scene_spec(gen, cfg, dtype)
        P, part, a_peg, a_plate = build_scene_cloud(spec, n_points, gen, cfg,
                                                    dtype)
        pts.append(P)
        parts.append(part)
        specs.append(spec)
        areas.append((a_peg, a_plate))
    P0 = torch.stack(pts)
    part = torch.stack(parts)
    area_peg = torch.tensor([a for a, _ in areas], dtype=dtype)
    area_plate = torch.tensor([b for _, b in areas], dtype=dtype)

    noise = torch.tensor([s['noise'] for s in specs], dtype=dtype)
    P = P0 + noise[:, None, None] * torch.randn(P0.shape, generator=gen,
                                                dtype=dtype)
    R = torch.stack([random_SO3(gen, dtype) for _ in range(n_samples)])
    t = torch.randn(n_samples, 1, 3, generator=gen, dtype=dtype) \
        * cfg['trans_scale']
    P = P @ R.transpose(-1, -2) + t
    P32 = P.to(torch.float32)

    Pd = P32.double().to(device)
    partd = part.to(device)
    Kc = peg_contact_K(Pd, partd, area_peg, area_plate,
                       sigma_c=cfg['sigma_c'],
                       contact_radius=cfg['contact_radius'],
                       candidate_k=cfg['contact_candidates'])
    Kb = body_K_areas(Pd, area_peg + area_plate, sigma_b=cfg['body_sigma'],
                      body_radius=cfg['body_radius'],
                      candidate_k=cfg['body_candidates'])
    out = {
        'points': P32,
        'part': part,
        'K_contact': Kc.cpu(),
        'K_body': Kb.cpu(),
        # float64: these are the label's exact Monte-Carlo weights, so
        # rounding them would break "stored K == label(stored points)"
        'area_peg': area_peg,
        'area_plate': area_plate,
        'stage': torch.tensor([STAGES.index(s['stage']) for s in specs],
                              dtype=torch.uint8),
        'profile': torch.tensor([s['kind_i'] for s in specs],
                                dtype=torch.uint8),
        'n_peg': part.sum(-1).to(torch.int32),
    }
    for name in _SPEC_SCALARS:
        out[name] = torch.tensor([float(s[name]) for s in specs],
                                 dtype=torch.float32)
    if return_canonical:
        out['canonical'] = P0
        out['specs'] = specs
    return out

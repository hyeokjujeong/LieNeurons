"""[SUPERSEDED gen-1]
Presentation GIF for experiment B: congruence-equivariant K under SE(3).

A peg-shaped point cloud is moved along a three-phase SE(3) trajectory
(pure rotation -> pure translation -> screw motion back to the identity).
At every frame the untrained ModelB (Pluecker encoder + LNLinear/LNLieBracket
backbone + Klein head) is re-run on the transformed cloud, and its output
K(T.P) is compared against the congruence transport Ad_T^{-T} K(P0) Ad_T^{-1}
of the frame-0 output.

K is drawn as an ellipsoid built from its top-3 eigenpairs: with
K = sum_i lambda_i w_i w_i^T (eigh, descending), the 3x3 matrix

    E = Pi_m ( sum_{i<=3} lambda_i w_i w_i^T ) Pi_m^T

is the MOMENT block of the rank-3 truncation (wrench storage is [f; m]).
The moment slot is the one that carries the translation coupling of the
coadjoint action, so E rotates rigidly under pure rotation and genuinely
deforms under translation -- exactly the congruence story.  (The force block
is the wrong choice for display: the f-f block of K is exactly invariant
under pure translation.)  Semi-axes are sqrt(eig E); the predicted cage is
drawn 1.5% larger than the model surface so that both remain visible where
they coincide.

The weight seed is chosen (from a scan) so the untrained K has a
well-conditioned top-3 spectrum: lambda_3/lambda_4 >= 4.7 along the whole
trajectory (no truncation crossing) and ellipsoid anisotropy <= 7.4.

Two frame variants are rendered from the same pipeline (see FRAME_TEXT):

  'spatial'  everything in the fixed frame {s}.  K_s(t) obeys the congruence
             law and the ellipsoid genuinely deforms under translation.
  'body'     the same K pulled back to a moving frame {b} welded to the cloud,
             K_b = Ad_{T_sb}^T K_s Ad_{T_sb}.  Equivariance collapses this to
             the constant K(P0), so the ellipsoid rides along rigidly and the
             top-3 truncation can never cross.  Read as: equivariance is the
             thing that does NOT happen.  The pullback uses the known T, so
             the naive head -- which gets the identical pullback and still
             drifts by O(1) once p != 0 -- is what makes the panel evidence.

Run from the repo root:
    python experiment/pc_se3_congruence/legacy/visualize_expB.py
Writes figs/expB_congruence{,_body}.gif plus four key-frame PNGs each.
"""
import os
import sys

sys.path.append('.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullLocator, ScalarFormatter

from experiment.pc_se3_congruence.se3_utils import (
    adjoint, adjoint_inv, scaled_err, transform_cloud)
from experiment.pc_se3_congruence.encoders import PlueckerEncoder
from experiment.pc_se3_congruence.models import ModelB, NaiveHeadNoKlein

torch.set_default_dtype(torch.float64)

# ------------------------------------------------------------------ palette
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
BASELINE = '#c3c2b7'
BLUE = '#2a78d6'      # series 1: point cloud
ORANGE = '#eb6834'    # series 2: K from the model (ellipsoid)
AQUA = '#1baf7a'      # series 3: congruence-predicted K (cage)
ORD_RAMP = ['#1c5cab', '#3987e5', '#86b6ef']   # ordinal blue 550/400/250

N_POINTS = 430
WEIGHT_SEED = 11      # picked for a well-conditioned top-3 spectrum
FPS = 20
AXIS_FLOOR = 0.08     # safety floor on displayed semi-axes (never triggers
                      # for the shipped seed: anisotropy stays <= 7.4)
CAGE_INFLATE = 1.015  # cage drawn 1.5% larger, stated on the figure


# ---------------------------------------------------------------- peg cloud
def make_peg(n_total=N_POINTS, seed=0):
    """Round peg on a square base plate, surface-sampled, ~1.85 units tall."""
    rng = np.random.default_rng(seed)
    half = 0.85                   # plate half-width
    z0, z1 = -0.925, -0.575       # plate bottom/top
    r_s, z2 = 0.32, 0.925         # shaft radius, shaft top

    def plate_top(n):
        out = []
        while len(out) < n:
            xy = rng.uniform(-half, half, size=(n, 2))
            out.extend(xy[np.hypot(xy[:, 0], xy[:, 1]) > r_s + 0.01].tolist())
        xy = np.array(out[:n])
        return np.column_stack([xy, np.full(n, z1)])

    def plate_bottom(n):
        xy = rng.uniform(-half, half, size=(n, 2))
        return np.column_stack([xy, np.full(n, z0)])

    def plate_sides(n):
        u = rng.uniform(-half, half, size=n)
        z = rng.uniform(z0, z1, size=n)
        side = rng.integers(0, 4, size=n)
        x = np.where(side == 0, half, np.where(side == 1, -half, u))
        y = np.where(side == 2, half, np.where(side == 3, -half, u))
        return np.column_stack([x, y, z])

    def shaft_side(n):
        th = rng.uniform(0, 2 * np.pi, size=n)
        z = rng.uniform(z1, z2, size=n)
        return np.column_stack([r_s * np.cos(th), r_s * np.sin(th), z])

    def shaft_cap(n):
        rr = r_s * np.sqrt(rng.uniform(0, 1, size=n))
        th = rng.uniform(0, 2 * np.pi, size=n)
        return np.column_stack([rr * np.cos(th), rr * np.sin(th), np.full(n, z2)])

    counts = {'top': 62, 'bottom': 66, 'sides': 60, 'shaft': 76, 'cap': 16}
    scale = n_total / sum(counts.values())
    counts = {k: max(4, int(round(v * scale))) for k, v in counts.items()}
    pts = np.concatenate([plate_top(counts['top']), plate_bottom(counts['bottom']),
                          plate_sides(counts['sides']), shaft_side(counts['shaft']),
                          shaft_cap(counts['cap'])])
    pts += rng.normal(0, 0.004, size=pts.shape)   # jitter: no kNN distance ties
    return torch.tensor(pts, dtype=torch.float64)


# ------------------------------------------------------------- SE(3) motion
def rodrigues(axis, theta):
    u = axis / np.linalg.norm(axis)
    ux = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(theta) * ux + (1 - np.cos(theta)) * (ux @ ux)


def smoothstep(s):
    return s * s * (3 - 2 * s)


AXIS = np.array([0.35, 1.0, 0.55])
P_MAX = np.array([1.5, 0.9, 0.7])
N1, N2, N3 = 36, 36, 48
PHASES = [(0, N1, 'phase 1 — pure rotation'),
          (N1, N1 + N2, 'phase 2 — pure translation'),
          (N1 + N2, N1 + N2 + N3, 'phase 3 — screw motion')]


def trajectory():
    """Rotate 0->120 deg, translate 0->p_max, screw back to the identity."""
    Ts = []
    for i in range(N1):
        s = smoothstep(i / (N1 - 1))
        Ts.append((rodrigues(AXIS, s * 2 * np.pi / 3), np.zeros(3)))
    R1 = rodrigues(AXIS, 2 * np.pi / 3)
    for i in range(N2):
        s = smoothstep(i / (N2 - 1))
        Ts.append((R1, s * P_MAX))
    for i in range(1, N3 + 1):
        s = smoothstep(i / N3)
        Ts.append((rodrigues(AXIS, (2 + 4 * s) * np.pi / 3), (1 - s) * P_MAX))
    return Ts


# ------------------------------------------------------------ K -> ellipsoid
def top3_moment_block(K):
    """Top-3 eigenpairs of K; E = moment block of the rank-3 truncation."""
    lam, W = torch.linalg.eigh(K)             # ascending
    lam3, W3 = lam[3:], W[:, 3:]              # top-3
    M = W3[3:6, :]                            # moment parts ([f; m] storage)
    return ((M * lam3) @ M.T).numpy(), lam.numpy()[::-1]


def ellipsoid_frame(E, scale):
    """Sorted principal axes (desc) and rotation for drawing, floored axes."""
    ev, V = np.linalg.eigh(E)
    ev, V = ev[::-1], V[:, ::-1]
    a = scale * np.sqrt(np.maximum(ev, 0.0))
    a = np.maximum(a, AXIS_FLOOR * a[0])
    return a, V


_SPH_U, _SPH_V = np.meshgrid(np.linspace(0, 2 * np.pi, 44),
                             np.linspace(0, np.pi, 26))
_SPHERE = np.stack([np.cos(_SPH_U) * np.sin(_SPH_V),
                    np.sin(_SPH_U) * np.sin(_SPH_V),
                    np.cos(_SPH_V)])


def ellipsoid_surface(a, V, center):
    pts = np.einsum('ij,jkl->ikl', V * a, _SPHERE)
    return pts[0] + center[0], pts[1] + center[1], pts[2] + center[2]


_RING_T = np.linspace(0, 2 * np.pi, 90)


def ellipsoid_rings(a, V, center):
    """Three principal-plane ellipses (the dashed cage)."""
    rings = []
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        pts = (np.outer(V[:, i] * a[i], np.cos(_RING_T)) +
               np.outer(V[:, j] * a[j], np.sin(_RING_T)))
        rings.append(pts + center[:, None])
    return rings


# ------------------------------------------------------------- precompute
def precompute(frame='spatial'):
    torch.manual_seed(WEIGHT_SEED)
    model = ModelB(PlueckerEncoder(k=8), (8, 16, 16, 8))
    naive = ModelB(PlueckerEncoder(k=8), (8, 16, 16, 8))
    naive.load_state_dict(model.state_dict())
    naive.head = NaiveHeadNoKlein()

    P0 = make_peg().unsqueeze(0)
    c0 = P0[0].mean(dim=0).numpy()
    with torch.no_grad():
        K0 = model(P0)[0]
        K0n = naive(P0)[0]

    data = []
    with torch.no_grad():
        for R, p in trajectory():
            Rt, pt = torch.tensor(R), torch.tensor(p)
            Pt = transform_cloud(P0, Rt, pt)
            Ai = adjoint_inv(Rt, pt)
            K_rec, K_pred = model(Pt)[0], Ai.T @ K0 @ Ai
            K_recn, K_predn = naive(Pt)[0], Ai.T @ K0n @ Ai
            if frame == 'body':
                # pull back to {b}: K_b = Ad_{T_sb}^T K_s Ad_{T_sb}.  For an
                # equivariant K the two Ad's cancel and K_pred collapses to the
                # constant K(P0) -- the frame-0 reference the cage now draws.
                A = adjoint(Rt, pt)
                K_rec, K_pred = A.T @ K_rec @ A, A.T @ K_pred @ A
                K_recn, K_predn = A.T @ K_recn @ A, A.T @ K_predn @ A
            E_rec, lam = top3_moment_block(K_rec)
            E_pred, _ = top3_moment_block(K_pred)
            if frame == 'body':
                # E lives in {b}; rotate into world coordinates to draw it
                # (a covariance-like 3x3, so R E R^T).  The translation is
                # carried by the ellipsoid center, as in the spatial variant.
                E_rec, E_pred = R @ E_rec @ R.T, R @ E_pred @ R.T
            data.append(dict(
                cloud=Pt[0].numpy(), center=R @ c0 + p,
                E_rec=E_rec, E_pred=E_pred, lam=lam,
                err=max(scaled_err(K_rec, K_pred), 1e-17),
                err_naive=max(scaled_err(K_recn, K_predn), 1e-17)))
    return data


# -------------------------------------------------------- per-frame wording
# err_label / err_label_alt / err_note name the two error curves; they are keys
# rather than literals so a counter-example render (broken model on screen) can
# relabel them without the panel lying about which curve is which.
_ERR_LABELS = dict(err_label='ModelB (Klein head)',
                   err_label_alt='naive head, no $Q$',
                   err_note='naive head (no $Q$): fails once $p\\neq 0$')

FRAME_TEXT = {
    'spatial': dict(
        **_ERR_LABELS, suffix='', ax_target=1.55,
        title='Experiment B — congruence-equivariant stiffness from a point cloud',
        subtitle=(r'$K(T\!\cdot\!P)\;=\;\mathrm{Ad}_T^{-\top}\,K(P)\,'
                  r'\mathrm{Ad}_T^{-1}$'
                  '   —  ellipsoid: top-3 eigenpairs of $K$ (moment part), '
                  'recomputed by the network at every frame'),
        lam_title=(r'top-3 eigenvalues of $K(t)$   (relative to '
                   r'$\lambda_1(0)$, log)'),
        lam_note1='rotation:\nspectrum invariant',
        lam_note2='translation:\nspectrum shifts',
        err_title=(r'equivariance error   '
                   r'$\|K_{\mathrm{model}}-K_{\mathrm{pred}}\|/\|K\|$'),
        leg_model=r'$K(T\!\cdot\!P)$  network output',
        leg_pred=(r'$\mathrm{Ad}_T^{-\top}K(P)\,\mathrm{Ad}_T^{-1}$'
                  '  predicted cage'),
        caption=('ellipsoid = moment block of  '
                 r'$\sum_{i\leq 3}\lambda_i w_i w_i^{\top}$'
                 ',  semi-axes  $\\sqrt{\\mathrm{eig}}$ ;  cage drawn 1.5% '
                 'larger so both stay visible where they coincide;  '
                 'untrained weights')),
    'body': dict(
        # the body-frame semi-axes are constant, so the global scale is no
        # longer pinned by the growing translated ellipsoid -- shrink the
        # target so the peg stays visible inside it.
        **_ERR_LABELS, suffix='_body', ax_target=0.95,
        title='Experiment B — the same stiffness read in the body frame {b}',
        subtitle=(r'$K_b(t)=\mathrm{Ad}_{T_{sb}}^{\top}K_s(t)\,'
                  r'\mathrm{Ad}_{T_{sb}}\;\equiv\;K(P_0)$'
                  '   —  equivariance makes it constant, so the ellipsoid '
                  'rides along welded to the cloud'),
        lam_title=(r'top-3 eigenvalues of $K_b(t)$   (relative to '
                   r'$\lambda_1(0)$, log)'),
        lam_note1='rotation:\nno change',
        lam_note2='translation:\nno change either',
        err_title=(r'drift from frame 0   '
                   r'$\|K_b(t)-K_b(0)\|/\|K\|$'),
        leg_model=r'$K_b(t)$  network output, pulled back',
        leg_pred=r'$K_b(0)=K(P_0)$  frame-0 reference',
        caption=('same construction, evaluated on $K_b$ and drawn in {b};  the '
                 'pullback uses the known $T$ — the naive head gets the '
                 'identical pullback and still drifts, which is what makes '
                 'the flat orange curve evidence;  untrained weights')),
}


# ------------------------------------------------------------------ figure
def draw_ground(ax, lims, zfloor):
    (x0, x1), (y0, y1) = lims[0], lims[1]
    for x in np.arange(np.ceil(x0), x1 + 1e-9):
        ax.plot([x, x], [y0, y1], [zfloor, zfloor], color=GRID, lw=0.6,
                alpha=0.9, zorder=0)
    for y in np.arange(np.ceil(y0), y1 + 1e-9):
        ax.plot([x0, x1], [y, y], [zfloor, zfloor], color=GRID, lw=0.6,
                alpha=0.9, zorder=0)
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [zfloor] * 5,
            color=BASELINE, lw=0.9, zorder=0)


def draw_scene(ax, d, lims, zfloor, path):
    ax.set_axis_off()
    ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
    ax.set_box_aspect([h - l for l, h in lims])
    ax.view_init(elev=16, azim=-52)
    draw_ground(ax, lims, zfloor)

    P = d['cloud']
    # ground shadow + centroid path
    ax.scatter(P[:, 0], P[:, 1], np.full(len(P), zfloor),
               s=4, c=MUTED, alpha=0.12, linewidths=0, depthshade=False)
    if len(path) > 1:
        pp = np.array(path)
        ax.plot(pp[:, 0], pp[:, 1], pp[:, 2], color=BASELINE, lw=1.0, alpha=0.9)
    # point cloud
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=6.5, c=BLUE, alpha=0.92,
               linewidths=0, depthshade=True)
    # model ellipsoid (surface + principal axes)
    scale = d['_scale']
    a, V = ellipsoid_frame(d['E_rec'], scale)
    X, Y, Z = ellipsoid_surface(a, V, d['center'])
    ax.plot_surface(X, Y, Z, color=ORANGE, alpha=0.30, linewidth=0,
                    antialiased=True, shade=True)
    for i in range(3):
        seg = np.outer(V[:, i] * a[i], [-1, 1]) + d['center'][:, None]
        ax.plot(seg[0], seg[1], seg[2], color=ORANGE, lw=1.6, alpha=0.9,
                solid_capstyle='round')
    # predicted cage, slightly inflated so it stays visible on the surface
    a_p, V_p = ellipsoid_frame(d['E_pred'], scale * CAGE_INFLATE)
    for ring in ellipsoid_rings(a_p, V_p, d['center']):
        ax.plot(ring[0], ring[1], ring[2], color=AQUA, lw=2.1, ls=(0, (4, 3)),
                alpha=1.0)


def main(frame='spatial', bare=False):
    """bare=True renders the 3D scene alone -- no title, panels, legend or
    caption -- for slides that carry their own text."""
    txt = FRAME_TEXT[frame]
    sfx = txt['suffix'] + ('_bare' if bare else '')
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs')
    os.makedirs(out_dir, exist_ok=True)

    data = precompute(frame)
    T = len(data)

    # one global scale: largest semi-axis over the trajectory -> ax_target units
    max_ax = max(np.sqrt(max(np.linalg.eigvalsh(d['E_rec']).max(), 0))
                 for d in data)
    scale = txt['ax_target'] / max_ax
    for d in data:
        d['_scale'] = scale

    # global limits over clouds and ellipsoids
    allp = [d['cloud'] for d in data]
    for d in data:
        a, V = ellipsoid_frame(d['E_rec'], scale)
        allp.append(d['center'][None, :] + (V * a).T)
        allp.append(d['center'][None, :] - (V * a).T)
    allp = np.concatenate(allp)
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    pad = 0.03 * (hi - lo).max()
    lims = [(l - pad, h + pad) for l, h in zip(lo, hi)]
    zfloor = lims[2][0]

    lam = np.array([d['lam'][:3] for d in data]) / data[0]['lam'][0]
    err = np.array([d['err'] for d in data])
    err_n = np.array([d['err_naive'] for d in data])
    tt = np.arange(T)

    # ---------------------------------------------------------------- layout
    if bare:
        # scene only: no title, no side panels, no legend, no caption
        fig = plt.figure(figsize=(7.6, 6.4), dpi=100)
        fig.patch.set_facecolor(SURFACE)
        ax3 = fig.add_axes([-0.06, -0.06, 1.12, 1.12], projection='3d')
        ax3.set_facecolor(SURFACE)
    else:
        fig = plt.figure(figsize=(12.6, 6.4), dpi=100)
        fig.patch.set_facecolor(SURFACE)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.62, 1],
                              height_ratios=[1, 1], left=0.015, right=0.965,
                              top=0.86, bottom=0.10, hspace=0.52, wspace=0.16)
        ax3 = fig.add_subplot(gs[:, 0], projection='3d')
        ax3.set_facecolor(SURFACE)
        ax_lam = fig.add_subplot(gs[0, 1])
        ax_err = fig.add_subplot(gs[1, 1])

        fig.text(0.015, 0.955, txt['title'], fontsize=14, fontweight='bold',
                 color=INK)
        fig.text(0.015, 0.905, txt['subtitle'], fontsize=10.5, color=INK2)

    # static right panels ---------------------------------------------------
    if not bare:
        for ax in (ax_lam, ax_err):
            ax.set_facecolor(SURFACE)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(colors=MUTED, labelsize=8, length=0)
            ax.grid(True, color=GRID, lw=0.5)
            ax.set_xlim(0, T - 1)
            ax.axvspan(N1, N1 + N2, color=GRID, alpha=0.35)

        for k in range(3):
            ax_lam.plot(tt, lam[:, k], color=ORD_RAMP[k], lw=1.8)
            ax_lam.annotate(rf'$\lambda_{k + 1}$', (tt[-1], lam[-1, k]),
                            xytext=(4, 0), textcoords='offset points',
                            fontsize=8.5, color=ORD_RAMP[k], va='center')
        ax_lam.set_yscale('log')
        ax_lam.set_ylim(lam.min() * 0.45, lam.max() * 2.1)
        ax_lam.set_yticks([0.1, 0.3, 1, 3])
        ax_lam.yaxis.set_minor_locator(NullLocator())
        fmt = ScalarFormatter()
        fmt.set_scientific(False)
        ax_lam.yaxis.set_major_formatter(fmt)
        ax_lam.set_title(txt['lam_title'], fontsize=9.5, color=INK,
                         loc='left', pad=6)
        ax_lam.text(N1 / 2, 0.05, txt['lam_note1'], fontsize=7.4,
                    color=INK2, ha='center', va='bottom', linespacing=1.3,
                    transform=ax_lam.get_xaxis_transform())
        ax_lam.text(N1 + N2 / 2, 0.05, txt['lam_note2'],
                    fontsize=7.4, color=INK2, ha='center', va='bottom',
                    linespacing=1.3, transform=ax_lam.get_xaxis_transform())

        ax_err.plot(tt, err, color=ORANGE, lw=1.8, label=txt['err_label'])
        ax_err.plot(tt, err_n, color=MUTED, lw=1.4, ls=(0, (4, 3)),
                    label=txt['err_label_alt'])
        ax_err.text(N1 + N2 * 0.75, 0.70, txt['err_note'], fontsize=7.5,
                    color=INK2, ha='center',
                    transform=ax_err.get_xaxis_transform())
        ax_err.text(T * 0.45, 3e-14, 'float64 round-off floor', fontsize=7.5,
                    color=INK2, ha='center')
        ax_err.set_yscale('log')
        ax_err.set_ylim(1e-17, 3e1)
        ax_err.set_yticks([1e-16, 1e-12, 1e-8, 1e-4, 1e0])
        ax_err.yaxis.set_minor_locator(NullLocator())
        ax_err.set_title(txt['err_title'], fontsize=9.5, color=INK,
                         loc='left', pad=6)
        ax_err.set_xlabel('frame', fontsize=8.5, color=MUTED)
        ax_err.legend(loc='center left', fontsize=7.5, frameon=False,
                      labelcolor=INK2, handlelength=1.8, borderaxespad=0.2)

        cur_lam = ax_lam.axvline(0, color=INK, lw=0.9, alpha=0.55)
        cur_err = ax_err.axvline(0, color=INK, lw=0.9, alpha=0.55)
        err_txt = ax_err.text(0.985, 0.30, '', transform=ax_err.transAxes,
                              fontsize=8, color=INK, ha='right', va='bottom')

        legend_items = [
            Line2D([], [], marker='o', ls='none', ms=5, color=BLUE,
                   label='point cloud  $T\\!\\cdot\\!P$'),
            Patch(facecolor=ORANGE, alpha=0.45, label=txt['leg_model']),
            Line2D([], [], color=AQUA, lw=1.8, ls=(0, (4, 3)),
                   label=txt['leg_pred'])]
        fig.text(0.015, 0.022, txt['caption'], fontsize=7.8, color=MUTED)

    # the phase caption is the one label the bare variant keeps
    phase_txt = (fig.text(0.045, 0.045, '', fontsize=13.5, color=INK,
                          fontweight='bold') if bare else
                 fig.text(0.03, 0.115, '', fontsize=11.5, color=INK,
                          fontweight='bold'))

    path_hist = []

    def draw_frame(idx):
        ax3.cla()
        d = data[idx]
        if idx == 0:
            path_hist.clear()
        if len(path_hist) == 0 or not np.allclose(path_hist[-1], d['center']):
            path_hist.append(d['center'])
        draw_scene(ax3, d, lims, zfloor, path_hist)
        for name in [n for (b0, b1, n) in PHASES if b0 <= idx < b1]:
            phase_txt.set_text(name)
        if bare:
            return []
        ax3.legend(handles=legend_items, loc='upper left',
                   bbox_to_anchor=(0.0, 0.99), fontsize=8.3, frameon=False,
                   labelcolor=INK2, handlelength=1.6, borderaxespad=0.0)
        cur_lam.set_xdata([idx, idx])
        cur_err.set_xdata([idx, idx])
        err_txt.set_text('err $< 10^{-16}$' if err[idx] < 1e-16
                         else f'err = {err[idx]:.0e}')
        return []

    # frame sequence with short holds at the phase boundaries
    seq = ([0] * 8 + list(range(N1)) + [N1 - 1] * 6 +
           list(range(N1, N1 + N2)) + [N1 + N2 - 1] * 6 +
           list(range(N1 + N2, T)))

    print(f'[{frame}{"/bare" if bare else ""}] rendering {len(seq)} frames '
          f'({T} unique) ...')
    anim = animation.FuncAnimation(fig, lambda i: draw_frame(seq[i]),
                                   frames=len(seq), interval=1000 / FPS,
                                   blit=False)
    gif_path = os.path.join(out_dir, f'expB_congruence{sfx}.gif')
    anim.save(gif_path, writer=animation.PillowWriter(fps=FPS))
    print('wrote', gif_path)

    # key-frame stills for slides
    for name, idx in [('still_0_start', 0), ('still_1_rotated', N1 - 1),
                      ('still_2_translated', N1 + N2 - 1),
                      ('still_3_screw', N1 + N2 + N3 // 2)]:
        path_hist.clear()
        for j in range(idx + 1):       # rebuild the path up to this frame
            c = data[j]['center']
            if len(path_hist) == 0 or not np.allclose(path_hist[-1], c):
                path_hist.append(c)
        draw_frame(idx)
        p = os.path.join(out_dir, f'expB_{name}{sfx}.png')
        fig.savefig(p, dpi=110, facecolor=SURFACE)
        print('wrote', p)
    plt.close(fig)


if __name__ == '__main__':
    for _frame in ('spatial', 'body'):
        for _bare in (False, True):
            main(_frame, _bare)

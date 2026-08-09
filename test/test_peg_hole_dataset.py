"""Tests for the peg-and-hole PCD -> stiffness dataset.

Covers: profile geometry, scene feasibility (no peg/plate penetration,
stage-consistent gaps), label algebra (congruence equivariance with nonzero
translation, permutation invariance, PSD/SPD), the float32 storage protocol
(stored labels are exactly the label function of stored points), sharded
generation with resume, and loader subsampling determinism.
"""
import math
import os

import numpy as np
import pytest
import torch

from data_loader.peg_hole_data_loader import (PegHoleDataset,
                                              load_peg_hole_split)
from experiment.pc_se3_congruence.peg_hole_synth import (STAGES, body_K_areas,
                                                         compose_K,
                                                         generate_batch,
                                                         make_cfg,
                                                         make_profile,
                                                         peg_contact_K)
from experiment.pc_se3_congruence.se3_utils import (coadjoint, random_SE3,
                                                    scaled_err,
                                                    transform_cloud)

DTYPE = torch.float64
N = 256


@pytest.fixture(scope='module')
def batch():
    return generate_batch(24, N, seed=11, device='cpu')


@pytest.fixture(scope='module')
def canonical_batch():
    cfg = make_cfg(noise=(0.0, 0.0), trans_scale=0.0)
    return generate_batch(24, N, seed=5, cfg=cfg, device='cpu',
                          return_canonical=True), cfg


# ------------------------------------------------------------------ geometry
def test_profile_geometry():
    circ = make_profile('circle', 0.3, 0.5)
    # inscribed 128-gon: relative area deficit (2 pi / m)^2 / 6 ~ 4e-4
    assert abs(circ.area.item() - math.pi * 0.09) < 1e-3 * circ.area.item()
    assert abs(circ.perimeter.item() - 2 * math.pi * 0.3) < 1e-3
    assert abs(circ.inradius().item() - 0.3) < 1e-3
    pts = torch.tensor([[0.0, 0.0], [0.29, 0.0], [0.31, 0.0]], dtype=DTYPE)
    m = circ.signed_margin(pts)
    assert m[0] > 0.29 and m[1] > 0 and m[2] < 0
    for kind in ('triangle', 'square', 'hexagon', 'rect', 'ellipse', 'dee'):
        prof = make_profile(kind, 0.3, 0.5)
        assert prof.area.item() > 0
        inner = prof.sample_interior(64, torch.Generator().manual_seed(0))
        assert prof.contains(inner).all()
        b = prof.sample_boundary(64, torch.Generator().manual_seed(0))
        assert prof.signed_margin(b).abs().max() < 1e-9


def test_scene_composition(batch):
    part = batch['part']
    frac = part.double().mean(dim=1)
    assert (frac >= 0.25).all() and (frac <= 0.6).all()
    assert (batch['n_peg'].long() == part.long().sum(-1)).all()
    assert batch['points'].dtype == torch.float32
    assert set(batch['stage'].tolist()) <= {0, 1, 2}


# --------------------------------------------------------------- feasibility
def test_no_penetration_and_stage_gaps(canonical_batch):
    (b, cfg) = canonical_batch
    for i, spec in enumerate(b['specs']):
        P = b['canonical'][i]
        peg = b['part'][i].bool()
        Pp = P[peg]
        inside_rect = ((Pp[:, 0].abs() <= spec['Wx'] / 2 - 1e-9)
                       & (Pp[:, 1].abs() <= spec['Wy'] / 2 - 1e-9)
                       & (Pp[:, 2] < -1e-9) & (Pp[:, 2] > -spec['T'] + 1e-9))
        in_hole = spec['hole'].contains(Pp[:, :2], margin=-1e-9)
        assert not (inside_rect & ~in_hole).any(), \
            f'sample {i} ({spec["stage"]}): peg penetrates the plate'
        z_min = Pp[:, 2].min().item()
        if spec['stage'] == 'free':
            assert z_min >= cfg['free_gap'][0] - 1e-6
        elif spec['stage'] == 'search':
            # geometric minimum lies in search_gap; the SAMPLED minimum can
            # sit slightly above it (finite surface sampling under tilt)
            assert cfg['search_gap'][0] - 1e-6 <= z_min \
                <= cfg['search_gap'][1] + 0.03
        else:
            assert z_min < 0 and spec['depth'] > 0


def test_contact_gap_by_stage(batch):
    P, part, stage = batch['points'].double(), batch['part'].bool(), \
        batch['stage']
    sigma_c = 0.09
    for i in range(P.shape[0]):
        gap = torch.cdist(P[i][part[i]], P[i][~part[i]]).min().item()
        if stage[i] == 0:                                   # free
            assert gap > 3 * sigma_c
        else:                                               # search / insert
            assert gap < 1.5 * sigma_c


# -------------------------------------------------------------------- labels
def test_label_equivariance(batch):
    P = batch['points'][:8].double()
    part = batch['part'][:8]
    ap = batch['area_peg'][:8].double()
    al = batch['area_plate'][:8].double()
    Kc = peg_contact_K(P, part, ap, al)
    Kb = body_K_areas(P, ap + al)
    gen = torch.Generator().manual_seed(3)
    for _ in range(3):
        R, p = random_SE3(1.0, gen)
        rho = coadjoint(R, p)
        Pt = transform_cloud(P, R, p)
        for K0, K1 in ((Kc, peg_contact_K(Pt, part, ap, al)),
                       (Kb, body_K_areas(Pt, ap + al))):
            err = scaled_err(K1, rho @ K0 @ rho.transpose(-1, -2))
            assert err < 5e-9, err


def test_label_permutation_invariance(batch):
    P = batch['points'][:8].double()
    part = batch['part'][:8]
    ap, al = batch['area_peg'][:8].double(), batch['area_plate'][:8].double()
    perm = torch.randperm(P.shape[1], generator=torch.Generator()
                          .manual_seed(1))
    K0 = peg_contact_K(P, part, ap, al)
    K1 = peg_contact_K(P[:, perm], part[:, perm], ap, al)
    assert scaled_err(K1, K0) < 1e-12
    assert scaled_err(body_K_areas(P[:, perm], ap + al),
                      body_K_areas(P, ap + al)) < 1e-12


def test_label_spd(batch):
    lam_c = torch.linalg.eigvalsh(batch['K_contact'])
    assert lam_c[:, 0].min() > -1e-12                       # PSD
    K = compose_K(batch['K_contact'], batch['K_body'], make_cfg()['lambda_body'])
    lam = torch.linalg.eigvalsh(K)
    assert lam[:, 0].min() > 1e-9                           # SPD
    assert (lam[:, -1] / lam[:, 0]).max() < 1e6


def test_contact_rank_semantics(batch):
    """No proximity => contact term EXACTLY 0 (hard Wendland support);
    insertion => O(1) wrench span."""
    nc = batch['K_contact'].norm(dim=(1, 2))
    stage = batch['stage']
    if (stage == 0).any():
        assert nc[stage == 0].max() == 0.0
    if (stage == 2).any():
        assert nc[stage == 2].median() > 1e-3


def test_generation_reproducible():
    a = generate_batch(4, N, seed=42, device='cpu')
    b = generate_batch(4, N, seed=42, device='cpu')
    assert torch.equal(a['points'], b['points'])
    assert torch.equal(a['K_contact'], b['K_contact'])
    c = generate_batch(4, N, seed=43, device='cpu')
    assert not torch.equal(a['points'], c['points'])


# ----------------------------------------------------- storage + loader
@pytest.fixture(scope='module')
def shard_dir(tmp_path_factory):
    import data_gen.gen_peg_hole_pcd as gen
    out = str(tmp_path_factory.mktemp('peg_hole'))
    args = type('A', (), dict(out=out, n_train=12, n_val=6, n_test=6,
                              n_points=N, shard_size=8, seed=7,
                              device='cpu'))()
    gen.generate(args, make_cfg())
    return out


def test_storage_exact_roundtrip(shard_dir):
    P, K, info = load_peg_hole_split(shard_dir, 'val', extras=True)
    assert P.shape == (6, N, 3) and K.shape == (6, 6, 6)
    cfg = make_cfg()
    ap, al = info['area_peg'].double(), info['area_plate'].double()
    Kc = peg_contact_K(P, info['part'], ap, al, sigma_c=cfg['sigma_c'],
                       candidate_k=cfg['contact_candidates'])
    Kb = body_K_areas(P, ap + al, sigma_b=cfg['body_sigma'],
                      candidate_k=cfg['body_candidates'])
    # float32 rounding happened BEFORE labelling, so this must be exact
    assert scaled_err(Kc, info['K_contact']) < 1e-14
    assert scaled_err(Kb, info['K_body']) < 1e-14
    assert torch.allclose(K, compose_K(info['K_contact'], info['K_body'],
                                       info['lambda_body']))


def test_generate_resume_skips_existing(shard_dir):
    import data_gen.gen_peg_hole_pcd as gen
    meta0 = gen.load_meta(shard_dir)
    mtimes = {f: os.path.getmtime(os.path.join(shard_dir, f))
              for f in os.listdir(shard_dir) if f.endswith('.npz')}
    args = type('A', (), dict(out=shard_dir, n_train=12, n_val=6, n_test=6,
                              n_points=N, shard_size=8, seed=7,
                              device='cpu'))()
    gen.generate(args, make_cfg())
    meta1 = gen.load_meta(shard_dir)
    assert meta0 == meta1
    for f, t in mtimes.items():
        assert os.path.getmtime(os.path.join(shard_dir, f)) == t


def test_relabel_makes_the_subsampled_target_a_function_of_the_input(shard_dir):
    """The point of relabel=True: with it, K is exactly the label function of
    the points the model receives; without it, K is the label of a cloud the
    model never sees, and the regression has an irreducible floor."""
    cfg = make_cfg()
    n_sub = N // 2
    P, K, info = load_peg_hole_split(shard_dir, 'val', n_points=n_sub, seed=1,
                                     extras=True)
    assert P.shape[1] == n_sub
    ap, al = info['area_peg'].double(), info['area_plate'].double()
    Kc = peg_contact_K(P, info['part'], ap, al, sigma_c=cfg['sigma_c'],
                       contact_radius=cfg['contact_radius'],
                       candidate_k=cfg['contact_candidates'])
    Kb = body_K_areas(P, ap + al, sigma_b=cfg['body_sigma'],
                      body_radius=cfg['body_radius'],
                      candidate_k=cfg['body_candidates'])
    assert scaled_err(Kc, info['K_contact']) < 1e-14
    assert scaled_err(Kb, info['K_body']) < 1e-14
    assert torch.allclose(K, compose_K(Kc, Kb, info['lambda_body']))

    # without relabelling the same points carry the FULL-cloud label, which is
    # a different matrix -- this is the ill-posed pairing, kept only for
    # comparison runs
    P0, K0, i0 = load_peg_hole_split(shard_dir, 'val', n_points=n_sub, seed=1,
                                     extras=True, relabel=False)
    assert torch.equal(P0, P)
    assert scaled_err(i0['K_contact'], info['K_contact']) > 1e-3

    # relabelling is a no-op when nothing is subsampled
    Pf, Kf = load_peg_hole_split(shard_dir, 'val')
    Pn, Kn = load_peg_hole_split(shard_dir, 'val', relabel=False)
    assert torch.equal(Pf, Pn) and torch.equal(Kf, Kn)


def test_relabel_cache_is_consistent(shard_dir):
    a = load_peg_hole_split(shard_dir, 'val', n_points=N // 2, seed=2)
    assert os.path.isdir(os.path.join(shard_dir, 'cache', f'n{N // 2}_seed2'))
    b = load_peg_hole_split(shard_dir, 'val', n_points=N // 2, seed=2)
    c = load_peg_hole_split(shard_dir, 'val', n_points=N // 2, seed=2,
                            cache=False)
    for x, y in ((b, a), (c, a)):
        assert torch.equal(x[0], y[0]) and torch.allclose(x[1], y[1])


def test_loader_subsample_and_dataset(shard_dir):
    P1, K1 = load_peg_hole_split(shard_dir, 'train', n=10, n_points=64,
                                 seed=3)
    P2, K2 = load_peg_hole_split(shard_dir, 'train', n=10, n_points=64,
                                 seed=3)
    assert torch.equal(P1, P2) and torch.equal(K1, K2)
    P3, _ = load_peg_hole_split(shard_dir, 'train', n=10, n_points=64,
                                seed=4)
    assert not torch.equal(P1, P3)
    assert P1.shape == (10, 64, 3)
    Pf, Kf = load_peg_hole_split(shard_dir, 'train', n=10)
    ds = PegHoleDataset(shard_dir, 'train')
    assert len(ds) == 12
    Pi, Ki, part_i = ds[9]
    assert torch.equal(Pi, Pf[9]) and torch.allclose(Ki, Kf[9])
    with pytest.raises(ValueError):
        load_peg_hole_split(shard_dir, 'train', n=999)

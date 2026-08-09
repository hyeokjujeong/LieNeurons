"""Structural tests for the pointwise wrench -> stiffness pipeline.

Axis convention under test (the stiffness matrix is the only K):
    P [B, N, 3] -> edge wrench [B, N, k, 6] -> X [B, C, 6, N] -> Z [B, H, 6, N]
    -> K [B, 6, 6].
"""
import pytest
import torch

from experiment.pc_se3_congruence.data_synth import (c2_clouds, lattice_clouds,
                                                     symmetric_clouds,
                                                     tetra_orbit_clouds)
from experiment.pc_se3_congruence.encoders import WrenchEdgeEncoder
from experiment.pc_se3_congruence.models import WrenchSecondMomentModel
from experiment.pc_se3_congruence.pointwise_graph import (build_local_graph,
                                                          degree_matched_radius,
                                                          wendland_c2)
from experiment.pc_se3_congruence.pointwise_models import (
    LocalWrenchSetEncoder, PointwiseStiffnessModel, force_pair, klein_pair)
from experiment.pc_se3_congruence.se3_utils import (coadjoint, random_SE3,
                                                    scaled_err,
                                                    transform_cloud)

DTYPE = torch.float64


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    # the shipping default: candidate_k only has to cover the support, and
    # a starved budget breaks tie-invariance for real (see the lattice test)
    kw.setdefault('candidate_k', 64)
    return PointwiseStiffnessModel(**kw).to(DTYPE).eval()


def _clouds(n=3, npts=32, seed=5):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, npts, 3, generator=gen, dtype=DTYPE), gen


# --------------------------------------------------------------- graph layer
def test_wendland_window_vanishes_with_zero_derivative_at_the_cutoff():
    q = torch.tensor([0.0, 0.5, 1.0, 1.5], dtype=DTYPE, requires_grad=True)
    w = wendland_c2(q)
    assert w[0].item() == pytest.approx(1.0)
    assert w[2].item() == 0.0 and w[3].item() == 0.0
    w.sum().backward()
    # Both value and slope vanish at q = 1: an edge leaves the support without
    # an O(1) jump in either the forward pass or the gradient.
    assert q.grad[2].abs().item() < 1e-12


def test_graph_support_is_invariant_and_reports_truncation():
    P, gen = _clouds()
    g0 = build_local_graph(P, candidate_k=64)
    R, p = random_SE3(1.0, gen, dtype=DTYPE)
    g1 = build_local_graph(transform_cloud(P, R, p), candidate_k=64)
    torch.testing.assert_close(g0.window, g1.window, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(g0.radius, g1.radius, rtol=1e-11, atol=1e-11)
    assert 0.0 <= g0.truncation_frac <= 1.0

    # A support that swallows every point must flag truncation.
    tight = build_local_graph(P, candidate_k=4, radius_mode='global_scale',
                              radius_alpha=10.0)
    assert tight.truncation_frac > 0.0


# ------------------------------------------- degree-matched radius (default)
def _surface_cloud(n, gen):
    """Points on a sphere: intrinsic dimension 2, where the (k/N)^(1/3)
    density correction of 'density_scaled' is wrong by construction."""
    v = torch.randn(2, n, 3, generator=gen, dtype=DTYPE)
    return v / v.norm(dim=-1, keepdim=True)


def _curve_cloud(n, gen):
    """Points along a helix: intrinsic dimension 1, the extreme case."""
    t = torch.linspace(0.0, 6.0, n, dtype=DTYPE)
    c = torch.stack([t.cos(), t.sin(), 0.3 * t], dim=-1)
    return c.unsqueeze(0).repeat(2, 1, 1) \
        + 1e-3 * torch.randn(2, n, 3, generator=gen, dtype=DTYPE)


@pytest.mark.parametrize('make,n', [
    (lambda n, g: torch.randn(2, n, 3, generator=g, dtype=DTYPE), 256),
    (_surface_cloud, 256),
    (_curve_cloud, 256),
])
def test_degree_matched_radius_hits_the_target_on_any_distribution(make, n):
    """The point of the mode: mean degree = target_k for a volumetric blob
    (intrinsic dim 3), a sphere (2) and a helix (1) alike -- the fixed-exponent
    modes cannot, since they hard-code one of those dimensions."""
    gen = torch.Generator().manual_seed(0)
    P = make(n, gen)
    for target in (8, 16, 32):
        # the budget has to scale with the target, not with N: density
        # heterogeneity puts the max in-support degree at ~3.2x the mean, so
        # 4x target_k is the working rule of thumb.
        g = build_local_graph(P, candidate_k=4 * target, target_k=target)
        assert abs(g.mean_degree - target) <= 0.15 * target, \
            f'target {target}, got {g.mean_degree}'
        assert g.truncation_frac == 0.0
        assert g.required_candidate_k <= 4 * target


def test_degree_matched_identity_holds_even_on_a_degenerate_spectrum():
    """A lattice has huge exact-tie shells, so a mean degree of exactly
    target_k is simply not attainable by ANY radius.  What the calibration
    still guarantees is the quantile sandwich

        mean count(d <  r)  <=  target_k  <=  mean count(d <= r),

    i.e. r is the tightest radius that does not undershoot.  The windowed
    degree lands on the lower branch because wendland is 0 at q = 1 and drops
    the boundary shell whole -- that gap is the tie-safety property itself.
    On non-degenerate clouds the two branches differ by 1/N and the sandwich
    collapses to the equality the docstring advertises."""
    gen = torch.Generator().manual_seed(0)
    P = lattice_clouds(2, 6, gen, dtype=DTYPE)
    N = P.shape[1]
    d = torch.cdist(P, P, compute_mode='donot_use_mm_for_euclid_dist')
    d = d + torch.eye(N, dtype=DTYPE) * 1e12
    for target in (8, 16, 32):
        r = degree_matched_radius(d, target)
        below = (d < r).sum(-1).to(DTYPE).mean().item()
        upto = (d <= r).sum(-1).to(DTYPE).mean().item()
        assert below <= target <= upto, f'{below} !<= {target} !<= {upto}'
        g = build_local_graph(P, candidate_k=64, target_k=target)
        assert abs(g.mean_degree - below) < 1e-9   # boundary shell dropped
        assert g.truncation_frac == 0.0

    # and on a generic cloud the sandwich really is tight
    Q = torch.randn(2, 200, 3, generator=gen, dtype=DTYPE)
    dq = torch.cdist(Q, Q, compute_mode='donot_use_mm_for_euclid_dist')
    dq = dq + torch.eye(200, dtype=DTYPE) * 1e12
    r = degree_matched_radius(dq, 16)
    assert (dq <= r).sum(-1).to(DTYPE).mean().item() - 16 < 2.0 / 200


def test_degree_matched_radius_is_invariant_and_relabelling_proof():
    gen = torch.Generator().manual_seed(1)
    P = torch.randn(2, 96, 3, generator=gen, dtype=DTYPE)
    g0 = build_local_graph(P, candidate_k=32)
    R, p = random_SE3(1.0, gen, dtype=DTYPE)
    g1 = build_local_graph(transform_cloud(P, R, p), candidate_k=32)
    perm = torch.randperm(P.shape[1], generator=gen)
    g2 = build_local_graph(P[:, perm], candidate_k=32)
    for g in (g1, g2):
        torch.testing.assert_close(g.radius, g0.radius, rtol=1e-12, atol=1e-12)
    # exact ties (a lattice) must not move the calibrated radius either
    L = lattice_clouds(2, 5, gen, dtype=DTYPE)
    gl0 = build_local_graph(L, candidate_k=32)
    gl1 = build_local_graph(L[:, torch.randperm(L.shape[1], generator=gen)],
                            candidate_k=32)
    torch.testing.assert_close(gl1.radius, gl0.radius, rtol=1e-12, atol=1e-12)


def test_required_candidate_k_is_the_actionable_budget():
    """candidate_k is a memory budget, not a model parameter: the graph
    reports exactly how large it has to be, and meeting it zeroes the
    truncation."""
    gen = torch.Generator().manual_seed(2)
    P = torch.randn(2, 256, 3, generator=gen, dtype=DTYPE)
    starved = build_local_graph(P, candidate_k=4, target_k=32)
    assert starved.truncation_frac > 0.0
    assert starved.candidate_k == 4 and starved.required_candidate_k > 4

    ok = build_local_graph(P, candidate_k=starved.required_candidate_k,
                           target_k=32)
    assert ok.truncation_frac == 0.0
    assert abs(ok.mean_degree - 32) <= 0.15 * 32


def test_candidate_budget_beyond_the_support_changes_nothing():
    """The reason the budget is only a budget: candidates outside the support
    have window 0, so any k >= required gives the SAME graph.  This is what
    makes a fixed, generous candidate_k safe."""
    gen = torch.Generator().manual_seed(3)
    P = torch.randn(2, 200, 3, generator=gen, dtype=DTYPE)
    ref = build_local_graph(P, candidate_k=64)
    assert ref.truncation_frac == 0.0
    need = ref.required_candidate_k
    for k in (need, 100, 199):
        g = build_local_graph(P, candidate_k=k)
        assert g.truncation_frac == 0.0
        torch.testing.assert_close(g.window[..., :need], ref.window[..., :need],
                                   rtol=0, atol=0)
        extra = g.window[..., need:]
        assert extra.numel() == 0 or extra.abs().max().item() == 0.0


def test_default_candidate_k_covers_the_default_radius():
    """The measured claim behind the default: at candidate_k = 64 the
    degree-matched radius truncates nothing, on every shape family in the
    suite."""
    gen = torch.Generator().manual_seed(4)
    clouds = [torch.randn(2, n, 3, generator=gen, dtype=DTYPE)
              for n in (32, 128, 512)]
    clouds += [_surface_cloud(512, gen), _curve_cloud(256, gen),
               lattice_clouds(2, 5, gen, dtype=DTYPE),
               symmetric_clouds(2, 128, gen, dtype=DTYPE),
               c2_clouds(2, 128, gen, dtype=DTYPE),
               tetra_orbit_clouds(2, 120, gen, dtype=DTYPE)]
    for P in clouds:
        g = build_local_graph(P)                      # all defaults
        assert g.truncation_frac == 0.0, \
            f'N={P.shape[1]} needs candidate_k {g.required_candidate_k}'
        assert g.required_candidate_k <= 64


def test_edge_wrench_obeys_the_coadjoint_law():
    P, gen = _clouds()
    enc = LocalWrenchSetEncoder().to(DTYPE)
    R, p = random_SE3(1.0, gen, dtype=DTYPE)
    Pt = transform_cloud(P, R, p)
    w0 = enc.edge_wrenches(P, build_local_graph(P, candidate_k=64))
    wt = enc.edge_wrenches(Pt, build_local_graph(Pt, candidate_k=64))
    rho = coadjoint(R, p)
    expected = torch.einsum('ij,bnkj->bnki', rho, w0)
    assert scaled_err(wt, expected) < 1e-12


# ---------------------------------------------------------------- invariants
def test_klein_and_force_pairings_are_coadjoint_invariants():
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(2, 5, 6, 7, generator=gen, dtype=DTYPE)
    y = torch.randn(2, 5, 6, 7, generator=gen, dtype=DTYPE)
    R, p = random_SE3(1.0, gen, dtype=DTYPE)
    rho = coadjoint(R, p)
    xa = torch.einsum('ij,bcjn->bcin', rho, x)
    ya = torch.einsum('ij,bcjn->bcin', rho, y)
    torch.testing.assert_close(klein_pair(xa, ya), klein_pair(x, y),
                               rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(force_pair(xa, ya), force_pair(x, y),
                               rtol=1e-11, atol=1e-11)


# -------------------------------------------------------- end-to-end pipeline
VARIANTS = [
    {},
    {'pool': 'basis', 'bracket': 'none'},
    {'pool': 'attention'},
    {'bracket': 'none'},
    {'bracket': 'pairwise', 'bracket_channels': 4},
    {'use_bracket_layers': False},
    {'gate': 'none'},
    {'gate': 'full'},
    {'use_global_context': False},
    {'message_passing': True},
    {'pool': 'sum'},
    {'pool': 'mean'},
    {'pool': 'basis', 'bracket': 'pairwise', 'bracket_channels': 4},
    {'normalize': 'beta'},
    {'normalize': 'one'},
    {'beta_mode': 'uniform'},
    {'use_force_invariant': True},
    {'radius_mode': 'density_scaled'},
    {'radius_mode': 'global_scale'},
    {'radius_mode': 'knn_adaptive'},
    {'radius_mode': 'knn_shell'},
    {'radius_mode': 'fixed', 'radius_value': 1.0},
]


@pytest.mark.parametrize('kw', VARIANTS)
def test_stiffness_is_congruence_equivariant_and_permutation_invariant(kw):
    P, gen = _clouds()
    model = _model(**kw)
    K = model(P)
    assert K.shape == (P.shape[0], 6, 6)
    torch.testing.assert_close(K, K.transpose(-1, -2), rtol=1e-12, atol=1e-14)

    R, p = random_SE3(1.0, gen, dtype=DTYPE)
    rho = coadjoint(R, p)
    got = model(transform_cloud(P, R, p))
    assert scaled_err(got, rho @ K @ rho.transpose(-1, -2)) < 1e-11

    perm = torch.randperm(P.shape[1], generator=gen)
    assert scaled_err(model(P[:, perm]), K) < 1e-11


def test_basis_pooling_has_no_encoder_parameters():
    """The simple default is a FIXED lift: channel c is the distance shell
    rho_c, and the learned radial kernel lives in the LNLinear that follows.
    If a parameter ever appears here, that claim is no longer true."""
    model = _model(pool='basis', bracket='none')
    assert list(model.set_encoder.parameters()) == []


def test_basis_pooling_weights_are_a_partition_over_distance():
    """Shell weights are non-negative, vanish outside the support, and their
    channel sum is a function of q alone -- so no anchor is silently starved."""
    P, _ = _clouds()
    model = _model(pool='basis', bracket='none')
    graph = model.build_graph(P)
    a = model.set_encoder.pooling_weights(graph)          # [B, N, k, C]
    assert (a >= 0).all()
    outside = graph.window == 0
    assert a[outside].abs().max() == 0.0
    inside = ~outside
    assert a.sum(-1)[inside].min() > 0


def test_factor_matrix_reproduces_the_stiffness():
    """K = L L^T exactly: L is emitted first, K is formed from it.  A Cholesky
    factor of an equivariant K would not itself be equivariant."""
    P, _ = _clouds()
    model = _model()
    L = model.factors(P)
    assert L.shape[1] == 6
    assert scaled_err(L @ L.transpose(-1, -2), model(P)) < 1e-12


def test_stiffness_is_psd():
    P, _ = _clouds()
    lam = torch.linalg.eigvalsh(_model()(P))
    assert (lam >= -1e-12 * lam[:, -1:].abs()).all()


@pytest.mark.parametrize('make,npts', [
    (lambda n, k, g: symmetric_clouds(n, k, g, eta=0.0, dtype=DTYPE), 48),
    (lambda n, k, g: c2_clouds(n, k, g, eta=0.0, dtype=DTYPE), 48),
    (lambda n, k, g: tetra_orbit_clouds(n, k, g, eta=0.0, dtype=DTYPE), 48),
])
def test_rank_survives_exact_symmetry(make, npts):
    """Global vector pooling confines every equivariant channel to Fix_H(rho)
    and drops the rank.  Keeping the point axis lets the symmetry PERMUTE the
    factors instead of fixing them, so K stays full rank."""
    gen = torch.Generator().manual_seed(17)
    P = make(3, npts, gen)
    lam = torch.linalg.eigvalsh(_model(candidate_k=24)(P))
    assert (lam[:, 0] > 1e-8 * lam[:, -1]).all()


def test_exact_distance_ties_do_not_change_the_stiffness():
    """The decisive tie test: on a cubic lattice the identity of the 'c-th
    nearest neighbour' is arbitrary.  Set aggregation is blind to it; using
    neighbour rank as a channel index is not."""
    gen = torch.Generator().manual_seed(23)
    P = lattice_clouds(2, 3, gen, dtype=DTYPE, trans_scale=1.0)
    model = _model(candidate_k=64)
    K = model(P)
    perm = torch.randperm(P.shape[1], generator=gen)
    assert scaled_err(model(P[:, perm]), K) < 1e-11

    R, p = random_SE3(1.0, gen, dtype=DTYPE)
    rho = coadjoint(R, p)
    assert scaled_err(model(transform_cloud(P, R, p)),
                      rho @ K @ rho.transpose(-1, -2)) < 1e-11


def test_rank_channel_backbone_is_the_one_that_breaks_on_ties():
    """Regression guard for the diagnosis: the failure is the RANK CHANNEL, not
    the compact kernel.  If this ever passes, the comparison in
    verify_pointwise.py check C has lost its meaning."""
    gen = torch.Generator().manual_seed(23)
    P = lattice_clouds(2, 3, gen, dtype=DTYPE, trans_scale=1.0)
    torch.manual_seed(0)
    rank_model = WrenchSecondMomentModel(
        WrenchEdgeEncoder(graph='kernel', candidate_k=16),
        weight_mode='learned',
        backbone_channels=(16, 16, 16, 8)).to(DTYPE).eval()
    with torch.no_grad():
        K = rank_model(P)
        perm = torch.randperm(P.shape[1], generator=gen)
        assert scaled_err(rank_model(P[:, perm]), K) > 1e-3


def test_gradients_reach_every_trainable_parameter():
    """Two branches start at exactly zero by design -- the residual bracket
    direction ``dir_v`` and the last layer of the beta MLP -- so on step 0 their
    upstream partners see no gradient.  Both carry gradient themselves, so one
    optimiser step unblocks the whole graph; that is what is asserted here."""
    P, _ = _clouds(n=2, npts=32)
    model = _model(message_passing=True, bracket='pairwise',
                   bracket_channels=4)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    def step():
        opt.zero_grad(set_to_none=True)
        model(P).square().sum().backward()
        return {n: (q.grad is not None and q.grad.abs().sum() > 0)
                for n, q in model.named_parameters() if q.requires_grad}

    first = step()
    assert first['head.beta_mlp.4.weight'], 'beta head must carry gradient'
    assert all(v for n, v in first.items() if n.endswith('dir_v.weight')), \
        'zero-initialised bracket direction must still carry gradient'
    opt.step()

    dead = [n for n, ok in step().items() if not ok]
    assert dead == [], f'no gradient reached after one step: {dead}'

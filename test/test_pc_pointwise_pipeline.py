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
                                                          wendland_c2)
from experiment.pc_se3_congruence.pointwise_models import (
    LocalWrenchSetEncoder, PointwiseStiffnessModel, force_pair, klein_pair)
from experiment.pc_se3_congruence.se3_utils import (coadjoint, random_SE3,
                                                    scaled_err,
                                                    transform_cloud)

DTYPE = torch.float64


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    kw.setdefault('candidate_k', 16)
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
    g0 = build_local_graph(P, candidate_k=16)
    R, p = random_SE3(1.0, gen, dtype=DTYPE)
    g1 = build_local_graph(transform_cloud(P, R, p), candidate_k=16)
    torch.testing.assert_close(g0.window, g1.window, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(g0.radius, g1.radius, rtol=1e-11, atol=1e-11)
    assert 0.0 <= g0.truncation_frac <= 1.0

    # A support that swallows every point must flag truncation.
    tight = build_local_graph(P, candidate_k=4, radius_alpha=10.0)
    assert tight.truncation_frac > 0.0


def test_edge_wrench_obeys_the_coadjoint_law():
    P, gen = _clouds()
    enc = LocalWrenchSetEncoder().to(DTYPE)
    R, p = random_SE3(1.0, gen, dtype=DTYPE)
    Pt = transform_cloud(P, R, p)
    w0 = enc.edge_wrenches(P, build_local_graph(P, candidate_k=16))
    wt = enc.edge_wrenches(Pt, build_local_graph(Pt, candidate_k=16))
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
    model = _model(candidate_k=16)
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

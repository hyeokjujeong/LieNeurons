"""[CURRENT] Synthetic cloud generators and analytic stiffness targets.

Despite the Experiment-B origin described below, this module is on the current
path and has no replacement: it supplies the abstract-cloud datasets
(``sample_clouds`` / ``symmetric_clouds`` / ``c2_clouds`` / ``tetra_orbit_clouds``
/ ``lattice_clouds``) and the analytic contact-spring targets used by
``blockage_bench.py``, ``verify_pointwise.py``, ``peg_hole_synth.py`` and the
pointwise test suite.  The realistic peg-and-hole scenes in ``peg_hole_synth.py``
ADD to these rather than replace them -- the symmetric and lattice clouds are
what make rank collapse and exact distance ties reproducible in isolation.

Two ground-truth generators, both exactly congruence-equivariant by
construction (K(T.P) = Ad_T^{-T} K(P) Ad_T^{-1} in the [m; f] wrench basis,
matching the output basis of KleinHeadB):

  (1) contact_spring_K — analytic, model-independent target.  Pose-free
      adaptation of docs/exp.md section 7.6: every kNN pair (i, j) contributes
      a zero-pitch "contact spring" wrench  w_ij = (m, f) = (r_i x d_ij, d_ij)
      with an SE(3)-invariant weight k(||d_ij||),

          K(P) = (1 / N k) sum_ij  k(||d_ij||) w_ij w_ij^T .

      Since w_ij -> Ad_T^{-T} w_ij (same computation as WrenchPlueckerEncoder)
      and the weights are invariant, K obeys the congruence law exactly.

  (2) a frozen randomly-initialized ModelB (built in train.py) — target
      guaranteed to lie in the model class; realizability sanity check.

Cloud sampling: anisotropic Gaussian blobs (per-axis scales in
[aniso_lo, aniso_hi]) placed at a random SE(3) pose with ||p|| ~ trans_scale.
trans_scale is kept O(1): docs/exp.md T6 shows the affine-invariant metric is
exact there but loses precision for ||p|| >~ 1e2 (log/inv-sqrt conditioning).
"""
import torch

from experiment.pc_se3_congruence.encoders import (compact_wendland_weights,
                                                   knn_indices)
from experiment.pc_se3_congruence.se3_utils import random_SO3


def sample_clouds(n_samples, n_points, gen=None, dtype=torch.float64,
                  trans_scale=1.0, aniso=(0.5, 2.0)):
    """Anisotropic Gaussian clouds at random SE(3) poses.  [S, N, 3]."""
    z = torch.randn(n_samples, n_points, 3, generator=gen, dtype=dtype)
    lo, hi = aniso
    s = lo + (hi - lo) * torch.rand(n_samples, 1, 3, generator=gen, dtype=dtype)
    P = z * s
    R = torch.stack([random_SO3(gen, dtype) for _ in range(n_samples)])
    p = torch.randn(n_samples, 1, 3, generator=gen, dtype=dtype) * trans_scale
    return P @ R.transpose(-1, -2) + p


def symmetric_clouds(n_samples, n_points, gen=None, dtype=torch.float64,
                     eta=0.0, aniso=(0.5, 2.0), trans_scale=0.0):
    """Centro-symmetric adversarial clouds: P = {±a_i} + eta * asymmetry.

    Diagnostic benchmark for the bracket-only blockage
    (bracket_blockage_analysis.md §3.3.4): at eta = 0 every point has an exact
    antipode, so the encoder's f-channels (mean rank-c kNN differences) vanish
    IDENTICALLY.  By the rank-collapse theorem any LNLinear + covector-bracket
    (+ invariant-gating) model then outputs K_ff = 0, K_fm = 0, rank(K) <= 3 —
    while contact_spring_K targets are rank-6 SPD with O(1) ff-blocks.
    Regression is provably impossible; eta > 0 interpolates the difficulty
    (f-channels scale ~ eta, so the model's ff-side is noise-limited).

    n_points must be even (antipodal pairs).  trans_scale shifts the symmetry
    center away from the origin; the collapse persists (f-channels are
    translation-invariant), which is itself a useful sanity check.
    """
    assert n_points % 2 == 0, 'centro-symmetric cloud needs an even n_points'
    half = n_points // 2
    z = torch.randn(n_samples, half, 3, generator=gen, dtype=dtype)
    lo, hi = aniso
    s = lo + (hi - lo) * torch.rand(n_samples, 1, 3, generator=gen, dtype=dtype)
    a = z * s
    P = torch.cat([a, -a], dim=1)                                # exact antipodes
    if eta > 0:
        P = P + eta * torch.randn(P.shape, generator=gen, dtype=dtype)
    R = torch.stack([random_SO3(gen, dtype) for _ in range(n_samples)])
    p = torch.randn(n_samples, 1, 3, generator=gen, dtype=dtype) * trans_scale
    return P @ R.transpose(-1, -2) + p


def c2_clouds(n_samples, n_points, gen=None, dtype=torch.float64,
              eta=0.0, aniso=(0.5, 2.0), trans_scale=0.0):
    """Clouds with a single C2 (180-deg) symmetry axis + eta * asymmetry.

    NOT centro-symmetric.  By the fixed-subspace lemma
    (bracket_blockage_analysis.md §6.5a) the encoder's f-channels are confined
    to the (pose-rotated) axis, so any bracket-only model has
    rank(K_ff) <= 1 at eta = 0 — while contact-spring targets have rank 3.
    """
    assert n_points % 2 == 0
    half = n_points // 2
    z = torch.randn(n_samples, half, 3, generator=gen, dtype=dtype)
    lo, hi = aniso
    s = lo + (hi - lo) * torch.rand(n_samples, 1, 3, generator=gen, dtype=dtype)
    a = z * s
    C2 = torch.diag(torch.tensor([-1., -1., 1.], dtype=dtype))   # 180 deg about z
    P = torch.cat([a, a @ C2.T], dim=1)
    if eta > 0:
        P = P + eta * torch.randn(P.shape, generator=gen, dtype=dtype)
    R = torch.stack([random_SO3(gen, dtype) for _ in range(n_samples)])
    p = torch.randn(n_samples, 1, 3, generator=gen, dtype=dtype) * trans_scale
    return P @ R.transpose(-1, -2) + p


_A4 = None


def _a4_group(dtype):
    """The 12 rotations of the tetrahedral group A4 (no -I: not centro-sym)."""
    global _A4
    if _A4 is None:
        perms = [torch.eye(3), torch.tensor([[0., 1, 0], [0, 0, 1], [1, 0, 0]]),
                 torch.tensor([[0., 0, 1], [1, 0, 0], [0, 1, 0]])]
        signs = [torch.diag(torch.tensor(s)) for s in
                 ([1., 1, 1], [1., -1, -1], [-1., 1, -1], [-1., -1, 1])]
        _A4 = [pm @ sg for pm in perms for sg in signs]
    return [g.to(dtype) for g in _A4]


def tetra_orbit_clouds(n_samples, n_points, gen=None, dtype=torch.float64,
                       eta=0.0, trans_scale=0.0):
    """A4-orbit clouds (tetrahedral rotation symmetry, no -I).

    Fix(A4) = {0}, so in exact arithmetic all f-channels would vanish; in
    practice symmetric orbits create structural kNN distance ties, so f_c is
    determined by tie-breaking noise (~10x suppressed).  See §6.5a caveat.
    n_points must be a multiple of 12.
    """
    assert n_points % 12 == 0
    m = n_points // 12
    G = _a4_group(dtype)
    b = torch.randn(n_samples, m, 3, generator=gen, dtype=dtype)
    P = torch.cat([b @ Rm.T for Rm in G], dim=1)
    if eta > 0:
        P = P + eta * torch.randn(P.shape, generator=gen, dtype=dtype)
    R = torch.stack([random_SO3(gen, dtype) for _ in range(n_samples)])
    p = torch.randn(n_samples, 1, 3, generator=gen, dtype=dtype) * trans_scale
    return P @ R.transpose(-1, -2) + p


def lattice_clouds(n_samples, n_side=3, gen=None, dtype=torch.float64,
                   spacing=1.0, jitter=0.0, trans_scale=0.0):
    """Cubic-lattice clouds -- the maximally TIE-DEGENERATE benchmark.

    A lattice of side ``n_side`` has huge families of exactly equal pairwise
    distances, so the identity of the "c-th nearest neighbour" is arbitrary.
    Any model that uses neighbour RANK as a channel index therefore produces a
    different answer for two relabellings of the SAME cloud, even though the
    geometry is identical.  Tie robustness is a strictly different requirement
    from rank preservation (:func:`symmetric_clouds`, :func:`c2_clouds`), and
    this generator isolates it.

    ``jitter`` > 0 breaks the ties continuously; the equivariance error of a
    tie-fragile model as a function of jitter is the near-tie robustness curve.
    n_points = n_side ** 3.
    """
    g = torch.arange(n_side, dtype=dtype) - (n_side - 1) / 2.0
    grid = torch.stack(torch.meshgrid(g, g, g, indexing='ij'), dim=-1)
    P = (grid.reshape(1, -1, 3) * spacing).repeat(n_samples, 1, 1)
    if jitter > 0:
        P = P + jitter * torch.randn(P.shape, generator=gen, dtype=dtype)
    R = torch.stack([random_SO3(gen, dtype) for _ in range(n_samples)])
    p = torch.randn(n_samples, 1, 3, generator=gen, dtype=dtype) * trans_scale
    return P @ R.transpose(-1, -2) + p


def contact_spring_K(P, k=12, sigma_k=0.5):
    """Analytic congruence-equivariant SPD target, [S, 6, 6] in [m; f] order.

    P: [S, N, 3].  For each point and its k nearest neighbors:
    w = (m, f) = (r_i x d_ij, d_ij), weight exp(-||d_ij||^2/(2 sigma_k^2)).
    """
    S, N, _ = P.shape
    idx = knn_indices(P, k)                                      # [S, N, k]
    nbr = torch.gather(P.unsqueeze(2).expand(S, N, k, 3), 1,
                       idx.unsqueeze(-1).expand(S, N, k, 3))
    d = nbr - P.unsqueeze(2)                                     # [S, N, k, 3]
    m = torch.cross(P.unsqueeze(2).expand_as(d), d, dim=-1)      # r_i x d_ij
    w = torch.cat([m, d], dim=-1)                                # [S, N, k, 6]
    kw = torch.exp(-(d * d).sum(-1) / (2.0 * sigma_k ** 2))      # [S, N, k]
    K = torch.einsum('snk,snki,snkj->sij', kw, w, w) / (N * k)
    return 0.5 * (K + K.transpose(-1, -2))


def contact_spring_all_pairs_K(P, sigma_k=0.5):
    """Tie-robust all-pairs version of :func:`contact_spring_K`.

    Every ordered pair ``i != j`` contributes, so there is no kNN rank or
    k-th-neighbor boundary to become ambiguous on an exactly symmetric cloud.
    This target is permutation invariant and congruence equivariant even when
    many pairwise distances are exactly equal.
    """
    S, N, _ = P.shape
    d = P.unsqueeze(1) - P.unsqueeze(2)                       # r_j - r_i
    r = P.unsqueeze(2).expand(S, N, N, 3)
    mask = ~torch.eye(N, dtype=torch.bool, device=P.device)
    d = d[:, mask].reshape(S, N * (N - 1), 3)
    r = r[:, mask].reshape(S, N * (N - 1), 3)
    m = torch.cross(r, d, dim=-1)
    w = torch.cat([m, d], dim=-1)
    kw = torch.exp(-d.square().sum(-1) / (2.0 * sigma_k ** 2))
    K = torch.einsum('se,sei,sej->sij', kw, w, w) / (N * (N - 1))
    return 0.5 * (K + K.transpose(-1, -2))


def contact_spring_kernel_K(P, candidate_k=32, sigma_k=0.5):
    """Local, boundary-robust contact-spring target.

    Only ``candidate_k`` nearest edges per anchor are materialized.  Their
    analytic radial spring weight is multiplied by a compact Wendland window
    that reaches zero at the farthest candidate.  Consequently an exact
    distance shell cut by the candidate boundary contributes zero on both
    sides, while the stored edge count remains O(N * candidate_k).
    """
    S, N, _ = P.shape
    edge_k = min(candidate_k, N - 1)
    idx = knn_indices(P, edge_k)
    nbr = torch.gather(P.unsqueeze(2).expand(S, N, edge_k, 3), 1,
                       idx.unsqueeze(-1).expand(S, N, edge_k, 3))
    d = nbr - P.unsqueeze(2)
    m = torch.cross(P.unsqueeze(2).expand_as(d), d, dim=-1)
    w = torch.cat([m, d], dim=-1)
    sqdist = d.square().sum(-1)
    radial = torch.exp(-sqdist / (2.0 * sigma_k ** 2))
    window = compact_wendland_weights(sqdist, candidate_dim=-1)
    kw = radial * window
    K = torch.einsum('snk,snki,snkj->sij', kw, w, w) / (N * edge_k)
    return 0.5 * (K + K.transpose(-1, -2))


def spd_stats(K):
    """min/max eigenvalue and condition-number stats over a batch of SPD."""
    lam = torch.linalg.eigvalsh(K)
    lmin, lmax = lam[:, 0], lam[:, -1]
    cond = lmax / lmin
    return {
        'lam_min': lmin.min().item(),
        'lam_max': lmax.max().item(),
        'cond_median': cond.median().item(),
        'cond_max': cond.max().item(),
    }

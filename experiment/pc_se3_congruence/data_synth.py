"""Synthetic point-cloud -> stiffness data for the Experiment-B training run.

Two ground-truth generators, both exactly congruence-equivariant by
construction (K(T.P) = Ad_T^{-T} K(P) Ad_T^{-1} in the [f; m] wrench basis,
matching the output basis of KleinHeadB):

  (1) contact_spring_K — analytic, model-independent target.  Pose-free
      adaptation of docs/exp.md section 7.6: every kNN pair (i, j) contributes
      a zero-pitch "contact spring" wrench  w_ij = (f, m) = (d_ij, r_i x d_ij)
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

from experiment.pc_se3_congruence.encoders import knn_indices
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


def contact_spring_K(P, k=12, sigma_k=0.5):
    """Analytic congruence-equivariant SPD target, [S, 6, 6] in [f; m] order.

    P: [S, N, 3].  For each point and its k nearest neighbors:
    w = (f, m) = (d_ij, r_i x d_ij), weight exp(-||d_ij||^2 / (2 sigma_k^2)).
    """
    S, N, _ = P.shape
    idx = knn_indices(P, k)                                      # [S, N, k]
    nbr = torch.gather(P.unsqueeze(2).expand(S, N, k, 3), 1,
                       idx.unsqueeze(-1).expand(S, N, k, 3))
    d = nbr - P.unsqueeze(2)                                     # [S, N, k, 3]
    m = torch.cross(P.unsqueeze(2).expand_as(d), d, dim=-1)      # r_i x d_ij
    w = torch.cat([d, m], dim=-1)                                # [S, N, k, 6]
    kw = torch.exp(-(d * d).sum(-1) / (2.0 * sigma_k ** 2))      # [S, N, k]
    K = torch.einsum('snk,snki,snkj->sij', kw, w, w) / (N * k)
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

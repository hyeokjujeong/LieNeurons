"""Tie-safe local graph for the pointwise wrench -> stiffness pipeline.

NOTATION (fixed once for the whole pointwise pipeline; the stiffness matrix is
the only object called K).

    B   batch size
    N   points per cloud
    k   candidate neighbours per point (``k_nbr``) -- an UNORDERED set axis,
        never a channel index
    C   latent channel multiplicity of the Lie-Neuron backbone
    H   final factor channels of the second-moment head
    K   the 6x6 stiffness matrix

    P          [B, N, 3]
    edge wrench[B, N, k, 6]      stored [f; m]
    X          [B, C, 6, N]      repo Lie-Neuron layout (point axis trailing)
    Z          [B, H, 6, N]
    K          [B, 6, 6]

WHY THE GRAPH NEEDS ITS OWN MODULE.  The second-moment head is invariant to a
relabelling of the points only if the neighbourhood assignment itself is
set-equivariant,

    N_{pi(i)}(T.P) = pi( N_i(P) ),

for every rigid motion T and every relabelling pi.  Multiplying a scalar kernel
onto an already-broken graph does not restore that property, so the support is
built here from smooth invariant quantities and every boundary edge is faded out
by a window that is exactly zero at the cutoff.

Two families are provided:

  smooth support ('global_scale', 'density_scaled', 'fixed')
      The support radius is a function of the whole cloud (RMS radius) or a
      constant, never of an order statistic of one anchor's distances.  Edges
      cross the boundary with weight -> 0, so an exact distance shell sitting on
      the boundary contributes zero from both sides.  The support is a physical
      length, which is what a fixed radius has to be if the model is ever to
      transfer between sampling densities.

  anchor-adaptive support ('knn_adaptive', 'knn_shell')
      The radius is the anchor's own k-th neighbour distance.  'knn_shell'
      additionally closes the tied shell with hard weights (degree may exceed
      support_k).

WHAT THE VERIFICATION ACTUALLY SHOWS (verify_pointwise.py, check C).  All four
smooth modes AND 'knn_adaptive' pass exact-tie permutation invariance at machine
precision, because d_{i,(k)} is a continuous, permutation-invariant order
statistic and the pooled sum runs over a SET.  The measured tie failure
(~2.6e-1 relative on a cubic lattice) comes from the other direction entirely:
using neighbour RANK as a channel index, which is what the existing edge-tensor
model does when an LN backbone mixes its candidate channels.  So the adaptive
radius is kept as a legitimate option, not as a known-broken control -- its real
cost is that the whole profile q_ij = d_ij / r_i is tied to one order statistic
and therefore to a single outlying neighbour.

Candidate truncation is the one remaining way for the graph itself to break
set-equivariance: if more than ``candidate_k`` points fall inside the support,
the materialised set depends on top-k tie breaking.  :func:`build_local_graph`
therefore reports ``truncation_frac``; keep it at 0.
"""
import dataclasses
import warnings

import torch
import torch.nn as nn

_BIG = 1e12
_EPS = 1e-12

# Truncation silently removes the set-equivariance guarantee, and the default
# radius is calibrated for small clouds -- at N = 128 with candidate_k = 32 it
# already reaches 0.36.  Warn once per process rather than every forward.
TRUNCATION_WARN_THRESHOLD = 0.01
_warned = False


def wendland_c2(q):
    """C2 Wendland window  phi(q) = (1-q)^4 (1+4q) on [0, 1), 0 beyond.

    phi(1) = phi'(1) = 0, so an edge leaving the support does so with vanishing
    value AND vanishing derivative -- the boundary is smooth for both the
    forward pass and the gradient.
    """
    one_minus_q = (1.0 - q).clamp_min(0.0)
    return one_minus_q.pow(4) * (1.0 + 4.0 * q)


def cloud_scale(P):
    """RMS radius about the centroid: [B, N, 3] -> [B, 1, 1].

    Permutation invariant and rigid-motion invariant, and smooth in P -- the
    properties an order statistic such as ``d_{i,(k)}`` lacks.
    """
    c = P.mean(dim=1, keepdim=True)
    var = (P - c).square().sum(-1).mean(dim=1)                 # [B]
    return var.clamp_min(_EPS).sqrt().view(-1, 1, 1)


@dataclasses.dataclass
class LocalGraph:
    """Materialised candidate neighbourhood, all tensors sharing [B, N, k].

    idx       neighbour indices (rank-ordered; the ORDER IS NEVER USED as a
              feature, only to bound the candidate set)
    edge_vec  d_ij = p_j - p_i                                  [B, N, k, 3]
    dist      ||d_ij||                                          [B, N, k]
    q         d_ij / r_i (dimensionless)                        [B, N, k]
    window    compact support weight, 0 outside the support     [B, N, k]
    radius    r_i                                               [B, N, 1]
    """

    idx: torch.Tensor
    edge_vec: torch.Tensor
    dist: torch.Tensor
    q: torch.Tensor
    window: torch.Tensor
    radius: torch.Tensor
    truncation_frac: float
    mean_degree: float
    max_degree: float

    @property
    def stats(self):
        return {'graph_truncation_frac': self.truncation_frac,
                'graph_mean_degree': self.mean_degree,
                'graph_max_degree': self.max_degree}


RADIUS_MODES = ('global_scale', 'density_scaled', 'fixed', 'knn_adaptive',
                'knn_shell')


def build_local_graph(P, candidate_k=32, radius_mode='global_scale',
                      radius_alpha=0.75, radius_value=None, support_k=8,
                      target_k=16, tie_eps=0.0,
                      dist_compute_mode='donot_use_mm_for_euclid_dist'):
    """P: [B, N, 3] -> :class:`LocalGraph`.

    radius_mode:
      global_scale   r = alpha * rms_radius(P).  Smooth + invariant; the number
                     of in-support neighbours grows with N at fixed alpha.
      density_scaled r = alpha * rms_radius(P) * (target_k / N)^(1/3).  Same
                     smoothness, but the expected degree stays ~target_k as N
                     changes (N is a constant of the cloud, not an order
                     statistic, so this is still tie-safe).
      fixed          r = radius_value, a physical length.
      knn_adaptive   r_i = d_{i,(support_k)}, anchor adaptive.  Continuous and
                     permutation invariant, but the whole profile is tied to a
                     single order statistic.
      knn_shell      hard tie-closed kNN: window = 1 for d <= d_{i,(support_k)}
                     + tie_eps.  Degree may exceed support_k on a tied shell.

    ``dist_compute_mode`` defaults to the non-mm cdist kernel: the mm identity
    ||a||^2 - 2a.b + ||b||^2 loses the exact symmetry d_ij = d_ji that tie
    handling relies on.
    """
    if radius_mode not in RADIUS_MODES:
        raise ValueError(f'unknown radius_mode={radius_mode!r}; '
                         f'expected one of {RADIUS_MODES}')
    if radius_mode == 'fixed' and radius_value is None:
        raise ValueError("radius_mode='fixed' needs radius_value")

    B, N, _ = P.shape
    k = int(min(candidate_k, N - 1))
    if k < 1:
        raise ValueError(f'need at least 2 points to build a graph, got N={N}')

    d_full = torch.cdist(P, P, compute_mode=dist_compute_mode)
    eye = torch.eye(N, dtype=P.dtype, device=P.device)
    dist, idx = (d_full + eye * _BIG).topk(k, dim=-1, largest=False)

    nbr = torch.gather(P.unsqueeze(2).expand(B, N, k, 3), 1,
                       idx.unsqueeze(-1).expand(B, N, k, 3))
    edge_vec = nbr - P.unsqueeze(2)                            # [B, N, k, 3]

    if radius_mode in ('knn_adaptive', 'knn_shell'):
        sk = int(min(support_k, k))
        radius = dist[..., sk - 1:sk]                          # [B, N, 1]
    else:
        scale = cloud_scale(P)                                 # [B, 1, 1]
        if radius_mode == 'global_scale':
            radius = radius_alpha * scale
        elif radius_mode == 'density_scaled':
            radius = radius_alpha * scale * (float(target_k) / N) ** (1.0 / 3.0)
        else:
            radius = torch.full((B, 1, 1), float(radius_value),
                                dtype=P.dtype, device=P.device)
        radius = radius.expand(B, N, 1)

    q = dist / radius.clamp_min(_EPS)
    if radius_mode == 'knn_shell':
        window = (dist <= radius + tie_eps).to(P.dtype)
    else:
        window = wendland_c2(q)

    # An anchor whose farthest CANDIDATE still lies inside the support has had
    # in-support neighbours silently dropped by top-k; that is the only path
    # left for tie breaking to influence the output.
    truncation = (dist[..., -1:] < radius).to(P.dtype).mean().item()
    degree = (window > 0).sum(-1).to(P.dtype)
    global _warned
    if truncation > TRUNCATION_WARN_THRESHOLD and not _warned:
        _warned = True
        warnings.warn(
            f'local graph truncation_frac={truncation:.3f} at N={N}, '
            f'candidate_k={k}, radius_mode={radius_mode}: in-support '
            'neighbours were dropped by top-k, so the neighbourhood is no '
            'longer a pure function of the geometry and set-equivariance is '
            'not guaranteed.  Raise candidate_k, lower radius_alpha, or use '
            "radius_mode='density_scaled' (degree stays ~target_k as N grows).",
            RuntimeWarning, stacklevel=2)
    return LocalGraph(idx=idx, edge_vec=edge_vec, dist=dist, q=q, window=window,
                      radius=radius, truncation_frac=truncation,
                      mean_degree=degree.mean().item(),
                      max_degree=degree.max().item())


class EdgeInvariants(nn.Module):
    """SE(3)- and permutation-invariant scalars attached to every candidate edge.

    Every feature is a function of DISTANCES only, hence invariant under the
    rigid motion and unchanged by a relabelling of the points.  Neighbour RANK
    is deliberately absent: it is the one edge quantity that jumps at a distance
    tie, and it is exactly what the old rank-channel encoders used as a channel
    index.

    Features (dim = n_rbf + 5):
      q                    normalised distance d_ij / r_i
      window               compact support weight
      log1p(d_ij)          absolute length (breaks the scale degeneracy of q)
      soft degree          sum_l window_il, broadcast to every edge of anchor i
      d_ij / mean_d_i      local density-relative length
      RBF(q)               n_rbf Gaussians on [0, 1]
    """

    def __init__(self, n_rbf=8):
        super().__init__()
        if n_rbf < 1:
            raise ValueError('n_rbf must be >= 1')
        self.n_rbf = n_rbf
        self.register_buffer('centers', torch.linspace(0.0, 1.0, n_rbf))
        self.width = 1.0 / max(n_rbf - 1, 1)

    @property
    def dim(self):
        return self.n_rbf + 5

    def forward(self, graph):
        """:class:`LocalGraph` -> [B, N, k, dim]."""
        q, win, dist = graph.q, graph.window, graph.dist
        centers = self.centers.to(q.dtype).view(1, 1, 1, -1)
        rbf = torch.exp(-((q.unsqueeze(-1) - centers) / self.width).square())
        deg = win.sum(-1, keepdim=True)                         # [B, N, 1]
        mean_d = (win * dist).sum(-1, keepdim=True) / deg.clamp_min(_EPS)
        ratio = dist / mean_d.clamp_min(_EPS)
        base = torch.stack([q, win, torch.log1p(dist),
                            deg.expand_as(q), ratio], dim=-1)
        return torch.cat([base, rbf], dim=-1)

"""[CURRENT MODEL]
Tie-safe local graph for the pointwise wrench -> stiffness pipeline.

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
    edge wrench[B, N, k, 6]      stored [m; f]
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

Three families are provided:

  degree-matched support ('degree_matched', THE DEFAULT)
      The radius is calibrated so that the mean in-support degree equals
      ``target_k`` exactly, in closed form, as an order statistic of the
      pairwise-distance MULTISET (:func:`degree_matched_radius`).  It assumes
      no density model and no intrinsic dimension, so a volumetric blob, a
      surface scan, a curve and a lattice all get the same effective
      neighbourhood size.  It is still a global physical length (one radius
      per cloud), so it transfers across sampling densities the way the
      smooth modes do.

  smooth support ('global_scale', 'density_scaled', 'fixed')
      The support radius is a function of the whole cloud (RMS radius) or a
      constant, never of an order statistic of one anchor's distances.  Edges
      cross the boundary with weight -> 0, so an exact distance shell sitting on
      the boundary contributes zero from both sides.  The support is a physical
      length, which is what a fixed radius has to be if the model is ever to
      transfer between sampling densities.

      Both scaled variants bake in an assumption that measurement contradicts.
      'global_scale' holds the radius at a fixed fraction of the cloud extent,
      so the degree grows without bound in N (measured: mean degree 6.2 -> 23.6
      -> 55.6 with truncation 0.71 for iid volumetric N = 32 -> 128 -> 512 at
      alpha = 0.75).  'density_scaled' divides that by (target_k / N)^(1/3),
      which is the correct density correction ONLY for an intrinsic dimension
      of 3; on a surface (peg-and-hole scans, intrinsic dimension 2) the degree
      still drifts 16.5 -> 24.3 from N = 512 to 2048, and on a curve
      (intrinsic dimension 1) it reaches 59.9 with truncation 0.69.  Prefer
      'degree_matched' unless a run has to reproduce an earlier configuration.

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
the materialised set depends on top-k tie breaking.  ``candidate_k`` is a pure
memory budget -- once it covers the support, materialising more candidates
changes nothing, because the extra ones sit outside the window and contribute
exactly 0.  So the requirement is the single inequality

    candidate_k  >=  max_i |{j : d_ij < r_i}|,

and :func:`build_local_graph` reports both sides of it: ``truncation_frac`` is
the fraction of anchors that violate it, and ``required_candidate_k`` is the
right-hand side, i.e. the value to set ``candidate_k`` to.  Both are measured
from the FULL distance matrix, which is independent of the truncation they are
detecting.  Keep ``truncation_frac`` at 0.

The budget is NOT adjusted automatically.  A truncation warning almost always
means the radius is wrong rather than the budget: at candidate_k = 64 the
default 'degree_matched' radius leaves truncation at 0 on every distribution
measured (volumetric N = 32..1024, surface scans, curves, lattices, symmetric
orbits), whereas the old 'global_scale' alpha = 0.75 radius needs 800
candidates at N = 2048 -- a 10 GB edge tensor.  Silently growing the budget
there would replace a clear diagnostic with an out-of-memory error.
"""
import dataclasses
import warnings

import torch
import torch.nn as nn

_BIG = 1e12
_EPS = 1e-12

# Truncation silently removes the set-equivariance guarantee.  Warn once per
# process rather than every forward.
TRUNCATION_WARN_THRESHOLD = 0.01
_warned = False


def reset_warnings():
    """Re-arm the once-per-process graph warning (used by the tests)."""
    global _warned
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


def degree_matched_radius(d_masked, target_k, alpha=1.0):
    """Global radius whose MEAN in-support degree is exactly ``target_k``.

    ``d_masked``: [B, N, N] pairwise distances with the diagonal set to _BIG.
    Returns [B, 1, 1].

    The calibration is closed form, not a search.  Writing deg_i(r) for the
    number of neighbours of anchor i within r,

        (1/N) sum_i deg_i(r) = target_k
          <=>  |{(i, j) : i != j, d_ij <= r}| = N * target_k
          <=>  r = the (N * target_k)-th smallest off-diagonal pairwise
                   distance,

    because the left-hand side counts exactly the ordered pairs whose distance
    is at most r.  So one ``kthvalue`` over the distance multiset gives the
    radius that hits the target degree, with

      * no density model and no intrinsic-dimension exponent -- the reason
        this transfers between volumetric, surface, curve and lattice clouds
        where (target_k / N)^(1/3) does not;
      * permutation invariance and rigid invariance, because the MULTISET of
        pairwise distances carries both (relabelling permutes the multiset
        onto itself, a rigid motion leaves every distance unchanged);
      * continuity in P even at exact ties -- an order statistic's VALUE is
        continuous where the identity of the point attaining it is not, which
        is the same argument that makes d_{i,(k)} tie-safe (module docstring).

    ``alpha`` scales the calibrated radius; alpha = 1 hits ``target_k``.

    CAVEAT on exactly-degenerate spectra.  With ties the exact statement is
    the quantile sandwich

        mean_i count(d_ij <  r)  <=  target_k  <=  mean_i count(d_ij <= r),

    which collapses to the equality above when distances are distinct (the two
    branches then differ by 1/N).  On a cubic lattice the shells are
    {1, sqrt2, sqrt3, ...}, so the attainable mean counts jump 6.7 -> 10.0 ->
    16.7 -> ... and target_k = 8 is not attainable by ANY radius.  The degree
    the encoder sees is additionally WINDOWED, and wendland_c2 is exactly 0 at
    q = 1, so a shell sitting precisely on the boundary contributes nothing --
    the property that makes the boundary tie-safe in the first place.  The
    windowed degree therefore lands on the lower branch.  The discreteness is
    in the cloud, not in the calibration.
    """
    B, N, _ = d_masked.shape
    if target_k < 1:
        raise ValueError(f'target_k must be >= 1, got {target_k}')
    m = int(min(N * int(target_k), N * (N - 1)))
    r = d_masked.reshape(B, -1).kthvalue(m, dim=-1).values      # [B]
    return alpha * r.view(B, 1, 1)


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
    truncation_frac: float      # fraction of anchors with in_support > k
    mean_degree: float
    max_degree: float
    candidate_k: int = 0        # candidates actually materialised
    required_candidate_k: int = 0   # max in-support degree over all anchors

    @property
    def stats(self):
        return {'graph_truncation_frac': self.truncation_frac,
                'graph_mean_degree': self.mean_degree,
                'graph_max_degree': self.max_degree,
                'graph_candidate_k': float(self.candidate_k),
                'graph_required_candidate_k': float(self.required_candidate_k)}


RADIUS_MODES = ('degree_matched', 'global_scale', 'density_scaled', 'fixed',
                'knn_adaptive', 'knn_shell')

# alpha = 1 means "use the calibrated radius as is" for degree_matched; the
# older modes keep the value their published runs were tuned at.
_DEFAULT_ALPHA = {'degree_matched': 1.0}
_FALLBACK_ALPHA = 0.75


def default_radius_alpha(radius_mode):
    return _DEFAULT_ALPHA.get(radius_mode, _FALLBACK_ALPHA)


def build_local_graph(P, candidate_k=64, radius_mode='degree_matched',
                      radius_alpha=None, radius_value=None, support_k=8,
                      target_k=16, tie_eps=0.0,
                      dist_compute_mode='donot_use_mm_for_euclid_dist'):
    """P: [B, N, 3] -> :class:`LocalGraph`.

    radius_mode:
      degree_matched r = alpha * (N*target_k)-th smallest pairwise distance,
                     i.e. the radius at which the MEAN in-support degree is
                     exactly target_k (:func:`degree_matched_radius`).  The
                     default: it is the only mode that holds the degree fixed
                     across intrinsic dimension, density and N.
      global_scale   r = alpha * rms_radius(P).  Smooth + invariant; the number
                     of in-support neighbours grows without bound with N.
      density_scaled r = alpha * rms_radius(P) * (target_k / N)^(1/3).  Same
                     smoothness, and the degree stays ~target_k as N changes
                     ONLY for volumetric clouds -- the exponent is an intrinsic
                     dimension 3 assumption (module docstring).
      fixed          r = radius_value, a physical length.
      knn_adaptive   r_i = d_{i,(support_k)}, anchor adaptive.  Continuous and
                     permutation invariant, but the whole profile is tied to a
                     single order statistic.
      knn_shell      hard tie-closed kNN: window = 1 for d <= d_{i,(support_k)}
                     + tie_eps.  Degree may exceed support_k on a tied shell.

    ``radius_alpha=None`` picks :func:`default_radius_alpha` for the mode.

    ``candidate_k`` is a memory budget, not a model parameter; it only has to
    cover the support.  The returned graph reports ``required_candidate_k`` --
    set ``candidate_k`` to at least that and ``truncation_frac`` is 0.  The
    default 64 suffices for every distribution measured with the default
    radius (module docstring).

    ``dist_compute_mode`` defaults to the non-mm cdist kernel: the mm identity
    ||a||^2 - 2a.b + ||b||^2 loses the exact symmetry d_ij = d_ji that tie
    handling relies on.
    """
    if radius_mode not in RADIUS_MODES:
        raise ValueError(f'unknown radius_mode={radius_mode!r}; '
                         f'expected one of {RADIUS_MODES}')
    if radius_mode == 'fixed' and radius_value is None:
        raise ValueError("radius_mode='fixed' needs radius_value")
    if radius_alpha is None:
        radius_alpha = default_radius_alpha(radius_mode)

    B, N, _ = P.shape
    k = int(min(candidate_k, N - 1))
    if k < 1:
        raise ValueError(f'need at least 2 points to build a graph, got N={N}')
    d_full = torch.cdist(P, P, compute_mode=dist_compute_mode)
    eye = torch.eye(N, dtype=P.dtype, device=P.device)
    d_masked = d_full + eye * _BIG

    # ---- radius first, from the FULL distance matrix where possible.  The
    # in-support count below has to be independent of the candidate budget,
    # since detecting a too-small budget is exactly its job.
    if radius_mode in ('knn_adaptive', 'knn_shell'):
        sk = int(min(support_k, N - 1))
        radius = d_masked.topk(sk, dim=-1, largest=False).values[..., sk - 1:sk]
    elif radius_mode == 'degree_matched':
        radius = degree_matched_radius(d_masked, target_k,
                                       radius_alpha).expand(B, N, 1)
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

    # Exact in-support degree, computed before any truncation can hide it.
    if radius_mode == 'knn_shell':
        in_support = (d_masked <= radius + tie_eps).sum(-1)
    else:
        in_support = (d_masked < radius).sum(-1)               # [B, N]
    required = int(in_support.max().item())

    dist, idx = d_masked.topk(k, dim=-1, largest=False)
    nbr = torch.gather(P.unsqueeze(2).expand(B, N, k, 3), 1,
                       idx.unsqueeze(-1).expand(B, N, k, 3))
    edge_vec = nbr - P.unsqueeze(2)                            # [B, N, k, 3]

    q = dist / radius.clamp_min(_EPS)
    if radius_mode == 'knn_shell':
        window = (dist <= radius + tie_eps).to(P.dtype)
    else:
        window = wendland_c2(q)

    # An anchor with MORE in-support neighbours than candidates has had some
    # of them silently dropped by top-k; that is the only path left for tie
    # breaking to influence the output.  Measured against the exact in-support
    # count, not against "is the farthest candidate inside the support" -- the
    # latter also fires when the budget is exactly sufficient (in_support == k)
    # and nothing was dropped at all.
    truncation = (in_support > k).to(P.dtype).mean().item()
    degree = (window > 0).sum(-1).to(P.dtype)
    global _warned
    if truncation > TRUNCATION_WARN_THRESHOLD and not _warned:
        _warned = True
        hint = ('' if radius_mode == 'degree_matched' else
                ".  A required_candidate_k this far above target_k usually "
                "means the RADIUS is wrong, not the budget: radius_mode="
                "'degree_matched' holds the mean degree at target_k for any "
                'point distribution')
        warnings.warn(
            f'local graph truncation_frac={truncation:.3f} at N={N}, '
            f'candidate_k={k}, radius_mode={radius_mode}: the support holds '
            f'up to {required} neighbours, so top-k dropped in-support edges '
            'and the neighbourhood is no longer a pure function of the '
            f'geometry -- set-equivariance is not guaranteed.  Set '
            f'candidate_k >= {required}, or shrink the support via '
            f'radius_alpha / target_k{hint}.', RuntimeWarning, stacklevel=2)
    return LocalGraph(idx=idx, edge_vec=edge_vec, dist=dist, q=q, window=window,
                      radius=radius, truncation_frac=truncation,
                      mean_degree=degree.mean().item(),
                      max_degree=degree.max().item(),
                      candidate_k=k, required_candidate_k=required)


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

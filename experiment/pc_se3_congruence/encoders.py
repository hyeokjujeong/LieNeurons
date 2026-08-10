"""[COMPARISON ARMS] Global-pooling point cloud -> se(3) feature encoders.

The current encoder is
:class:`~experiment.pc_se3_congruence.pointwise_models.LocalWrenchSetEncoder`,
which removes the neighbour axis k by learned SET aggregation and keeps the
point axis N.  Everything in this module does the opposite -- it averages the
point axis into global vector channels and uses neighbour RANK as the channel
index -- and is kept only because ``blockage_bench.py`` selects between these
six encoders via ``--encoder`` to reproduce the reported baselines.

Rank channels are not invariant under a relabelling of an equal-distance shell,
so these encoders break equivariance on exact distance ties (cubic lattice,
tetrahedral orbits).  That is a property of the construction, not a bug to fix
here; see ``pointwise_models.py`` for the replacement.

EXCEPTION -- two module-level helpers below are CURRENT and depended on by the
label generators, not by the superseded encoders: ``knn_indices`` and
``compact_wendland_weights`` are imported by ``data_synth.py`` and
``peg_hole_synth.py`` to build the analytic contact-spring targets.  They are
plain geometry, unaffected by the pooling argument above.

Two constructions of the same underlying map (the origin-referenced Pluecker /
screw lifting  L0(r, n) = (n, r x n)  in ANGULAR-FIRST [omega; v] storage order;
wrench encoders store [m; f] to match):

  (1) PlueckerEncoder  — closed form, no parameters: directions are pairwise
      differences d_ij = r_j - r_i, moment r_i x d_ij = r_i x r_j.
  (2) LearnableLiftEncoder — directions are the output of a small VN-DGCNN
      (SO(3)-equivariant direction field on centered coordinates), lifted at
      the anchor and transported back to the origin by Ad_{(I,c)}.

Both output twist features V in R^{B x C x 6 x 1} with V(T.P) = Ad_T V(P).
"""
import torch
import torch.nn as nn

from experiment.pc_se3_congruence.se3_utils import hat3


# ---------------------------------------------------------------- kNN helper
def knn_indices(P, k):
    """P: [B, N, 3] -> idx [B, N, k], nearest first, self excluded.
    Pairwise distances are SE(3)-invariant, so the graph and the ordering of
    neighbors are invariant (assuming no exact ties)."""
    d = torch.cdist(P, P)                                   # [B, N, N]
    d = d + torch.eye(P.shape[1], dtype=P.dtype, device=P.device) * 1e10
    return d.topk(k, dim=-1, largest=False).indices


def compact_wendland_weights(sqdist, candidate_dim=-1, eps=1e-12):
    """Smooth compact kernel whose support ends at the outer candidate.

    ``sqdist`` contains distances to a fixed number of nearest candidates.
    Let ``r`` be the farthest candidate distance.  We use the C2 Wendland
    window

        phi(q) = (1 - q)^4 (1 + 4 q),  0 <= q < 1,
                 0,                     q >= 1,

    where ``q = distance / r``.  The farthest selected edge therefore has
    exactly zero weight.  If an equal-distance shell straddles the candidate
    boundary, every edge on that shell also has zero weight, so arbitrary
    top-k tie breaking cannot change the tensor sum.  Near-boundary swaps are
    suppressed smoothly rather than producing an O(1) jump.
    """
    cutoff_sq = sqdist.amax(dim=candidate_dim, keepdim=True)
    q = torch.sqrt(sqdist / cutoff_sq.clamp_min(eps))
    one_minus_q = (1.0 - q).clamp_min(0.0)
    return one_minus_q.pow(4) * (1.0 + 4.0 * q)


# ------------------------------------------------------- (1) Pluecker encoder
class PlueckerEncoder(nn.Module):
    """Closed-form pairwise lifting.  Channel c = rank-c nearest neighbor
    (rank is distance-ordered, hence SE(3)-invariant)."""

    def __init__(self, k=8):
        super().__init__()
        self.k = k

    def forward(self, P):
        """P: [B, N, 3] -> V: [B, k, 6, 1]"""
        B, N, _ = P.shape
        idx = knn_indices(P, self.k)                         # [B, N, k]
        nbr = torch.gather(P.unsqueeze(2).expand(B, N, self.k, 3), 1,
                           idx.unsqueeze(-1).expand(B, N, self.k, 3))
        d = nbr - P.unsqueeze(2)                             # [B, N, k, 3]
        m = torch.cross(P.unsqueeze(2).expand_as(d), d, dim=-1)  # r_i x d_ij
        xi = torch.cat([d, m], dim=-1)                       # [B, N, k, 6], [omega; v]
        V = xi.mean(dim=1)                                   # [B, k, 6]
        return V.unsqueeze(-1)                               # [B, k, 6, 1]


# ------------------------------------------------ minimal Vector Neuron stack
class VNLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels) *
                                   (1.0 / in_channels ** 0.5))

    def forward(self, x):
        """x: [B, C, 3, ...] -> [B, C', 3, ...]"""
        return torch.einsum('oc,bc...->bo...', self.weight.to(x.dtype), x)


class VNLeakyReLU(nn.Module):
    def __init__(self, in_channels, negative_slope=0.2, eps=1e-12):
        super().__init__()
        self.dir = VNLinear(in_channels, in_channels)
        self.alpha = negative_slope
        self.eps = eps

    def forward(self, x):
        """x: [B, C, 3, ...]"""
        d = self.dir(x)
        dot = (x * d).sum(dim=2, keepdim=True)
        dsq = (d * d).sum(dim=2, keepdim=True)
        folded = x - (dot / (dsq + self.eps)) * d
        return self.alpha * x + (1 - self.alpha) * torch.where(dot >= 0, x, folded)


class VNEdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = VNLinear(2 * in_channels, out_channels)
        self.act = VNLeakyReLU(out_channels)

    def forward(self, x, idx):
        """x: [B, C, 3, N], idx: [B, N, k] -> [B, C', 3, N]"""
        B, C, _, N = x.shape
        k = idx.shape[-1]
        xt = x.permute(0, 3, 1, 2)                            # [B, N, C, 3]
        nbr = torch.gather(xt.unsqueeze(2).expand(B, N, k, C, 3), 1,
                           idx.unsqueeze(-1).unsqueeze(-1).expand(B, N, k, C, 3))
        ctr = xt.unsqueeze(2).expand_as(nbr)
        edge = torch.cat([nbr - ctr, ctr], dim=3)             # [B, N, k, 2C, 3]
        edge = edge.permute(0, 3, 4, 1, 2)                    # [B, 2C, 3, N, k]
        out = self.act(self.lin(edge))                        # [B, C', 3, N, k]
        return out.mean(dim=-1)                               # [B, C', 3, N]


# --------------------------------------------- (2) learnable lifting encoder
class LearnableLiftEncoder(nn.Module):
    """VN-DGCNN direction field + Pluecker lifting.

    mode:
      'origin'           lift directly with origin-referenced moment r x n
      'anchor_transport' lift at the anchor c, then transport by Ad_{(I,c)}
                         (mathematically identical to 'origin'; Lemma 3.5)
      'no_transport'     anchor lift WITHOUT transport — negative control,
                         only SO(3) x SO(3)-equivariant.
    """

    def __init__(self, out_channels=8, hidden=8, k=8, mode='anchor_transport'):
        super().__init__()
        assert mode in ('origin', 'anchor_transport', 'no_transport')
        self.k = k
        self.mode = mode
        self.conv1 = VNEdgeConv(1, hidden)
        self.conv2 = VNEdgeConv(hidden, hidden)
        self.lin_out = VNLinear(hidden, out_channels)
        self.act_out = VNLeakyReLU(out_channels)

    def forward(self, P):
        """P: [B, N, 3] -> V: [B, C, 6, 1]"""
        B, N, _ = P.shape
        c = P.mean(dim=1, keepdim=True)                       # [B, 1, 3]
        x = (P - c).transpose(1, 2).unsqueeze(1)              # [B, 1, 3, N]
        idx = knn_indices(P, self.k)                          # invariant graph
        f = self.conv1(x, idx)
        f = self.conv2(f, idx)
        n = self.act_out(self.lin_out(f))                     # [B, C, 3, N]

        r = P.transpose(1, 2).unsqueeze(1)                    # [B, 1, 3, N]
        if self.mode == 'origin':
            ref = r
        else:
            ref = r - c.transpose(1, 2).unsqueeze(1)          # r - c
        m = torch.cross(ref.expand_as(n), n, dim=2)           # [B, C, 3, N]
        xi = torch.cat([n, m], dim=2)                         # [B, C, 6, N]
        V = xi.mean(dim=-1)                                   # [B, C, 6]

        if self.mode == 'anchor_transport':
            # Ad_{(I,c)} in [omega; v] order: v <- v + c x omega
            ch = hat3(c.squeeze(1))                           # [B, 3, 3]
            V = torch.cat([V[:, :, 0:3],
                           V[:, :, 3:6] +
                           torch.einsum('bij,bcj->bci', ch, V[:, :, 0:3])],
                          dim=-1)
        return V.unsqueeze(-1)                                # [B, C, 6, 1]


# ------------------------------------------- (3) wrench (covector) encoders
# A pure force with direction n acting along the line through r has the
# origin-referenced wrench
#
#     W(r, n) = (m, f) = (r x n, n),        stored [m; f] as in models.py.
#
# Under T = (R, p):  f' = R n = R f  and
#     m' = (Rr + p) x Rn = R(r x n) + p x Rn = R m + p x R f,
# i.e.  W(T.(r, n)) = Ad_T^{-T} W(r, n)  — the coadjoint law, obtained with no
# Q constructed anywhere.  Same numbers as the twist Pluecker lift, opposite
# slot order and opposite transformation type: the Pluecker coordinates of a
# line are a zero-pitch twist when read [omega; v] and a pure-force wrench
# when read [m; f].

class WrenchPlueckerEncoder(nn.Module):
    """Closed-form pairwise pure-force lifting: channel c = rank-c nearest
    neighbor, f = d_ij, m = r_i x d_ij.  Output W(T.P) = Ad_T^{-T} W(P)."""

    def __init__(self, k=8):
        super().__init__()
        self.k = k

    def forward(self, P):
        """P: [B, N, 3] -> W: [B, k, 6, 1] stored [m; f]"""
        B, N, _ = P.shape
        idx = knn_indices(P, self.k)                          # [B, N, k]
        nbr = torch.gather(P.unsqueeze(2).expand(B, N, self.k, 3), 1,
                           idx.unsqueeze(-1).expand(B, N, self.k, 3))
        f = nbr - P.unsqueeze(2)                              # [B, N, k, 3]
        m = torch.cross(P.unsqueeze(2).expand_as(f), f, dim=-1)  # r_i x d_ij
        W = torch.cat([m, f], dim=-1).mean(dim=1)             # [B, k, 6]
        return W.unsqueeze(-1)                                # [B, k, 6, 1]


class WrenchLearnableLiftEncoder(nn.Module):
    """VN-DGCNN direction field + pure-force lifting into se(3)*.

    The direction field is identical to LearnableLiftEncoder (translation-
    invariant + SO(3)-equivariant); only the lift changes: f = n and
    m = (r - ref) x n, with the anchor moment transported back to the origin
    by the coadjoint  Ad_{(I,c)}^{-T} : (m, f) -> (m + c x f, f).

    mode:
      'origin'           m = r x n directly
      'anchor_transport' m = (r - c) x n, then m <- m + c x f  (identical)
      'no_transport'     anchor lift WITHOUT transport — negative control,
                         only SO(3) x SO(3)-equivariant.
    """

    def __init__(self, out_channels=8, hidden=8, k=8, mode='anchor_transport'):
        super().__init__()
        assert mode in ('origin', 'anchor_transport', 'no_transport')
        self.k = k
        self.mode = mode
        self.conv1 = VNEdgeConv(1, hidden)
        self.conv2 = VNEdgeConv(hidden, hidden)
        self.lin_out = VNLinear(hidden, out_channels)
        self.act_out = VNLeakyReLU(out_channels)

    def forward(self, P):
        """P: [B, N, 3] -> W: [B, C, 6, 1] stored [m; f]"""
        B, N, _ = P.shape
        c = P.mean(dim=1, keepdim=True)                       # [B, 1, 3]
        x = (P - c).transpose(1, 2).unsqueeze(1)              # [B, 1, 3, N]
        idx = knn_indices(P, self.k)                          # invariant graph
        h = self.conv1(x, idx)
        h = self.conv2(h, idx)
        n = self.act_out(self.lin_out(h))                     # [B, C, 3, N]

        r = P.transpose(1, 2).unsqueeze(1)                    # [B, 1, 3, N]
        if self.mode == 'origin':
            ref = r
        else:
            ref = r - c.transpose(1, 2).unsqueeze(1)          # r - c
        m = torch.cross(ref.expand_as(n), n, dim=2)           # [B, C, 3, N]
        W = torch.cat([m, n], dim=2).mean(dim=-1)             # [B, C, 6], [m; f]

        if self.mode == 'anchor_transport':
            # Ad_{(I,c)}^{-T} in [m; f] order: m <- m + c x f
            ch = hat3(c.squeeze(1))                           # [B, 3, 3]
            W = torch.cat([W[:, :, 0:3] +
                           torch.einsum('bij,bcj->bci', ch, W[:, :, 3:6]),
                           W[:, :, 3:6]], dim=-1)
        return W.unsqueeze(-1)                                # [B, C, 6, 1]


class WrenchEdgeEncoder(nn.Module):
    """Keep every pure-force wrench until the final tensor pooling.

    Unlike :class:`WrenchPlueckerEncoder`, this encoder does *not* average the
    point dimension into global vector channels.  Its output has the standard
    Lie-Neuron layout ``[B, C, 6, M]``:

    - ``graph='knn'``: ``C=k`` neighbor ranks and ``M=N`` anchor points.  A
      second-moment head is invariant to permutations of the k channels, but a
      tie cut by the k-th-neighbor boundary can still change the selected set.
    - ``graph='kernel'``: ``C=min(candidate_k,N-1)`` local candidates and
      ``M=N`` anchors.  The second-moment head applies a compact kernel that
      is zero at the outer candidate, removing the hard boundary jump while
      retaining bounded local storage.
    - ``graph='all'``: ``C=1`` and ``M=N(N-1)`` directed edges.  There is no
      neighbor ranking or boundary, so exact distance ties are harmless.

    Wrenches are stored as ``[m; f]`` with ``f=r_j-r_i`` and
    ``m=r_i x f`` and therefore transform by ``Ad_T^{-T}``.
    """

    def __init__(self, k=8, graph='knn', candidate_k=None):
        super().__init__()
        if graph not in ('knn', 'kernel', 'all'):
            raise ValueError(
                f"unknown graph={graph!r}; expected 'knn', 'kernel', or 'all'")
        self.k = k
        self.graph = graph
        self.candidate_k = candidate_k if candidate_k is not None else 4 * k

    def forward(self, P):
        """P: [B, N, 3] -> W: [B, k, 6, N] or [B, 1, 6, N(N-1)]."""
        B, N, _ = P.shape
        if self.graph in ('knn', 'kernel'):
            edge_k = self.k if self.graph == 'knn' else self.candidate_k
            edge_k = min(edge_k, N - 1)
            idx = knn_indices(P, edge_k)
            nbr = torch.gather(
                P.unsqueeze(2).expand(B, N, edge_k, 3), 1,
                idx.unsqueeze(-1).expand(B, N, edge_k, 3))
            f = nbr - P.unsqueeze(2)                          # [B, N, K, 3]
            m = torch.cross(P.unsqueeze(2).expand_as(f), f, dim=-1)
            return torch.cat([m, f], dim=-1).permute(0, 2, 3, 1)

        # All ordered pairs i != j.  The edge order may change under a point
        # permutation, but the second-moment sum in the head is order-free.
        f = P.unsqueeze(1) - P.unsqueeze(2)                   # [B, N_i, N_j, 3]
        mask = ~torch.eye(N, dtype=torch.bool, device=P.device)
        f = f[:, mask].reshape(B, N * (N - 1), 3)
        r = P.unsqueeze(2).expand(B, N, N, 3)
        r = r[:, mask].reshape(B, N * (N - 1), 3)
        m = torch.cross(r, f, dim=-1)
        return torch.cat([m, f], dim=-1).transpose(1, 2).unsqueeze(1)

class BracketPlueckerEncoder(nn.Module):
    """Wrench Pluecker lift + PRE-POOLING covector-bracket channels (design A).

    Parity fix for the rank-collapse (bracket_blockage_analysis.md §3.3.4,
    project summary §7-1A): under the antipodal map d -> -d the first-moment
    f-channels are ODD and cancel on symmetric clouds, while the per-point
    bracket output f = d_a x d_b flips sign twice -> EVEN, so it survives
    mean pooling.  Geometrically d_a x d_b is a local surface-normal
    (pseudo-vector) direction.

    Channels: k first-moment wrenches (odd f / even m) + (k-1) consecutive
    rank-pair bracket wrenches (even f / odd m) = 2k-1 total — the two
    parity families cover both slots regardless of symmetry.

    NOTE (stabilizer limit): PROPER symmetries in SE(3) (e.g. a C2 axis)
    confine *any* equivariant vector channel to the fixed subspace — this
    encoder fixes improper (-I) collapse but provably cannot fix C2.
    """

    def __init__(self, k=8):
        super().__init__()
        self.k = k
        self.out_channels = 2 * k - 1

    def forward(self, P):
        """P: [B, N, 3] -> W: [B, 2k-1, 6, 1] stored [m; f]"""
        B, N, _ = P.shape
        idx = knn_indices(P, self.k)
        nbr = torch.gather(P.unsqueeze(2).expand(B, N, self.k, 3), 1,
                           idx.unsqueeze(-1).expand(B, N, self.k, 3))
        f = nbr - P.unsqueeze(2)                              # [B, N, k, 3]
        m = torch.cross(P.unsqueeze(2).expand_as(f), f, dim=-1)
        w = torch.cat([m, f], dim=-1)                         # [B, N, k, 6]

        # per-point covector bracket of consecutive rank pairs (a, a+1)
        fa, ma = f[:, :, :-1], m[:, :, :-1]
        fb, mb = f[:, :, 1:], m[:, :, 1:]
        br_f = torch.cross(fa, fb, dim=-1)                    # 짝수 (even)
        br_m = (torch.cross(fa, mb, dim=-1)
                - torch.cross(fb, ma, dim=-1))                # 홀수 (odd)
        wbr = torch.cat([br_m, br_f], dim=-1)                 # [B, N, k-1, 6]

        W = torch.cat([w, wbr], dim=2).mean(dim=1)            # [B, 2k-1, 6]
        return W.unsqueeze(-1)

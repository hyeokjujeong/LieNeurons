"""Point cloud -> se(3) feature encoders.

Two constructions of the same underlying map (the origin-referenced Pluecker /
screw lifting  L0(r, n) = (r x n, n)  in [v; omega] storage order):

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
        xi = torch.cat([m, d], dim=-1)                       # [B, N, k, 6], [v; omega]
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
        xi = torch.cat([m, n], dim=2)                         # [B, C, 6, N]
        V = xi.mean(dim=-1)                                   # [B, C, 6]

        if self.mode == 'anchor_transport':
            # Ad_{(I,c)} in [v; omega] order: v <- v + c x omega
            ch = hat3(c.squeeze(1))                           # [B, 3, 3]
            V = torch.cat([V[:, :, 0:3] +
                           torch.einsum('bij,bcj->bci', ch, V[:, :, 3:6]),
                           V[:, :, 3:6]], dim=-1)
        return V.unsqueeze(-1)                                # [B, C, 6, 1]


# ------------------------------------------- (3) wrench (covector) encoders
# A pure force with direction n acting along the line through r has the
# origin-referenced wrench
#
#     W(r, n) = (m, f) = (r x n, n),        stored [f; m] as in models.py.
#
# Under T = (R, p):  f' = R n = R f  and
#     m' = (Rr + p) x Rn = R(r x n) + p x Rn = R m + p x R f,
# i.e.  W(T.(r, n)) = Ad_T^{-T} W(r, n)  — the coadjoint law, obtained with no
# Q constructed anywhere.  Same numbers as the twist Pluecker lift, opposite
# slot order and opposite transformation type: the Pluecker coordinates of a
# line are a zero-pitch twist when read [v; omega] and a pure-force wrench
# when read [f; m].

class WrenchPlueckerEncoder(nn.Module):
    """Closed-form pairwise pure-force lifting: channel c = rank-c nearest
    neighbor, f = d_ij, m = r_i x d_ij.  Output W(T.P) = Ad_T^{-T} W(P)."""

    def __init__(self, k=8):
        super().__init__()
        self.k = k

    def forward(self, P):
        """P: [B, N, 3] -> W: [B, k, 6, 1] stored [f; m]"""
        B, N, _ = P.shape
        idx = knn_indices(P, self.k)                          # [B, N, k]
        nbr = torch.gather(P.unsqueeze(2).expand(B, N, self.k, 3), 1,
                           idx.unsqueeze(-1).expand(B, N, self.k, 3))
        f = nbr - P.unsqueeze(2)                              # [B, N, k, 3]
        m = torch.cross(P.unsqueeze(2).expand_as(f), f, dim=-1)  # r_i x d_ij
        W = torch.cat([f, m], dim=-1).mean(dim=1)             # [B, k, 6]
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
        """P: [B, N, 3] -> W: [B, C, 6, 1] stored [f; m]"""
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
        W = torch.cat([n, m], dim=2).mean(dim=-1)             # [B, C, 6], [f; m]

        if self.mode == 'anchor_transport':
            # Ad_{(I,c)}^{-T} in [f; m] order: m <- m + c x f
            ch = hat3(c.squeeze(1))                           # [B, 3, 3]
            W = torch.cat([W[:, :, 0:3],
                           W[:, :, 3:6] +
                           torch.einsum('bij,bcj->bci', ch, W[:, :, 0:3])],
                          dim=-1)
        return W.unsqueeze(-1)                                # [B, C, 6, 1]

"""Models for experiments A, B, C.

Backbone: Lie Neurons LNLinear + LNLieBracket only (algebra_type='se3').
Feature layout follows the repo: x in [B, F, 6, N] with x[..., 0:3, :] = v,
x[..., 3:6, :] = omega.

Heads:
  A (compliance-type):  C = Z Z^T / C_out          ->  Ad_T C Ad_T^T
  B (stiffness-type):   Y = Q Z,  K = Y Y^T / C_out ->  Ad_T^{-T} K Ad_T^{-1}
  C (covector input):   W -> Q^{-1}W (raise to twists) -> backbone
                          -> Q(.) (lower back) -> K

Q : se(3) -> se(3)* is the flat map (Klein Gram matrix), Q^{-1} : se(3)* ->
se(3) the sharp map.  Numerically equal (Q^2 = I) but distinct as maps.
"""
import torch
import torch.nn as nn

from core.lie_neurons_layers import LNLinear, LNLinearAndLieBracket
from experiment.pc_se3_congruence.se3_utils import klein_Q, klein_Q_inv


class Backbone(nn.Module):
    """Stack of LNLinear + LNLieBracket blocks (the only two layer types)."""

    def __init__(self, channels=(8, 16, 16, 8)):
        super().__init__()
        self.blocks = nn.ModuleList([
            LNLinearAndLieBracket(channels[i], channels[i + 1], algebra_type='se3')
            for i in range(len(channels) - 1)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class GramHeadA(nn.Module):
    """C = Z Z^T / C_out.  Since Z -> Ad_T Z, C -> Ad_T C Ad_T^T."""

    def forward(self, z):
        Z = z.squeeze(-1).transpose(1, 2)                     # [B, 6, C]
        return Z @ Z.transpose(1, 2) / Z.shape[-1]            # [B, 6, 6]


class KleinHeadB(nn.Module):
    """Y = Q Z maps twists to wrenches (Q Ad_T = Ad_T^{-T} Q), then
    K = Y Y^T / C_out transforms by congruence Ad_T^{-T} K Ad_T^{-1}."""

    def forward(self, z):
        Z = z.squeeze(-1).transpose(1, 2)                     # [B, 6, C]
        Y = klein_Q(dtype=Z.dtype, device=Z.device) @ Z
        return Y @ Y.transpose(1, 2) / Y.shape[-1]


class ModelA(nn.Module):
    def __init__(self, encoder, channels=(8, 16, 16, 8)):
        super().__init__()
        self.encoder = encoder
        self.backbone = Backbone(channels)
        self.head = GramHeadA()

    def forward(self, P):
        return self.head(self.backbone(self.encoder(P)))


class ModelB(nn.Module):
    def __init__(self, encoder, channels=(8, 16, 16, 8)):
        super().__init__()
        self.encoder = encoder
        self.backbone = Backbone(channels)
        self.head = KleinHeadB()

    def forward(self, P):
        return self.head(self.backbone(self.encoder(P)))


class ModelC(nn.Module):
    """Input: covector (wrench) features W in [B, C, 6, 1] transforming as
    W -> Ad_T^{-T} W.

    The SHARP map Q^{-1} : se(3)* -> se(3) raises them to twists, because
    Q^{-1} Ad_T^{-T} Q = Ad_T, so X = Q^{-1} W obeys X -> Ad_T X and the twist
    backbone runs unchanged.  The head then lowers back with the FLAT map Q
    (Y = Q Z) before forming the Gram, giving the congruence law.

    Q^{-1} and Q are the same matrix here (Q^2 = I), but they are maps between
    different spaces and are written distinctly for that reason.
    """

    def __init__(self, channels=(8, 16, 16, 8)):
        super().__init__()
        self.backbone = Backbone(channels)
        self.head = KleinHeadB()

    def forward(self, W):
        Qinv = klein_Q_inv(dtype=W.dtype, device=W.device)
        X = torch.einsum('ij,bcjn->bcin', Qinv, W)            # wrench -> twist
        return self.head(self.backbone(X))


# ------------------------------------------- covector-space (wrench) algebra
# Storage: for TWISTS the repo order is [v; omega], so slot0 = v, slot1 = omega.
# A wrench F pairs with a twist by F^T xi = F_0 . v + F_1 . omega, so to make
# that equal the physical power m . omega + f . v we must store F as [f; m]:
# slot0 = f (force), slot1 = m (moment).  Under the coadjoint action
#     (m, f) -> (R m + p x R f,  R f),
# so for wrenches it is the FORCE slot that is translation-blind -- exactly the
# role omega plays for twists.  The two representations are swapped copies of
# each other, which is what Q implements.

def twist_bracket(X1, X2):
    """[X1, X2] on se(3), features [B, F, 6, N] stored [v; omega]."""
    v1, w1 = X1[:, :, 0:3], X1[:, :, 3:6]
    v2, w2 = X2[:, :, 0:3], X2[:, :, 3:6]
    return torch.cat([torch.cross(w1, v2, dim=2) - torch.cross(w2, v1, dim=2),
                      torch.cross(w1, w2, dim=2)], dim=2)


def covector_bracket(F1, F2):
    """The unique (up to scale) equivariant bilinear map se(3)* x se(3)* ->
    se(3)*, features [B, F, 6, N] stored [f; m]:

        [F1, F2]_* = ( f1 x m2 - f2 x m1 ,  f1 x f2 )   as (m, f)

    Written out with no Q anywhere.  It coincides with Q [Q^-1 F1, Q^-1 F2]
    (verified as experiment C6), which is *why* it exists: Q is an isomorphism
    of representations se(3) -> se(3)*, so it transports the Lie bracket.  A
    Lie algebra without an invariant non-degenerate form admits no such
    operation on its dual at all.
    """
    f1, m1 = F1[:, :, 0:3], F1[:, :, 3:6]
    f2, m2 = F2[:, :, 0:3], F2[:, :, 3:6]
    return torch.cat([torch.cross(f1, f2, dim=2),
                      torch.cross(f1, m2, dim=2) - torch.cross(f2, m1, dim=2)],
                     dim=2)


class LNCovectorBracket(nn.Module):
    """Mirror of LNLieBracket, but using covector_bracket. Native to se(3)*."""

    def __init__(self, in_channels):
        super().__init__()
        self.learn_dir = nn.Linear(in_channels, in_channels, bias=False)
        self.learn_dir2 = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x):
        d = self.learn_dir(x.transpose(1, -1)).transpose(1, -1)
        d2 = self.learn_dir2(x.transpose(1, -1)).transpose(1, -1)
        return x + covector_bracket(d2, d)


class LNLinearAndCovectorBracket(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = LNLinear(in_channels, out_channels)
        self.bracket = LNCovectorBracket(out_channels)

    def forward(self, x):
        return self.bracket(self.linear(x))


class CovectorBackbone(nn.Module):
    """LNLinear + LNCovectorBracket stack, operating entirely in se(3)*."""

    def __init__(self, channels=(8, 16, 16, 8)):
        super().__init__()
        self.blocks = nn.ModuleList([
            LNLinearAndCovectorBracket(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class ModelCNative(nn.Module):
    """Q-FREE covector pipeline: wrench in -> LNLinear + covector bracket ->
    Gram.  Since the features never leave se(3)*, Y -> Ad_T^{-T} Y already, and
    K = YY^T / C obeys the congruence law with no intertwiner anywhere."""

    def __init__(self, channels=(8, 16, 16, 8)):
        super().__init__()
        self.backbone = CovectorBackbone(channels)

    def forward(self, W):
        Y = self.backbone(W).squeeze(-1).transpose(1, 2)       # [B, 6, C]
        return Y @ Y.transpose(1, 2) / Y.shape[-1]


class ModelCNaiveBracket(nn.Module):
    """NEGATIVE CONTROL: run the ordinary twist backbone straight on wrench
    features, i.e. apply the se(3) Lie bracket to covectors as if they were
    twists.  The bracket is a map g x g -> g and is equivariant for Ad, not for
    Ad^{-T}; the two agree only on the orthogonal subgroup p = 0."""

    def __init__(self, channels=(8, 16, 16, 8)):
        super().__init__()
        self.backbone = Backbone(channels)

    def forward(self, W):
        Y = self.backbone(W).squeeze(-1).transpose(1, 2)
        return Y @ Y.transpose(1, 2) / Y.shape[-1]


# ------------------------------------ end-to-end PC -> L, K = L L^T pipeline
class CovectorGramHeadLK(nn.Module):
    """L = Z / sqrt(C) from the final wrench features Z in [B, 6, C], and
    K = L L^T.  Since Z -> Ad_T^{-T} Z, we get  L -> Ad_T^{-T} L  and
    K -> Ad_T^{-T} K Ad_T^{-1}  (congruence) for free.

    Order matters: output L and FORM K from it.  A Cholesky factor of an
    equivariant K would not itself be equivariant (Cholesky depends on the
    basis ordering), so L-first is the only correct direction."""

    def forward(self, z):
        Z = z.squeeze(-1).transpose(1, 2)                     # [B, 6, C]
        L = Z / Z.shape[-1] ** 0.5
        return L, L @ L.transpose(1, 2)


class ModelPC2K(nn.Module):
    """Fully covector-native stiffness pipeline:

        P -> pure-force wrench lift -> LNLinear + covector-bracket backbone
          -> (L, K = L L^T)

    Every feature from the encoder onward lives in se(3)* and transforms by
    the coadjoint Ad_T^{-T}; no Q is constructed anywhere.  Hence
        L(T.P) = Ad_T^{-T} L(P),    K(T.P) = Ad_T^{-T} K(P) Ad_T^{-1}."""

    def __init__(self, encoder, channels=(8, 16, 16, 8)):
        super().__init__()
        self.encoder = encoder
        self.backbone = CovectorBackbone(channels)
        self.head = CovectorGramHeadLK()

    def forward(self, P):
        return self.head(self.backbone(self.encoder(P)))


class ModelPC2KNaiveBracket(nn.Module):
    """NEGATIVE CONTROL: same wrench encoder, but the TWIST backbone (ordinary
    se(3) Lie bracket) run on the wrench features.  Equivariant for Ad, not
    Ad^{-T}: passes at p = 0, fails at O(1) once translation enters."""

    def __init__(self, encoder, channels=(8, 16, 16, 8)):
        super().__init__()
        self.encoder = encoder
        self.backbone = Backbone(channels)
        self.head = CovectorGramHeadLK()

    def forward(self, P):
        return self.head(self.backbone(self.encoder(P)))


# --------------------------------------------------------- negative controls
class NaiveHeadNoKlein(nn.Module):
    """K = Z Z^T without the Klein intertwiner: has the WRONG type — it
    transforms as Ad_T (.) Ad_T^T, not Ad_T^{-T} (.) Ad_T^{-1}."""

    def forward(self, z):
        Z = z.squeeze(-1).transpose(1, 2)
        return Z @ Z.transpose(1, 2) / Z.shape[-1]


def add_bias_to_first_linear(model, gen=None):
    """Break equivariance on purpose: give the first LNLinear a bias term."""
    fc = model.backbone.blocks[0].linear.fc
    biased = nn.Linear(fc.in_features, fc.out_features, bias=True,
                       dtype=fc.weight.dtype)
    with torch.no_grad():
        biased.weight.copy_(fc.weight)
        nn.init.normal_(biased.bias, std=0.5, generator=gen)
    model.backbone.blocks[0].linear.fc = biased
    return model

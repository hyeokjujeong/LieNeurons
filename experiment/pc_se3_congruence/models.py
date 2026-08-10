"""[COMPARISON ARMS] Global-pooling models -- experiments A, B, C.

None of these is the current architecture.  The current one is
:class:`~experiment.pc_se3_congruence.pointwise_models.PointwiseStiffnessModel`.
This module survives because ``blockage_bench.py`` still instantiates five of
these classes as the arms every current result is measured against, so it is
live code -- but nothing new should be built on it.

WHY THEY WERE SUPERSEDED.  Every model here pools the POINT axis N into global
vector channels before the head.  A symmetry of the cloud then has to FIX each
channel rather than permute them, so all channels are forced into Fix_H(rho) and
rank(K) collapses from 6 to 3 on centro-symmetric and C2 clouds.  Keeping the
point axis until the Gram (``pointwise_models.py``) removes the constraint,
because L(T.P) = A_T L(P) (Pi_T (x) I_H) and the permutation gauge cancels in
L L^T.  ``WrenchSecondMomentModel`` additionally indexes channels by NEIGHBOUR
RANK, which is not invariant under relabelling an equal-distance shell; it is
retained precisely as the arm that FAILS the exact-tie test (permutation error
1.9e-01 vs 3.0e-16), which is what test_pc_pointwise_pipeline.py asserts.

Section map (single ``# ====`` banners below mark the boundaries):
  1. twist-path models A/B                      -- Backbone .. ModelC
  2. covector-native path                       -- twist_bracket .. ModelPC2K
  3. rank-channel second moment (generation 2)  -- WrenchSecondMomentModel
  4. Klein-gate backbones                       -- klein_gram .. DualBackbone
  5. negative controls                          -- NaiveHeadNoKlein, add_bias_*

Imported by ``blockage_bench.py``: DualBackbone, GateBackbone, ModelB,
ModelPC2K, WrenchSecondMomentModel.  Everything else is reached only from
``legacy/``.

Backbone: Lie Neurons LNLinear + LNLieBracket only (algebra_type='se3').
Feature layout follows the repo (ANGULAR / MOMENT FIRST): x in [B, F, 6, N]
with x[..., 0:3, :] = omega (twists) or m (wrenches), x[..., 3:6, :] = v or f.

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


# ================================================ 1. twist-path models A and B
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
# Storage: for TWISTS the repo order is [omega; v], so slot0 = omega, slot1 = v.
# A wrench F pairs with a twist by F^T xi = F_0 . omega + F_1 . v, so to make
# that equal the physical power m . omega + f . v we must store F as [m; f]:
# slot0 = m (moment), slot1 = f (force).  Under the coadjoint action
#     (m, f) -> (R m + p x R f,  R f),
# so for wrenches it is the FORCE slot that is translation-blind -- exactly the
# role omega plays for twists.  The two representations are swapped copies of
# each other, which is what Q implements.

# ================================================= 2. covector-native path
def twist_bracket(X1, X2):
    """[X1, X2] on se(3), features [B, F, 6, N] stored [omega; v]:

        [xi1, xi2] = ( w1 x w2 ,  w1 x v2 - w2 x v1 )
    """
    w1, v1 = X1[:, :, 0:3], X1[:, :, 3:6]
    w2, v2 = X2[:, :, 0:3], X2[:, :, 3:6]
    return torch.cat([torch.cross(w1, w2, dim=2),
                      torch.cross(w1, v2, dim=2) - torch.cross(w2, v1, dim=2)],
                     dim=2)


def covector_bracket(F1, F2):
    """The unique (up to scale) equivariant bilinear map se(3)* x se(3)* ->
    se(3)*, features [B, F, 6, N] stored [m; f]:

        [F1, F2]_* = ( f1 x m2 - f2 x m1 ,  f1 x f2 )   as (m, f)

    Written out with no Q anywhere.  It coincides with Q [Q^-1 F1, Q^-1 F2]
    (verified as experiment C6), which is *why* it exists: Q is an isomorphism
    of representations se(3) -> se(3)*, so it transports the Lie bracket.  A
    Lie algebra without an invariant non-degenerate form admits no such
    operation on its dual at all.
    """
    m1, f1 = F1[:, :, 0:3], F1[:, :, 3:6]
    m2, f2 = F2[:, :, 0:3], F2[:, :, 3:6]
    return torch.cat([torch.cross(f1, m2, dim=2) - torch.cross(f2, m1, dim=2),
                      torch.cross(f1, f2, dim=2)], dim=2)


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


# ============================ 3. rank-channel second moment  [GENERATION 2]
# Retained as the arm that FAILS the exact-tie test: it reads neighbour RANK as
# a channel index, which is not invariant under relabelling an equal-distance
# shell.  test_pc_pointwise_pipeline.py asserts that failure.
class WrenchSecondMomentModel(nn.Module):
    """Late second-order pooling of point/edge wrench features.

    For edge wrenches ``W=[m;f]`` transforming by ``A=Ad_T^{-T}``, form

        K = (1 / E) sum_e alpha(||f_e||^2) W_e W_e^T.

    The scalar weight is invariant, hence ``K(T.P)=A K(P) A^T``.  Signs and
    permutations of symmetry-related edges disappear in the outer-product
    sum, so no global equivariant vector or canonical neighbor has to be
    selected.  The result is PSD; it is SPD exactly when the positively
    weighted wrenches span R^6.

    ``weight_mode``:
      learned  positive radial MLP (the trainable experiment)
      analytic exp(-||f||^2/(2 sigma^2)), matching the all-pair/kNN target
      uniform  structural rank/equivariance diagnostic with alpha=1
    """

    def __init__(self, encoder, weight_mode='learned', sigma=0.5,
                 hidden=32, backbone_channels=None):
        super().__init__()
        if weight_mode not in ('learned', 'analytic', 'uniform'):
            raise ValueError('weight_mode must be learned, analytic, or uniform')
        self.encoder = encoder
        self.weight_mode = weight_mode
        self.sigma = sigma
        self.backbone = (CovectorBackbone(backbone_channels)
                         if backbone_channels is not None else None)
        if self.backbone is not None:
            # Start each residual bracket branch at the identity.  Keeping one
            # direction random and zeroing the other preserves a non-zero
            # learning signal while avoiding an unstable random quadratic map
            # before the SPD/AIRM objective has calibrated feature scales.
            for block in self.backbone.blocks:
                nn.init.zeros_(block.bracket.learn_dir2.weight)
        if weight_mode == 'learned':
            self.weight_net = nn.Sequential(
                nn.Linear(1, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, 1))
            # Start from the benign radial kernel exp(-||f||^2), without
            # leaking the target sigma.  The network learns a positive local
            # inverse length scale; zeroing the last weight makes initialization
            # independent of random upstream activations.
            last = self.weight_net[-1]
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias,
                              torch.log(torch.expm1(torch.tensor(1.0))).item())
        else:
            self.weight_net = None

    def edge_weights(self, sqdist):
        """sqdist: [B, C, M] -> positive invariant weights of same shape."""
        if self.weight_mode == 'uniform':
            return torch.ones_like(sqdist)
        if self.weight_mode == 'analytic':
            return torch.exp(-sqdist / (2.0 * self.sigma ** 2))
        # A positive learned inverse length scale guarantees decay instead of
        # letting numerous long all-pair edges dominate at initialization.
        raw = self.weight_net(torch.log1p(sqdist).unsqueeze(-1)).squeeze(-1)
        rate = torch.nn.functional.softplus(raw) + 1e-12
        return torch.exp(-rate * sqdist)

    def forward(self, P):
        W = self.encoder(P)                                  # [B, C, 6, M]
        sqdist = W[:, :, 3:6].square().sum(dim=2)             # [B, C, M]  (f 슬롯)
        radial_alpha = self.edge_weights(sqdist)
        compact_alpha = None
        if getattr(self.encoder, 'graph', None) == 'kernel':
            # Candidate channels are distance ordered for every anchor M.
            # The compact window is exactly zero at the candidate boundary.
            from experiment.pc_se3_congruence.encoders import compact_wendland_weights
            compact_alpha = compact_wendland_weights(
                sqdist, candidate_dim=1)
        alpha = (radial_alpha if compact_alpha is None
                 else radial_alpha * compact_alpha)
        if self.backbone is None:
            K = torch.einsum('bcm,bcim,bcjm->bij', alpha, W, W)
            denom = W.shape[1] * W.shape[-1]
        else:
            # Apply the invariant radial/compact weight before channel mixing.
            # The backbone stays entirely in se(3)*, so every intermediate
            # feature and the final Gram tensor obey the coadjoint congruence
            # law.  This is an explicit rank-channel LN ablation; unlike the
            # no-backbone sum, general channel mixing is not invariant to a
            # permutation of equal-distance neighbor-rank channels.
            # Take the two square roots separately.  The compact window is a
            # fixed graph quantity (no parameter gradient); sqrt(radial*0)
            # would otherwise create the indeterminate 0*inf gradient at its
            # exactly-zero boundary.
            feature_weight = radial_alpha.sqrt()
            if compact_alpha is not None:
                feature_weight = feature_weight * compact_alpha.sqrt()
            X = W * feature_weight.unsqueeze(2)
            expected_channels = self.backbone.blocks[0].linear.fc.in_features
            if X.shape[1] < expected_channels:
                # Small clouds can have N-1 < candidate_k (e.g. N=32,
                # candidate_k=32).  Zero padding keeps one fixed LN parameter
                # shape without adding any edge contribution.
                pad = X.new_zeros(X.shape[0], expected_channels - X.shape[1],
                                  X.shape[2], X.shape[3])
                X = torch.cat([X, pad], dim=1)
            elif X.shape[1] > expected_channels:
                raise ValueError(
                    f'LN backbone expects {expected_channels} edge channels, '
                    f'but encoder produced {X.shape[1]}')
            Z = self.backbone(X)
            K = torch.einsum('bcim,bcjm->bij', Z, Z)
            denom = Z.shape[1] * Z.shape[-1]
        K = K / denom
        return 0.5 * (K + K.transpose(-1, -2))


# ====================================================== 4. Klein-gate backbones
# ----------------------------------------- invariant Gram gating (PDF §6.2)
def klein_gram(x):
    """Pairwise Klein pairings S_cd = x_c^T Q^{-1} x_d over channels.

    Storage-agnostic: for x stored [a; b] (covector [m; f] or twist [w; v]),
    x_c^T Q x_d = a_c . b_d + b_c . a_d, which is the correct invariant of the
    respective representation in both orders (Q is the block swap, Q^{-1}=Q).
    x: [B, C, 6, N] -> S: [B, C, C, N], invariant under the group action.
    """
    a, b = x[:, :, 0:3], x[:, :, 3:6]
    return (torch.einsum('bcin,bdin->bcdn', a, b)
            + torch.einsum('bcin,bdin->bcdn', b, a))


class GramGate(nn.Module):
    """General Q-Gram gate: any function of S(X) = X^T Q^{-1} X modulates
    channels (Coadjoint_Equivariant_Network.pdf §6.2).  Here each channel is
    gated by a small MLP over its row of the Gram matrix; the rank-1 bilinear
    Q-Gram gate  nu(X)_c = x_c phi(q_c^T Q^{-1} k_c)  (eq. 11) is the special
    case where the MLP collapses to a fixed bilinear form.

    The gate multiplies the WHOLE 6-vector channel (slot-wise gating breaks
    equivariance because Ad^{-T} is block-triangular; see
    bracket_blockage_analysis.md §4.1).  Bounded 1 + tanh — never divide by
    invariants (null-cone hazard, PDF Remark 6.5).
    """

    def __init__(self, in_channels, hidden=None):
        super().__init__()
        h = hidden or in_channels
        self.mlp = nn.Sequential(nn.Linear(in_channels, h), nn.Tanh(),
                                 nn.Linear(h, 1))

    def forward(self, x):                              # [B, C, 6, N]
        S = klein_gram(x)                              # [B, C, C, N] invariant
        g = self.mlp(S.permute(0, 1, 3, 2))            # [B, C, N, 1]
        g = 1 + torch.tanh(g.squeeze(-1))              # [B, C, N] bounded
        return g.unsqueeze(2) * x                      # 채널 전체에 곱


class LNLinearAndGramGate(nn.Module):
    """LNLinear + GramGate — bracket 대신 gate를 비선형성으로 쓰는 블록."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = LNLinear(in_channels, out_channels)
        self.gate = GramGate(out_channels)

    def forward(self, x):
        return self.gate(self.linear(x))


class GateBackbone(nn.Module):
    """LNLinear + GramGate stack.  Storage-agnostic (vector/covector 공용)."""

    def __init__(self, channels=(8, 16, 16, 8)):
        super().__init__()
        self.blocks = nn.ModuleList([
            LNLinearAndGramGate(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class DualBackbone(nn.Module):
    """Parallel bracket-branch + gate-branch, merged in a bracket layer.

    Lie Neurons 논문 Appendix C (Fig. 5)의 패턴: 두 병렬 브랜치를 지나온
    feature를 채널 concat 후 bracket 층에서 혼합한다.  method='vector'는
    twist bracket, 'covector'는 covector bracket을 사용.
    """

    def __init__(self, channels=(8, 16, 16, 8), method='vector'):
        super().__init__()
        from core.lie_neurons_layers import LNLinearAndLieBracket
        bracket_block = (
            (lambda i, o: LNLinearAndLieBracket(i, o, algebra_type='se3'))
            if method == 'vector' else LNLinearAndCovectorBracket)
        self.branch_bracket = nn.ModuleList([
            bracket_block(channels[i], channels[i + 1])
            for i in range(len(channels) - 1)])
        self.branch_gate = GateBackbone(channels)
        self.merge = bracket_block(2 * channels[-1], channels[-1])

    def forward(self, x):
        xb = x
        for blk in self.branch_bracket:
            xb = blk(xb)
        xg = self.branch_gate(x)
        return self.merge(torch.cat([xb, xg], dim=1))


# ==================================================== 5. negative controls
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

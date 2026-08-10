"""[CURRENT MODEL]
Pointwise wrench pipeline: keep the POINT axis until the Gram.

    P --local Pluecker lift--> w_ij           [B, N, k, 6]
      --learned invariant set pooling-------> X^(0)  [B, C_0, 6, N]
      --LN-Linear / covector bracket / Klein gate (+ optional message passing)
                                            -> X^(L)  [B, C_L, 6, N]
      --factor head-------------------------> Z       [B, H, 6, N]
      --late second moment------------------> K       [B, 6, 6]

WHAT IS DIFFERENT FROM THE EXISTING MODELS.

1.  The neighbour axis k is removed FIRST, by permutation-invariant learned set
    aggregation, and the point axis N is kept until the Gram.  The old encoders
    did the opposite: they averaged the point axis into global vector channels
    and used neighbour RANK as the channel index.  Rank channels are not
    invariant under a relabelling of an equal-distance shell, which is what
    broke the LN backbone ablation of :class:`WrenchSecondMomentModel`.

2.  Because the factor index is (i, h) and a symmetry of the cloud PERMUTES the
    points, symmetry no longer has to fix each factor -- it may permute them.
    With L = [sqrt(beta_ih) z_ih] in R^{6 x NH},

        L(T.P) = A_T L(P) (Pi_T (x) I_H),   A_T = Ad_T^{-T},

    and Pi_T (x) I_H is orthogonal, so K = L L^T / Z is unchanged by the gauge:

        K(T.P) = A_T K(P) A_T^T.

    Global vector pooling destroys that gauge and forces every channel into
    Fix_H(rho), which is what collapsed the rank on centro / C2 clouds.

3.  K is a second moment of LATENT covectors, not of raw Plucker wrenches.  A
    single covector bracket is already bilinear in the lift, so z_ih is a
    higher-order (and, through the gates, non-polynomial) function of P.

EQUIVARIANCE LEDGER (each line is checked in test_pc_pointwise_pipeline.py):
    lift               w_ij(T.P) = A_T w_pi(i)pi(j)(P)
    set pooling        invariant weights x coadjoint vectors
    LNLinear           mixes the CHANNEL axis only
    covector bracket   the unique equivariant bilinear map on se(3)*
    Klein gate         invariant scalar x the WHOLE 6-vector (never per slot:
                       Ad^{-T} is block triangular, so slotwise gating breaks)
    message passing    invariant scalar weights x neighbour covectors
    head               sum of beta z z^T with invariant beta

MEMORY.  The [B, N, k, 6] edge tensor exists only inside the encoder; the
backbone carries [B, C, 6, N].  Optional message passing re-materialises
[B, C_msg, 6, N, k], which is the one place where the k axis reappears -- keep
C_msg small (default 8) or leave it off.
"""
import torch
import torch.nn as nn

from core.lie_neurons_layers import LNLinear
from experiment.pc_se3_congruence.models import covector_bracket, klein_gram
from experiment.pc_se3_congruence.pointwise_graph import (EdgeInvariants,
                                                          build_local_graph)

_EPS = 1e-12

POOL_MODES = ('basis', 'basis_mean', 'attention', 'sum', 'mean')
BRACKET_MODES = ('none', 'separable', 'pairwise')
GATE_MODES = ('none', 'projected', 'full')
NORMALIZE_MODES = ('nh', 'beta', 'one')


# ------------------------------------------------------------------ helpers
def gather_neighbors(x, idx):
    """x: [B, C, 6, N], idx: [B, N, k] -> [B, C, 6, N, k]."""
    B, C, D, N = x.shape
    k = idx.shape[-1]
    index = idx.reshape(B, 1, 1, N * k).expand(B, C, D, N * k)
    return torch.gather(x, 3, index).reshape(B, C, D, N, k)


def klein_pair(a, b):
    """Klein pairing of two covector features stored [m; f] along dim 2.

    <a, b> = m_a . f_b + f_a . m_b.  Invariant under the coadjoint action: the
    two cross terms  Rf_a . (p x Rf_b)  and  (p x Rf_a) . Rf_b  are opposite
    determinants and cancel.  Contracts dim 2, so it works for both [B,C,6,N]
    and the gathered [B,C,6,N,k].
    """
    return ((a[:, :, 0:3] * b[:, :, 3:6]).sum(2)
            + (a[:, :, 3:6] * b[:, :, 0:3]).sum(2))


def force_pair(a, b):
    """f_a . f_b -- also invariant, because the force slot is translation blind
    (f -> R f).  It is the pairing induced by the equivariant projection
    N_*: (m, f) -> (0, f), the second element of the 2-dimensional space of
    equivariant bilinear maps on se(3)*.  Storage is [m; f], so f is slot 3:6."""
    return (a[:, :, 3:6] * b[:, :, 3:6]).sum(2)


def bounded_invariant(s):
    """Monotone squashing sign(s) log(1+|s|) of an invariant scalar.

    Klein pairings scale like |X|^2 and would otherwise saturate the gate MLPs
    as feature magnitudes drift during training.  A monotone function of an
    invariant is still an invariant, so nothing structural is spent here.
    """
    return torch.sign(s) * torch.log1p(s.abs())


def _mlp(in_dim, hidden, out_dim, depth=2):
    layers, d = [], in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.Tanh()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def _levi_civita(dtype=None):
    e = torch.zeros(3, 3, 3, dtype=dtype or torch.get_default_dtype())
    for i, a, b in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        e[i, a, b] = 1.0
        e[i, b, a] = -1.0
    return e


# ------------------------------------------------- (1) local set aggregation
class LocalWrenchSetEncoder(nn.Module):
    """[B, N, k, 6] edge wrenches -> [B, C_0, 6, N] point covector channels.

    Every mode has the same shape,

        X^(0)_{i,c} = sum_j a_{ij,c} w_ij,

    with an INVARIANT weight a and a sum over the neighbour SET.  What differs
    is how a is built.

    The point of this layer is not the weighting -- it is that the k axis is
    removed by a set reduction rather than by reading neighbour RANK as a
    channel index.  Rank is a discontinuous labelling of the neighbour set
    (it jumps at a distance tie); a distance-shell weight is a continuous one.
    Both turn k into C_0, only the second one is tie-safe.

    pool:
      basis_mean the simple DEFAULT, and it has NO PARAMETERS:

                     a_{ij,c} = window(q_ij) rho_c(q_ij) / sum_l (same),

                 with FIXED Gaussian shells rho_c.  Channel c is "the
                 neighbours near normalised distance c".  Nothing is lost by
                 not learning these weights, because the LNLinear that
                 immediately follows recovers any edge weight lying in
                 span{rho_c}:

                     sum_c W_{c'c} sum_j rho_c(q_ij) w_ij
                       = sum_j [sum_c W_{c'c} rho_c(q_ij)] w_ij,

                 i.e. the learned radial kernel just moves one layer later.
                 The per-shell normalisation is the one part that does NOT
                 commute out that way -- a ratio is not a linear combination of
                 shells -- so it is kept here: it makes a channel independent of
                 how many neighbours happen to land in its shell.
      basis      the same shells without the normalisation.
      attention  a = window * exp(eta) / sum_l window * exp(eta), eta an MLP of
                 all 13 edge invariants.  Buys a nonlinear, density-aware
                 weight; costs an MLP.  The window sits inside the
                 normalisation, otherwise an edge crossing the support boundary
                 would still move the denominator by O(1).
      sum        a = window * Phi
      mean       a = window * Phi / soft degree

    bracket (extra channels concatenated after the linear ones):
      none       the DEFAULT: linear channels only.  The encoder then has
                 EXACTLY ZERO parameters -- it is a fixed lift whose only job is
                 to produce something of the right mathematical type.
                 `separable` is redundant here: block 0 already computes
                 [X W_0 V_0, X W_0 U_0]_*, and W_0 U_0 ranges over every map
                 C_0 -> C_1, so it contains whatever the encoder bracket would
                 have produced.  Same argument as the pooling weights above --
                 anything the following LN block can absorb belongs there, not
                 here.
      separable  [X_{c1}, X_{c2}]_* on the pooled channels.  By bilinearity this
                 IS the double sum sum_{j,l} a_{ij,c1} a_{il,c2} [w_ij, w_il]_*,
                 i.e. the rank-1-coefficient case of design B.  Cost O(k).
      pairwise   genuine non-separable double sum with pair-invariant
                 coefficients b(q_j, q_l, cos) -- the only variant that survives
                 a locally antipodal neighbourhood, where every first moment
                 (and hence every separable bracket) vanishes.  Cost O(k^2):
                 the coefficient tensor is [B, N, k, k, C_b], so keep C_b small.
    """

    def __init__(self, out_channels=8, hidden=32, n_rbf=8, pool='basis_mean',
                 bracket='none', bracket_channels=None, bracket_hidden=16,
                 depth=2):
        super().__init__()
        if pool not in POOL_MODES:
            raise ValueError(f'pool must be one of {POOL_MODES}')
        if bracket not in BRACKET_MODES:
            raise ValueError(f'bracket must be one of {BRACKET_MODES}')
        self.pool = pool
        self.bracket = bracket
        self.linear_channels = out_channels
        if pool in ('basis', 'basis_mean'):
            # Fixed distance shells on the normalised radius; no parameters.
            self.invariants = None
            self.weight_mlp = None
            self.register_buffer('shell_centers',
                                 torch.linspace(0.0, 1.0, out_channels))
            self.shell_width = 1.0 / max(out_channels - 1, 1)
        else:
            self.invariants = EdgeInvariants(n_rbf)
            self.weight_mlp = _mlp(self.invariants.dim, hidden, out_channels,
                                   depth)

        cb = out_channels if bracket_channels is None else bracket_channels
        self.bracket_channels = cb if bracket != 'none' else 0
        if bracket == 'separable':
            self.proj_u = LNLinear(out_channels, cb)
            self.proj_v = LNLinear(out_channels, cb)
        elif bracket == 'pairwise':
            # (q_j, q_l, cos_jl, q_j q_l); only the part of b that is
            # ANTISYMMETRIC in (j, l) survives, because [w_j, w_l]_* is.
            self.pair_mlp = _mlp(4, bracket_hidden, cb, depth=1)
            self.register_buffer('eps3', _levi_civita())

    @property
    def out_channels(self):
        return self.linear_channels + self.bracket_channels

    # ---------------------------------------------------------------- pieces
    def edge_wrenches(self, P, graph):
        """P: [B, N, 3] -> w: [B, N, k, 6] stored [m; f].

        f = d_ij = p_j - p_i,  m = p_i x d_ij.  Under T = (R, p):
        f -> R f and m -> R m + p x R f, i.e. w -> Ad_T^{-T} w with no Q built.
        """
        d = graph.edge_vec
        m = torch.cross(P.unsqueeze(2).expand_as(d), d, dim=-1)
        return torch.cat([m, d], dim=-1)

    def pooling_weights(self, graph):
        """:class:`LocalGraph` -> a: [B, N, k, C] invariant pooling weights."""
        if self.pool in ('basis', 'basis_mean'):
            q = graph.q.unsqueeze(-1)                           # [B, N, k, 1]
            centers = self.shell_centers.to(q.dtype).view(1, 1, 1, -1)
            shells = torch.exp(-((q - centers) / self.shell_width).square())
            a = graph.window.unsqueeze(-1) * shells             # [B, N, k, C]
            if self.pool == 'basis_mean':
                # Per-shell convex combination: divides out how many neighbours
                # happen to fall in that shell, so an anchor in a dense region
                # does not simply get a larger feature.  This is the one part of
                # `attention` that is not recovered by the following LNLinear --
                # a ratio is not a linear combination of the shells.
                a = a / (a.sum(dim=2, keepdim=True) + _EPS)
            return a
        s = self.invariants(graph)                              # [B, N, k, F]
        raw = self.weight_mlp(s)                                # [B, N, k, C]
        win = graph.window.unsqueeze(-1)                        # [B, N, k, 1]
        if self.pool == 'attention':
            # Stabilise against the IN-SUPPORT maximum only.  An out-of-support
            # logit must not set the shift (it would underflow every real
            # weight), and an anchor with an empty support must return zeros
            # rather than 0 * inf.
            floor = torch.finfo(raw.dtype).min / 4
            shift = torch.where(win > 0, raw, torch.full_like(raw, floor)
                                ).amax(dim=2, keepdim=True)
            num = win * torch.exp((raw - shift).clamp(max=0.0))
            return num / (num.sum(dim=2, keepdim=True) + _EPS)
        if self.pool == 'mean':
            deg = win.sum(dim=2, keepdim=True)
            return win * raw / (deg + _EPS)
        return win * raw

    def pairwise_bracket(self, w, graph):
        """sum_{j,l} b_{ijl,c} [w_ij, w_il]_*  ->  [B, C_b, 6, N].

        The double sum is never materialised as 6-vectors: with
        T^{ab}_{c} = sum_{jl} b_{jl,c} u_j^a v_l^b the cross products are
        contracted through the Levi-Civita symbol, so the largest intermediate
        is [B, N, k, C_b, 3] instead of [B, N, k, k, 6].
        """
        m, f = w[..., 0:3], w[..., 3:6]
        q = graph.q
        n = f / f.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        cos = torch.einsum('bnjx,bnlx->bnjl', n, n)
        qj = q.unsqueeze(-1).expand_as(cos)
        ql = q.unsqueeze(-2).expand_as(cos)
        feats = torch.stack([qj, ql, cos, qj * ql], dim=-1)      # [B,N,k,k,4]
        coef = self.pair_mlp(feats)                              # [B,N,k,k,C_b]
        coef = coef * (graph.window.unsqueeze(-1) * graph.window.unsqueeze(-2)
                       ).unsqueeze(-1)

        eps3 = self.eps3.to(w.dtype)

        def contract(a1, a2):
            # sum_{jl} coef_{jl,c} (a1_j x a2_l), routed through the Levi-Civita
            # symbol so the largest intermediate is [B, N, k, C_b, 3].
            t = torch.einsum('bnjlc,bnjx,bnly->bncxy', coef, a1, a2)
            return torch.einsum('ixy,bncxy->bcin', eps3, t)      # [B,C_b,3,N]

        out_f = contract(f, f)                                   # f_j x f_l
        # sum coef (f_j x m_l - f_l x m_j); the second term equals -eps(T_mf).
        out_m = contract(f, m) + contract(m, f)
        # 저장은 [m; f] 이므로 moment 슬롯이 먼저다.
        return torch.cat([out_m, out_f], dim=2)                  # [B, C_b, 6, N]

    # --------------------------------------------------------------- forward
    def forward(self, P, graph):
        """P: [B, N, 3], graph -> X^(0): [B, C_0, 6, N]."""
        w = self.edge_wrenches(P, graph)                         # [B, N, k, 6]
        a = self.pooling_weights(graph)                          # [B, N, k, C]
        x = torch.einsum('bnkc,bnkj->bcjn', a, w)                # [B, C, 6, N]
        if self.bracket == 'separable':
            x = torch.cat([x, covector_bracket(self.proj_u(x), self.proj_v(x))],
                          dim=1)
        elif self.bracket == 'pairwise':
            x = torch.cat([x, self.pairwise_bracket(w, graph)], dim=1)
        return x


# --------------------------------------------------- (2) invariant messages
class InvariantMessagePassing(nn.Module):
    """M_i = sum_j gamma_ij (X_j W_m),  gamma invariant -> M is coadjoint.

    gamma is built from edge invariants and from projected Klein invariants of
    the endpoints,

        s_i = <U_i, V_i>,   cross_ij = <U_i, V_j>,

    with U = X R, V = X S small channel projections.  Everything fed to the MLP
    is a scalar invariant, so the message inherits the coadjoint law from X_j.

    Memory: gathers [B, C_msg, 6, N, k].  Keep C_msg small.
    """

    def __init__(self, channels, msg_channels=8, hidden=32, n_proj=4, n_rbf=8):
        super().__init__()
        self.value = LNLinear(channels, msg_channels)
        self.proj_u = LNLinear(channels, n_proj)
        self.proj_v = LNLinear(channels, n_proj)
        self.invariants = EdgeInvariants(n_rbf)
        self.gamma_mlp = _mlp(self.invariants.dim + 3 * n_proj, hidden,
                              msg_channels)
        self.out = LNLinear(msg_channels, channels)

    def forward(self, x, graph):
        """x: [B, C, 6, N] -> M: [B, C, 6, N]."""
        idx, win = graph.idx, graph.window
        B, N, k = idx.shape
        u, v = self.proj_u(x), self.proj_v(x)                    # [B, P, 6, N]
        v_j = gather_neighbors(v, idx)                           # [B,P,6,N,k]
        s_i = bounded_invariant(klein_pair(u, v))                # [B, P, N]
        s_j = torch.gather(s_i, 2, idx.reshape(B, 1, N * k)
                           .expand(B, s_i.shape[1], N * k)
                           ).reshape(B, s_i.shape[1], N, k)
        cross = bounded_invariant(klein_pair(u.unsqueeze(-1), v_j))

        edge_feat = self.invariants(graph)                       # [B, N, k, F]
        feats = torch.cat([edge_feat,
                           s_i.permute(0, 2, 1).unsqueeze(2).expand(
                               -1, -1, k, -1),
                           s_j.permute(0, 2, 3, 1),
                           cross.permute(0, 2, 3, 1)], dim=-1)
        gamma = self.gamma_mlp(feats) * win.unsqueeze(-1)        # [B,N,k,C_msg]
        deg = win.sum(-1).clamp_min(_EPS)                        # [B, N]

        y_j = gather_neighbors(self.value(x), idx)               # [B,Cm,6,N,k]
        msg = torch.einsum('bnkc,bcjnk->bcjn', gamma, y_j)
        return self.out(msg / deg.view(B, 1, 1, N))


# -------------------------------------------------------- (3) Klein-form gate
class KleinGate(nn.Module):
    """g_{i,c} = 1 + tanh( MLP( S_i, s_global ) ) applied to the WHOLE channel.

    The gate multiplies all six components of a channel.  Gating the f and m
    slots separately breaks equivariance because Ad_T^{-T} is block triangular:
    it mixes f into m, so the two slots cannot carry independent scalars.

    gram:
      projected  S_i = diag(U_i^T Q^{-1} V_i) in R^P from two channel
                 projections -- O(P) instead of the full O(C^2) Gram
      full       S_i = X_i^T Q^{-1} X_i in R^{C x C}, gated row by row (the
                 construction of models.GramGate)

    use_force adds the second invariant family f_c . f_d, which exists because
    the force slot is translation blind.  It is legal (a genuine invariant of
    the coadjoint action) but off by default so the gate stays exactly the
    Klein-form gate of the framework note.

    The global context is a MEAN over points of an invariant per-point
    embedding.  It carries whole-cloud shape information into every local
    factor WITHOUT ever creating a global equivariant vector, so it cannot
    reintroduce the stabiliser constraint that collapsed the rank.
    """

    def __init__(self, channels, hidden=32, n_proj=8, gram='projected',
                 use_global=True, ctx_dim=16, use_force=False):
        super().__init__()
        if gram not in ('projected', 'full'):
            raise ValueError("gram must be 'projected' or 'full'")
        self.gram = gram
        self.use_global = use_global
        self.use_force = use_force
        if gram == 'projected':
            self.proj_u = LNLinear(channels, n_proj)
            self.proj_v = LNLinear(channels, n_proj)
            in_dim = n_proj * (2 if use_force else 1)
            out_dim = channels
        else:
            in_dim = channels * (2 if use_force else 1)
            out_dim = 1
        self.ctx_mlp = _mlp(in_dim, hidden, ctx_dim, depth=1) if use_global \
            else None
        self.gate_mlp = _mlp(in_dim + (ctx_dim if use_global else 0), hidden,
                             out_dim)

    def invariants(self, x):
        """x: [B, C, 6, N] -> S: [B, C or 1, N, in_dim] gate inputs."""
        if self.gram == 'projected':
            u, v = self.proj_u(x), self.proj_v(x)
            s = [bounded_invariant(klein_pair(u, v))]            # [B, P, N]
            if self.use_force:
                s.append(bounded_invariant(force_pair(u, v)))
            s = torch.cat(s, dim=1).permute(0, 2, 1)             # [B, N, in]
            return s.unsqueeze(1)                                # [B, 1, N, in]
        s = [bounded_invariant(klein_gram(x))]                   # [B, C, C, N]
        if self.use_force:
            f = x[:, :, 3:6]                              # [m; f] 저장
            s.append(bounded_invariant(
                torch.einsum('bcin,bdin->bcdn', f, f)))
        return torch.cat(s, dim=2).permute(0, 1, 3, 2)           # [B, C, N, in]

    def forward(self, x):
        s = self.invariants(x)
        if self.use_global:
            ctx = self.ctx_mlp(s).mean(dim=(1, 2), keepdim=True)  # [B,1,1,D]
            s = torch.cat([s, ctx.expand(*s.shape[:-1], ctx.shape[-1])], dim=-1)
        g = self.gate_mlp(s)                                     # [B,*,N,C|1]
        if self.gram == 'projected':
            g = g.squeeze(1).permute(0, 2, 1)                    # [B, C, N]
        else:
            g = g.squeeze(-1)                                    # [B, C, N]
        return (1.0 + torch.tanh(g)).unsqueeze(2) * x


# ------------------------------------------------------ (4) pointwise block
class PointwiseCovectorBlock(nn.Module):
    """(optional message passing) -> LN-Linear -> covector bracket -> gate.

        M_i     = sum_j gamma_ij X_j W_m
        Xt_i    = X_i + M_i
        Y_i     = Xt_i W + [Xt_i U, Xt_i V]_*
        X'_i    = g_i * Y_i

    All parameters are shared across points; only the data-dependent
    coefficients (gamma, g) vary from point to point.  Everything acts on the
    trailing point axis, so one block is a genuine pointwise map -- the repo's
    [B, C, 6, N] layout already treats that axis as the free one.
    """

    def __init__(self, in_channels, out_channels, use_bracket=True,
                 gate='projected', message_passing=False, msg_channels=8,
                 hidden=32, n_proj=8, n_rbf=8, use_global_context=True,
                 use_force_invariant=False, zero_init_bracket=True):
        super().__init__()
        self.mp = (InvariantMessagePassing(in_channels, msg_channels, hidden,
                                           max(2, n_proj // 2), n_rbf)
                   if message_passing else None)
        self.linear = LNLinear(in_channels, out_channels)
        if use_bracket:
            self.dir_u = nn.Linear(out_channels, out_channels, bias=False)
            self.dir_v = nn.Linear(out_channels, out_channels, bias=False)
            if zero_init_bracket:
                # Start the residual bracket branch at the identity: a random
                # quadratic map in front of an AIRM objective is unstable before
                # feature scales have calibrated.  One direction stays random so
                # the branch still receives gradient.
                nn.init.zeros_(self.dir_v.weight)
        else:
            self.dir_u = self.dir_v = None
        self.gate = (None if gate == 'none' else
                     KleinGate(out_channels, hidden, n_proj, gate,
                               use_global_context, use_force=use_force_invariant))

    def forward(self, x, graph):
        if self.mp is not None:
            x = x + self.mp(x, graph)
        x = self.linear(x)
        if self.dir_u is not None:
            u = self.dir_u(x.transpose(1, -1)).transpose(1, -1)
            v = self.dir_v(x.transpose(1, -1)).transpose(1, -1)
            x = x + covector_bracket(v, u)
        if self.gate is not None:
            x = self.gate(x)
        return x


# ------------------------------------------------- (5) late second moment head
class LateSecondMomentHead(nn.Module):
    """Z = X W_head,  K = (1/Z(P)) sum_{i,h} beta_ih z_ih z_ih^T.

    Returns (L, K) with L = [sqrt(beta_ih / Z) z_ih] in R^{6 x NH}, so that
    K = L L^T exactly.  L is the learned analysis coframe: for a twist xi,
    ||L^T xi||^2 = xi^T K xi is the virtual work stored by the latent
    measurements, and K is their Fisher information.

    normalize:
      nh    Z = N * H          -- density-normalised (matches the existing
                                  synthetic targets, which divide by N k)
      beta  Z = sum beta       -- invariant to how densely the surface is
                                  sampled, given density-matched weights
      one   Z = 1              -- stiffness grows with the amount of contact
    """

    def __init__(self, in_channels, factors=8, hidden=32, n_proj=8,
                 weight_mode='learned', normalize='nh', use_global=True,
                 use_force_invariant=False, learn_scale=True):
        super().__init__()
        if weight_mode not in ('learned', 'uniform'):
            raise ValueError("weight_mode must be 'learned' or 'uniform'")
        if normalize not in NORMALIZE_MODES:
            raise ValueError(f'normalize must be one of {NORMALIZE_MODES}')
        self.factors = factors
        self.weight_mode = weight_mode
        self.normalize = normalize
        self.linear = LNLinear(in_channels, factors)
        # A single global scalar K -> exp(g) K.  It is trivially invariant, so
        # it costs no structure, and it gives the AIRM objective a direct
        # handle on the overall magnitude: a pure scale mismatch s contributes
        # sqrt(6) |log s| to the loss and would otherwise have to be squeezed
        # through the beta MLP.
        self.log_scale = nn.Parameter(torch.zeros(())) if learn_scale else None
        if weight_mode == 'learned':
            self.proj_u = LNLinear(in_channels, n_proj)
            self.proj_v = LNLinear(in_channels, n_proj)
            self.use_force = use_force_invariant
            in_dim = n_proj * (2 if use_force_invariant else 1)
            self.ctx_mlp = _mlp(in_dim, hidden, 16, depth=1) if use_global \
                else None
            self.beta_mlp = _mlp(in_dim + (16 if use_global else 0), hidden,
                                 factors)
            # beta = softplus(.) starts at 1 and is independent of the random
            # upstream activations.
            nn.init.zeros_(self.beta_mlp[-1].weight)
            nn.init.constant_(
                self.beta_mlp[-1].bias,
                torch.log(torch.expm1(torch.tensor(1.0))).item())

    def weights(self, x):
        """x: [B, C, 6, N] -> beta: [B, H, N], positive and invariant."""
        if self.weight_mode == 'uniform':
            return x.new_ones(x.shape[0], self.factors, x.shape[-1])
        u, v = self.proj_u(x), self.proj_v(x)
        s = [bounded_invariant(klein_pair(u, v))]
        if self.use_force:
            s.append(bounded_invariant(force_pair(u, v)))
        s = torch.cat(s, dim=1).permute(0, 2, 1)                 # [B, N, in]
        if self.ctx_mlp is not None:
            ctx = self.ctx_mlp(s).mean(dim=1, keepdim=True)
            s = torch.cat([s, ctx.expand(-1, s.shape[1], -1)], dim=-1)
        return torch.nn.functional.softplus(self.beta_mlp(s)).permute(0, 2, 1)

    def forward(self, x):
        z = self.linear(x)                                       # [B, H, 6, N]
        beta = self.weights(x)                                   # [B, H, N]
        if self.normalize == 'nh':
            denom = float(z.shape[1] * z.shape[-1])
        elif self.normalize == 'beta':
            denom = beta.sum(dim=(1, 2)).clamp_min(_EPS).view(-1, 1, 1, 1)
        else:
            denom = 1.0
        L = z * (beta.unsqueeze(2) / denom).sqrt()               # [B, H, 6, N]
        if self.log_scale is not None:
            L = L * torch.exp(0.5 * self.log_scale)
        K = torch.einsum('bhin,bhjn->bij', L, L)
        K = 0.5 * (K + K.transpose(-1, -2))
        return L.permute(0, 2, 1, 3).reshape(z.shape[0], 6, -1), K


# ------------------------------------------------------------ (6) full model
class PointwiseStiffnessModel(nn.Module):
    """P -> tie-safe graph -> set pooling -> pointwise LN blocks -> K.

    channels[0] is the number of LINEAR set-pooling channels; the encoder's
    total output is channels[0] + bracket channels, and the blocks then walk
    through channels[1:].  The recommended schedule is C_0 = 8 -> 16 -> 32 ->
    16 -> H = 8.

    ``forward`` returns K [B, 6, 6]; ``factors`` returns the coframe L
    [B, 6, N H] with K = L L^T.  Graph diagnostics of the last forward pass are
    in ``self.last_graph_stats``: keep ``graph_truncation_frac`` at 0, and if
    it is not, ``graph_required_candidate_k`` is the value ``candidate_k`` has
    to reach (or a sign that the radius is too large -- see
    ``pointwise_graph``).

    The default radius mode is 'degree_matched', which fixes the mean degree
    at ``target_k`` for ANY point distribution.  The earlier default
    ('global_scale') lets the degree grow without bound in N, and
    'density_scaled' assumes an intrinsic dimension of 3 -- see
    ``pointwise_graph`` for the measurements.
    """

    def __init__(self, channels=(8, 16, 32, 16), factors=8, candidate_k=64,
                 radius_mode='degree_matched', radius_alpha=None,
                 radius_value=None, support_k=8, target_k=16, tie_eps=0.0,
                 n_rbf=8, pool='basis_mean', bracket='none',
                 bracket_channels=None, use_bracket_layers=True,
                 gate='projected', use_global_context=True,
                 message_passing=False, msg_channels=8, hidden=32, n_proj=8,
                 normalize='nh', beta_mode='learned',
                 use_force_invariant=False, learn_scale=True,
                 dist_compute_mode='donot_use_mm_for_euclid_dist'):
        super().__init__()
        if len(channels) < 2:
            raise ValueError('channels needs at least (C_0, C_1)')
        self.graph_kwargs = dict(candidate_k=candidate_k,
                                 radius_mode=radius_mode,
                                 radius_alpha=radius_alpha,
                                 radius_value=radius_value,
                                 support_k=support_k, target_k=target_k,
                                 tie_eps=tie_eps,
                                 dist_compute_mode=dist_compute_mode)
        self.set_encoder = LocalWrenchSetEncoder(
            out_channels=channels[0], hidden=hidden, n_rbf=n_rbf, pool=pool,
            bracket=bracket, bracket_channels=bracket_channels)
        dims = (self.set_encoder.out_channels,) + tuple(channels[1:])
        self.blocks = nn.ModuleList([
            PointwiseCovectorBlock(
                dims[i], dims[i + 1], use_bracket=use_bracket_layers,
                gate=gate, message_passing=message_passing,
                msg_channels=msg_channels, hidden=hidden, n_proj=n_proj,
                n_rbf=n_rbf, use_global_context=use_global_context,
                use_force_invariant=use_force_invariant)
            for i in range(len(dims) - 1)])
        self.head = LateSecondMomentHead(
            dims[-1], factors=factors, hidden=hidden, n_proj=n_proj,
            weight_mode=beta_mode, normalize=normalize,
            use_global=use_global_context,
            use_force_invariant=use_force_invariant, learn_scale=learn_scale)
        self.last_graph_stats = {}

    def build_graph(self, P):
        graph = build_local_graph(P, **self.graph_kwargs)
        self.last_graph_stats = graph.stats
        return graph

    def features(self, P, graph=None):
        """P: [B, N, 3] -> X^(L): [B, C_L, 6, N]."""
        graph = self.build_graph(P) if graph is None else graph
        x = self.set_encoder(P, graph)
        for blk in self.blocks:
            x = blk(x, graph)
        return x

    def encoder(self, P):
        """X^(0) only -- the signature metrics.f_signal expects."""
        return self.set_encoder(P, self.build_graph(P))

    def factors(self, P):
        """L: [B, 6, N H] with K = L L^T."""
        return self.head(self.features(P))[0]

    def forward(self, P):
        return self.head(self.features(P))[1]

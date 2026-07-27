"""Numerical verification of congruence equivariance (experiments A, B, C).

Run from the repo root:
    python experiment/pc_se3_congruence/verify.py

All computations in float64, random (untrained) weights, CPU.  Each experiment
block reseeds (0/1/2/3/4) so that adding a check to one block cannot shift the
RNG stream, and hence the reported numbers, of the others.
Results are printed and dumped to experiment/pc_se3_congruence/results.json.
"""
import json
import os
import sys
import time

sys.path.append('.')

import torch

# --- tripwire: assert that no experiment ever touches the se(3) Killing form.
# se(3) admits no Ad-invariant inner product, so Killing-based gating and
# normalization are ill-founded here (see check_killing_degeneracy.py).  The
# backbone uses LNLinear + LNLieBracket only; the Klein form appears solely as
# the constant head intertwiner Y = QZ.
import core.lie_alg_util as _lau
import core.lie_neurons_layers as _lnl

_KILLING_CALLS = []
_orig_killingform = _lau.killingform


def _counting_killingform(*a, **kw):
    _KILLING_CALLS.append(1)
    return _orig_killingform(*a, **kw)


_lau.killingform = _counting_killingform
_lnl.killingform = _counting_killingform

from experiment.pc_se3_congruence.se3_utils import (
    adjoint, adjoint_inv, compose, klein_Q, klein_Q_inv, random_SE3,
    scaled_err, transform_cloud)
from experiment.pc_se3_congruence.encoders import (
    PlueckerEncoder, LearnableLiftEncoder, knn_indices)
from experiment.pc_se3_congruence.models import (
    ModelA, ModelB, ModelC, ModelCNative, ModelCNaiveBracket,
    NaiveHeadNoKlein, add_bias_to_first_linear, covector_bracket,
    twist_bracket)

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

B, N, K_NN = 2, 64, 8
CHANNELS = (8, 16, 16, 8)
TRANS_SCALES = [0.0, 1.0, 1e2, 1e4]
N_TRIALS = 5

results = {}


def sweep(fn):
    """fn(R, p) -> scaled error; return {scale: max over trials}."""
    out = {}
    for s in TRANS_SCALES:
        errs = []
        for _ in range(N_TRIALS):
            R, p = random_SE3(trans_scale=s)
            errs.append(fn(R, p))
        out[f"{s:g}"] = max(errs)
    return out


def fmt(d):
    return "  ".join(f"p~{k}: {v:.2e}" for k, v in d.items())


# ---------------------------------------------------------------- Experiment 0
print("=" * 76)
print("Experiment 0: algebraic identities")
print("=" * 76)

Q = klein_Q()          # flat  : se(3)  -> se(3)*
Qinv = klein_Q_inv()   # sharp : se(3)* -> se(3)   (same matrix, different type)


def check_hom(R, p):
    R2, p2 = random_SE3(trans_scale=1.0)
    lhs = adjoint(*compose((R2, p2), (R, p)))
    rhs = adjoint(R2, p2) @ adjoint(R, p)
    return scaled_err(lhs, rhs)


def check_klein_inv(R, p):
    A = adjoint(R, p)
    return scaled_err(A.T @ Q @ A, Q)


def check_flat(R, p):
    """(I-3) lowering: Q Ad_T Q^{-1} = Ad_T^{-T}   (twist -> wrench)"""
    return scaled_err(Q @ adjoint(R, p) @ Qinv, adjoint_inv(R, p).T)


def check_sharp(R, p):
    """(I-4) raising: Q^{-1} Ad_T^{-T} Q = Ad_T    (wrench -> twist)"""
    return scaled_err(Qinv @ adjoint_inv(R, p).T @ Q, adjoint(R, p))


results['O1_Ad_homomorphism'] = sweep(check_hom)
results['O2_AdT_Q_Ad_eq_Q'] = sweep(check_klein_inv)
results['O3_flat_QAdQinv_eq_AdinvT'] = sweep(check_flat)
results['O4_sharp_QinvAdinvTQ_eq_Ad'] = sweep(check_sharp)
results['O5_Q_is_an_involution'] = scaled_err(Q @ Qinv, torch.eye(6))
for k in ['O1_Ad_homomorphism', 'O2_AdT_Q_Ad_eq_Q',
          'O3_flat_QAdQinv_eq_AdinvT', 'O4_sharp_QinvAdinvTQ_eq_Ad']:
    print(f"{k:28s} {fmt(results[k])}")
print(f"{'O5_Q_Qinv_eq_I':28s} {results['O5_Q_is_an_involution']:.2e}   "
      f"(Q^2 = I, so flat and sharp share a matrix but not a type)")

# ---------------------------------------------------------------- Experiment L
print("=" * 76)
print("Experiment L: lifting layer (point cloud -> se(3))")
print("=" * 76)
torch.manual_seed(1)   # per-block seed: keeps this block reproducible in isolation

P0 = torch.randn(B, N, 3) * 2.0

# L1: the two closed-form moment expressions agree exactly:
#     r_i x d_ij  ==  r_i x r_j   (docs vs. pdf notation)
idx = knn_indices(P0, K_NN)
nbr = torch.gather(P0.unsqueeze(2).expand(B, N, K_NN, 3), 1,
                   idx.unsqueeze(-1).expand(B, N, K_NN, 3))
d = nbr - P0.unsqueeze(2)
m_pdf = torch.cross(P0.unsqueeze(2).expand_as(d), d, dim=-1)     # r_i x d_ij
m_docs = torch.cross(P0.unsqueeze(2).expand_as(nbr), nbr, dim=-1)  # r_i x r_j
results['L1_two_moment_formulas'] = {'max_abs_rel': scaled_err(m_pdf, m_docs)}
print(f"L1 r_i x d_ij == r_i x r_j    : {results['L1_two_moment_formulas']['max_abs_rel']:.2e}")

enc_plk = PlueckerEncoder(k=K_NN)
enc_lrn = LearnableLiftEncoder(out_channels=CHANNELS[0], k=K_NN,
                               mode='anchor_transport')
enc_org = LearnableLiftEncoder(out_channels=CHANNELS[0], k=K_NN, mode='origin')
enc_org.load_state_dict(enc_lrn.state_dict())
enc_bad = LearnableLiftEncoder(out_channels=CHANNELS[0], k=K_NN,
                               mode='no_transport')
enc_bad.load_state_dict(enc_lrn.state_dict())

# L2: transported anchor lift == origin-referenced lift (Lemma 3.5)
with torch.no_grad():
    results['L2_transport_eq_origin'] = {
        'max_abs_rel': scaled_err(enc_lrn(P0), enc_org(P0))}
print(f"L2 anchor-transport == origin : {results['L2_transport_eq_origin']['max_abs_rel']:.2e}")


def enc_equiv(encoder):
    def fn(R, p):
        with torch.no_grad():
            V1 = encoder(transform_cloud(P0, R, p))
            V2 = torch.einsum('ij,bcjn->bcin', adjoint(R, p), encoder(P0))
        return scaled_err(V1, V2)
    return fn


results['L3_pluecker_encoder_equiv'] = sweep(enc_equiv(enc_plk))
results['L4_learnable_encoder_equiv'] = sweep(enc_equiv(enc_lrn))
results['L5_no_transport_NEGATIVE'] = sweep(enc_equiv(enc_bad))
print(f"L3 Pluecker encoder equiv     : {fmt(results['L3_pluecker_encoder_equiv'])}")
print(f"L4 learnable encoder equiv    : {fmt(results['L4_learnable_encoder_equiv'])}")
print(f"L5 no-transport (NEG)         : {fmt(results['L5_no_transport_NEGATIVE'])}")

# ---------------------------------------------------------------- Experiment A
print("=" * 76)
print("Experiment A: f(P) = C with C(T.P) = Ad_T C Ad_T^T")
print("=" * 76)
torch.manual_seed(2)   # per-block seed: keeps this block reproducible in isolation

model_a_plk = ModelA(PlueckerEncoder(k=K_NN), CHANNELS)
model_a_lrn = ModelA(LearnableLiftEncoder(out_channels=CHANNELS[0], k=K_NN),
                     CHANNELS)


def congA(model):
    def fn(R, p):
        with torch.no_grad():
            C1 = model(transform_cloud(P0, R, p))
            A = adjoint(R, p)
            C2 = A @ model(P0) @ A.T
        return scaled_err(C1, C2)
    return fn


results['A1_endtoend_pluecker'] = sweep(congA(model_a_plk))
results['A2_endtoend_learnable'] = sweep(congA(model_a_lrn))
print(f"A1 end-to-end (Pluecker enc)  : {fmt(results['A1_endtoend_pluecker'])}")
print(f"A2 end-to-end (learnable enc) : {fmt(results['A2_endtoend_learnable'])}")

# backbone-only square of the diagram
V0 = model_a_plk.encoder(P0)


def backbone_equiv(R, p):
    with torch.no_grad():
        A = adjoint(R, p)
        Z1 = model_a_plk.backbone(torch.einsum('ij,bcjn->bcin', A, V0))
        Z2 = torch.einsum('ij,bcjn->bcin', A, model_a_plk.backbone(V0))
    return scaled_err(Z1, Z2)


results['A3_backbone_equiv'] = sweep(backbone_equiv)
print(f"A3 backbone LNLin+LNBracket   : {fmt(results['A3_backbone_equiv'])}")

# negative control: bias in the first LNLinear
model_a_bias = ModelA(PlueckerEncoder(k=K_NN), CHANNELS)
model_a_bias.load_state_dict(model_a_plk.state_dict())
add_bias_to_first_linear(model_a_bias)
results['A4_bias_NEGATIVE'] = sweep(congA(model_a_bias))
print(f"A4 bias in LNLinear (NEG)     : {fmt(results['A4_bias_NEGATIVE'])}")

# ---------------------------------------------------------------- Experiment B
print("=" * 76)
print("Experiment B: f(P) = K with K(T.P) = Ad_T^{-T} K Ad_T^{-1}")
print("=" * 76)
torch.manual_seed(3)   # per-block seed: keeps this block reproducible in isolation

model_b_plk = ModelB(PlueckerEncoder(k=K_NN), CHANNELS)
model_b_lrn = ModelB(LearnableLiftEncoder(out_channels=CHANNELS[0], k=K_NN),
                     CHANNELS)


def congB(model, post=lambda K: K):
    def fn(R, p):
        with torch.no_grad():
            K1 = post(model(transform_cloud(P0, R, p)))
            Ai = adjoint_inv(R, p)
            K2 = Ai.T @ post(model(P0)) @ Ai
        return scaled_err(K1, K2)
    return fn


results['B1_endtoend_pluecker'] = sweep(congB(model_b_plk))
results['B2_endtoend_learnable'] = sweep(congB(model_b_lrn))
print(f"B1 end-to-end (Pluecker enc)  : {fmt(results['B1_endtoend_pluecker'])}")
print(f"B2 end-to-end (learnable enc) : {fmt(results['B2_endtoend_learnable'])}")

# PSD / rank sanity of K
with torch.no_grad():
    Kout = model_b_plk(P0)
    eig = torch.linalg.eigvalsh(Kout)
results['B3_K_spd'] = {'min_eig': eig.min().item(),
                       'rank6_all': bool((eig > 1e-12 * eig.max()).all())}
print(f"B3 K PSD check                : min eig {eig.min():.3e}, "
      f"full rank = {results['B3_K_spd']['rank6_all']}")

# negative control: drop the Klein intertwiner (K = Z Z^T)
model_b_naive = ModelB(PlueckerEncoder(k=K_NN), CHANNELS)
model_b_naive.load_state_dict(model_b_plk.state_dict())
model_b_naive.head = NaiveHeadNoKlein()
results['B4_no_klein_NEGATIVE'] = sweep(congB(model_b_naive))
print(f"B4 head without Klein (NEG)   : {fmt(results['B4_no_klein_NEGATIVE'])}")

# negative control: K + eps*I regularization
eps_reg = 1e-3
results['B5_epsI_NEGATIVE'] = sweep(
    congB(model_b_plk, post=lambda K: K + eps_reg * torch.eye(6)))
print(f"B5 K + eps*I (NEG)            : {fmt(results['B5_epsI_NEGATIVE'])}")

# ---------------------------------------------------------------- Experiment C
print("=" * 76)
print("Experiment C: covector input, equivariance and cascade composition")
print("=" * 76)
torch.manual_seed(4)   # per-block seed: keeps this block reproducible in isolation

model_c = ModelC(CHANNELS)
W0 = torch.randn(B, CHANNELS[0], 6, 1)


def congC(R, p):
    with torch.no_grad():
        Ai = adjoint_inv(R, p)
        W1 = torch.einsum('ij,bcjn->bcin', Ai.T, W0)          # Ad^{-T} W
        K1 = model_c(W1)
        K2 = Ai.T @ model_c(W0) @ Ai
    return scaled_err(K1, K2)


results['C1_onestep_equiv'] = sweep(congC)
print(f"C1 one-step equivariance      : {fmt(results['C1_onestep_equiv'])}")


def cascade(R, p):
    """T1 fixed at ||p||~1; the sweep variable transforms T2."""
    with torch.no_grad():
        T1 = random_SE3(trans_scale=1.0)
        T2 = (R, p)
        Ai1 = adjoint_inv(*T1)
        Ai2 = adjoint_inv(*T2)
        Ai21 = adjoint_inv(*compose(T2, T1))
        K1 = model_c(torch.einsum('ij,bcjn->bcin', Ai1.T, W0))
        K2 = model_c(torch.einsum('ij,bcjn->bcin', Ai21.T, W0))
        ref = Ai2.T @ K1 @ Ai2
    return scaled_err(K2, ref)


results['C2_cascade_T2T1'] = sweep(cascade)
print(f"C2 cascade K2 = Ad2^-T K1 Ad2^-1: {fmt(results['C2_cascade_T2T1'])}")

# consistency: dual rep is a homomorphism
def dual_hom(R, p):
    T1 = random_SE3(trans_scale=1.0)
    lhs = adjoint_inv(*compose((R, p), T1)).T
    rhs = adjoint_inv(R, p).T @ adjoint_inv(*T1).T
    return scaled_err(lhs, rhs)


results['C3_dual_homomorphism'] = sweep(dual_hom)
print(f"C3 Ad_(T2T1)^-T = Ad2^-T Ad1^-T: {fmt(results['C3_dual_homomorphism'])}")

# --- C4-C7: is there a nonlinearity native to se(3)*, and does the ordinary
#     Lie bracket work there without Q?
print("-" * 76)
print("C4-C7: nonlinearity on the covector space itself (no Q)")
print("-" * 76)

model_c_naive = ModelCNaiveBracket(CHANNELS)
model_c_native = ModelCNative(CHANNELS)


def congC_model(model):
    def fn(R, p):
        with torch.no_grad():
            Ai = adjoint_inv(R, p)
            K1 = model(torch.einsum('ij,bcjn->bcin', Ai.T, W0))
            K2 = Ai.T @ model(W0) @ Ai
        return scaled_err(K1, K2)
    return fn


results['C4_naive_twist_bracket_on_covectors_NEGATIVE'] = sweep(
    congC_model(model_c_naive))

# C4b: isolate the culprit -- LNLinear alone is fine on covectors, because
# channel mixing acts on the right and commutes with ANY left representation.
class _LinearOnly(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        from core.lie_neurons_layers import LNLinear
        self.net = torch.nn.Sequential(*[LNLinear(channels[i], channels[i + 1])
                                         for i in range(len(channels) - 1)])

    def forward(self, W):
        Y = self.net(W).squeeze(-1).transpose(1, 2)
        return Y @ Y.transpose(1, 2) / Y.shape[-1]


results['C4b_linear_only_on_covectors'] = sweep(congC_model(_LinearOnly(CHANNELS)))
print(f"C4b LNLinear only on covectors     : "
      f"{fmt(results['C4b_linear_only_on_covectors'])}")
results['C5_native_covector_bracket_Qfree'] = sweep(congC_model(model_c_native))
print(f"C4 twist bracket on covectors (NEG): "
      f"{fmt(results['C4_naive_twist_bracket_on_covectors_NEGATIVE'])}")
print(f"C5 native covector bracket, Q-free : "
      f"{fmt(results['C5_native_covector_bracket_Qfree'])}")

# C6: the native covector bracket IS the Q-transported Lie bracket
F1 = torch.randn(B, CHANNELS[0], 6, 4)
F2 = torch.randn(B, CHANNELS[0], 6, 4)
with torch.no_grad():
    lhs = covector_bracket(F1, F2)
    raise_ = lambda F: torch.einsum('ij,bcjn->bcin', Qinv, F)
    rhs = torch.einsum('ij,bcjn->bcin', Q, twist_bracket(raise_(F1), raise_(F2)))
results['C6_covector_bracket_eq_Q_sandwich'] = scaled_err(lhs, rhs)
print(f"C6 [.,.]_* == Q[Q^-1., Q^-1.]      : "
      f"{results['C6_covector_bracket_eq_Q_sandwich']:.2e}")

# C7: dimension of the space of equivariant bilinear maps se(3)* x se(3)* -> se(3)*
#     (216 unknowns: 6 output components x 36 bilinear coefficients each)
rows = []
for _ in range(120):
    R, p = random_SE3(trans_scale=1.0)
    rho = adjoint_inv(R, p).T                       # coadjoint representation
    a, b = torch.randn(6), torch.randn(6)
    ra, rb = rho @ a, rho @ b
    # constraint: B(rho a, rho b) - rho B(a, b) = 0, linear in the 216 coeffs
    M = torch.zeros(6, 6, 36)
    outer_r = torch.einsum('i,j->ij', ra, rb).reshape(36)
    outer_0 = torch.einsum('i,j->ij', a, b).reshape(36)
    for o in range(6):
        M[o, o] += outer_r
        M[o] -= rho[o, :, None] * outer_0[None, :]
    rows.append(M.reshape(6, 216))
A_sys = torch.cat(rows, dim=0)
sv = torch.linalg.svdvals(A_sys)
dim_null = int((sv < 1e-8 * sv.max()).sum()) + (216 - len(sv))
results['C7_equivariant_bilinear_dim_on_dual'] = dim_null
print(f"C7 dim of equivariant bilinear maps on se(3)*: {dim_null}   "
      f"(matches dim 2 on se(3): bracket and N-bracket, transported by Q)")

# ------------------------------------------------------------- tripwire check
print("=" * 76)
print("Killing-form tripwire")
print("=" * 76)
results['Z1_killingform_calls'] = len(_KILLING_CALLS)
print(f"killingform calls across all experiments : {len(_KILLING_CALLS)}  "
      f"({'OK' if not _KILLING_CALLS else 'UNEXPECTED'})")
assert not _KILLING_CALLS, "experiments must not use the se(3) Killing form"

# ------------------------------------------------------------------- save
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'results.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nresults written to {out_path}")

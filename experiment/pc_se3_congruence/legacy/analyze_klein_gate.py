"""[SUPERSEDED gen-1 study]
Vector/covector coupling and Klein-form nonlinearity experiments.

This script answers two questions left open by the structural equivariance
checks in ``verify.py``:

1. Does staying in the covector representation remove the triangular
   translation-slot -> orientation-slot blockage of an se(3) bracket network?
2. Can the Klein form replace the degenerate Killing form in an LN-ReLU?

The answer to (1) is no: the coadjoint representation and its transported
bracket have the same triangular structure with the physical slot names
swapped.  The answer to (2) is nuanced.  A Klein-pairing *gate* is exactly
equivariant and restores cross-slot dependence, but the Klein form has
signature (3,3), so it is not a norm.  A VN-style normalized projection is
singular on the Klein null cone, which contains every Pluecker line feature.

Run from the repository root:

    conda run -n lieneurons python \
        experiment/pc_se3_congruence/legacy/analyze_klein_gate.py

Results are written to ``klein_gate_results.json`` next to this file.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.append(".")

import torch
import torch.nn as nn

from core.lie_neurons_layers import LNLinear, LNLinearAndLieBracket
from experiment.pc_se3_congruence.encoders import (
    PlueckerEncoder,
    WrenchPlueckerEncoder,
)
from experiment.pc_se3_congruence.models import (
    CovectorBackbone,
    ModelB,
    ModelPC2K,
)
from experiment.pc_se3_congruence.se3_utils import (
    adjoint,
    adjoint_inv,
    coadjoint,
    klein_Q,
    random_SE3,
    scaled_err,
    transform_cloud,
)


torch.set_default_dtype(torch.float64)
torch.manual_seed(17)

TRANS_SCALES = (0.0, 1.0, 1e2, 1e4)
N_TRIALS = 5
EPS = 1e-12


def klein_pair(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Channelwise Klein pairing for repo-order twists [v; w].

    The same coordinate formula applies to repo-order wrenches [f; m].
    Inputs have shape [B, C, 6, N], output [B, C, 1, N].
    """
    return ((x[:, :, 0:3] * d[:, :, 3:6]).sum(dim=2, keepdim=True)
            + (x[:, :, 3:6] * d[:, :, 0:3]).sum(dim=2, keepdim=True))


def killing_pair(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Normalized se(3) Killing form: it only sees angular slots."""
    return (x[:, :, 3:6] * d[:, :, 3:6]).sum(dim=2, keepdim=True)


def dual_killing_pair(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Transported degenerate form on wrenches [force; moment]."""
    return (x[:, :, 0:3] * d[:, :, 0:3]).sum(dim=2, keepdim=True)


def euclidean_pair(x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    return (x * d).sum(dim=2, keepdim=True)


def channel_mix(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """d_o = sum_i weight[o,i] x_i, leaving representation axes untouched."""
    return torch.einsum("oi,biqn->boqn", weight, x)


class PairingGate(nn.Module):
    """Bounded residual scalar gate x * (1 + tanh(B(x, Ux))).

    This is the safe use of an invariant indefinite form: B is used only to
    make a scalar, never as a norm or divisor.  ``form='klein'`` is equivariant
    under both Ad (twists) and Ad^{-T} (wrenches).
    """

    def __init__(self, channels: int, form: str = "klein"):
        super().__init__()
        self.learn_dir = nn.Linear(channels, channels, bias=False)
        self.form = form

    def score(self, x: torch.Tensor) -> torch.Tensor:
        d = self.learn_dir(x.transpose(1, -1)).transpose(1, -1)
        if self.form == "klein":
            return klein_pair(x, d)
        if self.form == "killing":
            return killing_pair(x, d)
        if self.form == "dual_killing":
            return dual_killing_pair(x, d)
        if self.form == "euclidean":
            return euclidean_pair(x, d)
        raise ValueError(self.form)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + torch.tanh(self.score(x)))


class DirectKleinRelu(nn.Module):
    """Literal sign/shear analogue of the existing LN-ReLU with B=Klein.

    It remains equivariant because B(x,d) is invariant.  It is deliberately
    unnormalized: dividing by B(d,d) would be singular on null directions.
    Without a positive metric this operation is a shear selected by a sign,
    not an orthogonal ReLU projection.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.learn_dir = nn.Linear(channels, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = self.learn_dir(x.transpose(1, -1)).transpose(1, -1)
        s = klein_pair(x, d)
        return torch.where(s <= 0, x, x - s * d)


class BracketRegressor(nn.Module):
    """Bracket-only comparator for the synthetic cross-slot target."""

    def __init__(self, channels: int = 2, hidden: int = 12):
        super().__init__()
        self.blocks = nn.ModuleList([
            LNLinearAndLieBracket(channels, hidden, algebra_type="se3"),
            LNLinearAndLieBracket(hidden, hidden, algebra_type="se3"),
        ])
        self.out = LNLinear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.out(x)


def transform_features(x: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,bcjn->bcin", rho, x)


def sweep_equivariance(module: nn.Module, representation: str) -> dict[str, float]:
    x = torch.randn(3, 4, 6, 2)
    y0 = module(x)
    out = {}
    for scale in TRANS_SCALES:
        errors = []
        for _ in range(N_TRIALS):
            R, p = random_SE3(trans_scale=scale)
            rho = adjoint(R, p) if representation == "vector" else coadjoint(R, p)
            lhs = module(transform_features(x, rho))
            rhs = transform_features(y0, rho)
            errors.append(scaled_err(lhs, rhs))
        out[f"{scale:g}"] = max(errors)
    return out


def relative_change(a: torch.Tensor, b: torch.Tensor) -> float:
    return scaled_err(a, b)


def fixed_cross_weight(channels: int) -> torch.Tensor:
    """Cyclic channel permutation: d_c=x_{c+1}, avoiding self-pairing."""
    w = torch.zeros(channels, channels)
    for c in range(channels):
        w[c, (c + 1) % channels] = 1.0
    return w


def fit_synthetic(model: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor,
                  x_test: torch.Tensor, y_test: torch.Tensor,
                  steps: int = 1800) -> dict[str, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(123)
    batch = 256
    for _ in range(steps):
        idx = torch.randint(x_train.shape[0], (batch,), generator=gen)
        pred = model(x_train[idx])
        loss = (pred - y_train[idx]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
    with torch.no_grad():
        pred = model(x_test)
        mse = (pred - y_test).square().mean().item()
        angular_mse = (pred[:, :, 3:6] - y_test[:, :, 3:6]).square().mean().item()
        target_energy = y_test.square().mean().item()
    return {
        "test_mse": mse,
        "test_nmse": mse / target_energy,
        "angular_mse": angular_mse,
    }


def main() -> None:
    results: dict[str, object] = {}

    # ---------------------------------------------------------------- forms
    Q = klein_Q()
    Qk = torch.zeros(6, 6)
    Qk[3:6, 3:6] = torch.eye(3)
    results["form_geometry"] = {
        "killing_rank": int(torch.linalg.matrix_rank(Qk)),
        "klein_rank": int(torch.linalg.matrix_rank(Q)),
        "killing_eigenvalues": torch.linalg.eigvalsh(Qk).tolist(),
        "klein_eigenvalues": torch.linalg.eigvalsh(Q).tolist(),
    }

    # ---------------------------------------- vector/covector path identity
    # The wrench lift is Q times the twist lift.  With transported bracket
    # weights, an all-covector model is therefore a coordinate-conjugate copy
    # of the vector-backbone + final-Q model, not a larger function class.
    torch.manual_seed(171)
    P = torch.randn(2, 64, 3)
    channels_path = (8, 16, 16, 8)
    vector_model = ModelB(PlueckerEncoder(k=8), channels_path)
    covector_model = ModelPC2K(WrenchPlueckerEncoder(k=8), channels_path)
    with torch.no_grad():
        for vector_block, covector_block in zip(
                vector_model.backbone.blocks, covector_model.backbone.blocks):
            covector_block.linear.load_state_dict(vector_block.linear.state_dict())
            covector_block.bracket.learn_dir.weight.copy_(
                vector_block.liebracket.learn_dir.weight)
            covector_block.bracket.learn_dir2.weight.copy_(
                vector_block.liebracket.learn_dir2.weight)

        twist_lift = vector_model.encoder(P)
        wrench_lift = covector_model.encoder(P)
        q_twist_lift = transform_features(twist_lift, Q)
        vector_K = vector_model(P)
        _, covector_K = covector_model(P)

    def pc_equivariance(model: nn.Module, returns_pair: bool) -> dict[str, float]:
        output = model(P)
        K0 = output[1] if returns_pair else output
        values = {}
        for scale in TRANS_SCALES:
            errors = []
            for _ in range(N_TRIALS):
                R, p = random_SE3(trans_scale=scale)
                transformed = model(transform_cloud(P, R, p))
                K1 = transformed[1] if returns_pair else transformed
                Ai = adjoint_inv(R, p)
                errors.append(scaled_err(K1, Ai.T @ K0 @ Ai))
            values[f"{scale:g}"] = max(errors)
        return values

    results["vector_covector_path_equivalence"] = {
        "wrench_lift_vs_Q_twist_lift": scaled_err(wrench_lift, q_twist_lift),
        "shared_weight_K_difference": scaled_err(vector_K, covector_K),
        "vector_path_equivariance": pc_equivariance(vector_model, False),
        "covector_path_equivariance": pc_equivariance(covector_model, True),
    }

    # -------------------------------------------------- triangular blockage
    torch.manual_seed(18)
    x = torch.randn(2, 4, 6, 1)
    x_v_changed = x.clone()
    x_v_changed[:, :, 0:3] += 3.0 * torch.randn_like(x_v_changed[:, :, 0:3])
    twist_net = nn.Sequential(
        LNLinearAndLieBracket(4, 8, algebra_type="se3"),
        LNLinearAndLieBracket(8, 4, algebra_type="se3"),
    )
    with torch.no_grad():
        tw0, tw1 = twist_net(x), twist_net(x_v_changed)

    f = torch.randn(2, 4, 6, 1)  # wrench storage [force; moment]
    f_m_changed = f.clone()
    f_m_changed[:, :, 3:6] += 3.0 * torch.randn_like(f_m_changed[:, :, 3:6])
    cov_net = CovectorBackbone((4, 8, 4))
    with torch.no_grad():
        co0, co1 = cov_net(f), cov_net(f_m_changed)

    gate = PairingGate(4, "klein")
    with torch.no_grad():
        gate.learn_dir.weight.copy_(fixed_cross_weight(4))
        gt0, gt1 = gate(x), gate(x_v_changed)
        gc0, gc1 = gate(f), gate(f_m_changed)

    results["cross_slot_dependency"] = {
        "twist_bracket_angular_change_after_v_change": relative_change(
            tw0[:, :, 3:6], tw1[:, :, 3:6]),
        "covector_bracket_force_change_after_moment_change": relative_change(
            co0[:, :, 0:3], co1[:, :, 0:3]),
        "klein_gate_twist_angular_change_after_v_change": relative_change(
            gt0[:, :, 3:6], gt1[:, :, 3:6]),
        "klein_gate_covector_force_change_after_moment_change": relative_change(
            gc0[:, :, 0:3], gc1[:, :, 0:3]),
    }

    # ----------------------------------------- Killing vs Klein sensitivity
    w_cross = fixed_cross_weight(4)
    d0 = channel_mix(x, w_cross)
    d1 = channel_mix(x_v_changed, w_cross)
    k0, k1 = killing_pair(x, d0), killing_pair(x_v_changed, d1)
    q0, q1 = klein_pair(x, d0), klein_pair(x_v_changed, d1)

    pure_t = torch.zeros_like(x)
    pure_t[:, :, 0:3] = torch.randn_like(pure_t[:, :, 0:3])
    pure_t_d = channel_mix(pure_t, w_cross)

    # Test raw (unpooled) line coordinates.  Each [r x n; n] is exactly on
    # the Klein null cone.  A pooled sum of lines need not itself be a line.
    r = torch.randn(4, 8, 3, 1)
    n = torch.randn(4, 8, 3, 1)
    raw_lines = torch.cat([torch.cross(r, n, dim=2), n], dim=2)
    self_scores = klein_pair(raw_lines, raw_lines)
    cross_scores = klein_pair(
        raw_lines, channel_mix(raw_lines, fixed_cross_weight(8)))
    results["pairing_sensitivity"] = {
        "killing_score_change_after_v_change": relative_change(k0, k1),
        "klein_score_change_after_v_change": relative_change(q0, q1),
        "pure_translation_klein_score_max_abs": klein_pair(
            pure_t, pure_t_d).abs().max().item(),
        "raw_pluecker_self_klein_max_abs": self_scores.abs().max().item(),
        "raw_pluecker_cross_klein_mean_abs": cross_scores.abs().mean().item(),
        "raw_pluecker_cross_nonzero_fraction": (
            cross_scores.abs() > 1e-12).double().mean().item(),
    }

    # ---------------------------------------------------------- equivariance
    modules = {
        "klein_bounded_gate": (PairingGate(4, "klein"), ("vector", "covector")),
        "klein_direct_relu": (DirectKleinRelu(4), ("vector", "covector")),
        "twist_killing_bounded_gate": (PairingGate(4, "killing"), ("vector",)),
        "covector_dual_killing_bounded_gate": (
            PairingGate(4, "dual_killing"), ("covector",)),
        "euclidean_bounded_gate_NEGATIVE": (
            PairingGate(4, "euclidean"), ("vector", "covector")),
    }
    torch.manual_seed(19)
    for module, _ in modules.values():
        nn.init.normal_(module.learn_dir.weight, std=0.3)
    results["equivariance"] = {}
    for name, (module, representations) in modules.items():
        results["equivariance"][name] = {
            representation: sweep_equivariance(module, representation)
            for representation in representations
        }

    # ------------------------------------- null cone / scale stability check
    # Two different Pluecker lines are individually null but generally have
    # nonzero mutual pairing.  Therefore B(x,d)/B(d,d) divides by exact zero.
    x_line = raw_lines[:, 0:1]
    d_line = raw_lines[:, 1:2]
    numerator = klein_pair(x_line, d_line)
    denominator = klein_pair(d_line, d_line)
    projected_eps = x_line - numerator / (denominator + EPS) * d_line

    direct = DirectKleinRelu(4)
    bounded = PairingGate(4, "klein")
    with torch.no_grad():
        direct.learn_dir.weight.copy_(fixed_cross_weight(4))
        bounded.learn_dir.weight.copy_(fixed_cross_weight(4))
    amplification = {}
    base = torch.randn(64, 4, 6, 1)
    with torch.no_grad():
        for scale in (1.0, 10.0, 100.0):
            xs = scale * base
            amplification[f"{scale:g}"] = {
                "direct_relu": (direct(xs).norm() / xs.norm()).item(),
                "bounded_gate": (bounded(xs).norm() / xs.norm()).item(),
            }
    results["stability"] = {
        "null_denominator_max_abs": denominator.abs().max().item(),
        "cross_numerator_mean_abs": numerator.abs().mean().item(),
        "eps_projection_amplification": (projected_eps.norm() / x_line.norm()).item(),
        "input_scale_amplification": amplification,
    }

    # --------------------------------------- synthetic expressivity/training
    # The target is itself a Klein gate.  Its angular output depends on v via
    # an invariant scalar, which bracket-only and Killing-only models cannot
    # represent because their angular paths see omega only.
    torch.manual_seed(20)
    n_train, n_test, channels = 4096, 1024, 2
    x_train = torch.randn(n_train, channels, 6, 1)
    x_test = torch.randn(n_test, channels, 6, 1)
    true_w = fixed_cross_weight(channels)

    def target_fn(z: torch.Tensor) -> torch.Tensor:
        score = klein_pair(z, channel_mix(z, true_w))
        return z * (1.0 + torch.tanh(score))

    y_train, y_test = target_fn(x_train), target_fn(x_test)
    train_results = {}
    for name, model in {
        "klein_gate": PairingGate(channels, "klein"),
        "killing_gate": PairingGate(channels, "killing"),
        "bracket_only": BracketRegressor(channels),
    }.items():
        torch.manual_seed(21)
        for parameter in model.parameters():
            if parameter.ndim >= 2:
                nn.init.normal_(parameter, std=0.2)
        train_results[name] = fit_synthetic(
            model, x_train, y_train, x_test, y_test)

    oracle = PairingGate(channels, "klein")
    with torch.no_grad():
        oracle.learn_dir.weight.copy_(true_w)
        oracle_err = (oracle(x_test) - y_test).abs().max().item()
    train_results["klein_oracle_max_abs"] = oracle_err
    results["synthetic_cross_slot_fit"] = train_results

    path = os.path.join(os.path.dirname(__file__), "klein_gate_results.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nresults written to {os.path.abspath(path)}")


if __name__ == "__main__":
    main()

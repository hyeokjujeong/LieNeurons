"""[SHARED UTIL]
Affine-invariant (AIRM) geodesic loss on SPD matrices, gradient-safe.

The minimal geodesic distance on S^{++}_n with the affine-invariant metric
g_A(X, Y) = tr(A^{-1} X A^{-1} Y) is

    d(A, B) = || log(A^{-1/2} B A^{-1/2}) ||_F
            = ( sum_i log^2 lambda_i )^{1/2},

where lambda_i are the (generalized) eigenvalues of the pencil (B, A),
i.e. the eigenvalues of A^{-1/2} B A^{-1/2}.

Why this metric for the K-regression task: it is invariant under ANY joint
congruence, d(W A W^T, W B W^T) = d(A, B) for all invertible W.  Since the
SE(3) action on K is exactly a congruence by W = Ad_T^{-T}, the loss is
SE(3)-invariant: together with the equivariant network, the training loss of
a sample does not depend on the frame in which it is expressed.

Gradient safety: the loss uses ONLY eigenvalues (a smooth spectral function),
never eigenvectors.  The backward of `eigvalsh` for an eigenvalue-only output
is  V diag(dL/dlambda) V^T,  which contains NO 1/(lambda_i - lambda_j)
divided differences -- it stays finite and correct even for exactly repeated
eigenvalues.  (A naive matrix log built as V log(Lambda) V^T and backprop'd
through `eigh` does NOT have this property: its backward has 0/0 terms at
degenerate eigenvalues and produces NaN.)

Closed-form gradient (for reference / sanity checks):
    with M = S B S, S = A^{-1/2}:
        d/dB  d^2(A, B) = 2 * S (M^{-1} log M) S,
which diverges like 2 log(lam)/lam as an eigenvalue lam -> 0.  The AIRM
distance is a natural barrier of the SPD cone; use `eig_floor` to cap the
gradient magnitude early in training if predictions start near-singular.

WHICH FUNCTION TO USE.  ``affine_invariant_d`` is THE loss every current
training run optimises (``blockage_bench.py``, ``peghole_baseline.py``); it
whitens with the Cholesky factor of the target and reports how many eigenvalues
hit the guard.  It used to live in the now-superseded ``legacy/train.py``, which
made the current entry points depend on legacy code; it lives here instead.
The ``airm_dist`` / ``airm_dist2`` pair below is the same distance written with
an explicit inverse square root -- more general (it accepts a precomputed
A^{-1/2}) but slower, and kept as the reference implementation.
``log_euclidean_dist2`` is a comparison metric only: it is NOT SE(3)-invariant.
"""
import torch

EIG_CLAMP = 1e-12


def affine_invariant_d(chol_gt, K_pred):
    """d = ||log(K_gt^{-1/2} K_pred K_gt^{-1/2})||_F per sample.

    chol_gt: [B, 6, 6] lower Cholesky of K_gt (no grad), K_pred: [B, 6, 6].
    Returns (d [B], n_clamped) — n_clamped counts eigenvalues at the guard
    (invariant quantity, so clamping does not break loss invariance).
    """
    X = torch.linalg.solve_triangular(chol_gt, K_pred, upper=False)
    A = torch.linalg.solve_triangular(chol_gt, X.transpose(-1, -2), upper=False)
    A = 0.5 * (A + A.transpose(-1, -2))
    lam = torch.linalg.eigvalsh(A)
    n_clamped = int((lam <= EIG_CLAMP).sum().item())
    d2 = torch.log(lam.clamp_min(EIG_CLAMP)).square().sum(-1)
    return torch.sqrt(d2), n_clamped


def sqrtm_spd(A):
    """Symmetric square root of an SPD matrix (via eigh)."""
    lam, V = torch.linalg.eigh(A)
    return (V * lam.clamp_min(0.0).sqrt().unsqueeze(-2)) @ V.transpose(-1, -2)


def inv_sqrtm_spd(A, eps=1e-12):
    """Symmetric inverse square root of an SPD matrix (via eigh)."""
    lam, V = torch.linalg.eigh(A)
    return (V * lam.clamp_min(eps).rsqrt().unsqueeze(-2)) @ V.transpose(-1, -2)


def airm_dist2(B, A=None, A_inv_sqrt=None, eig_floor=None):
    """Squared AIRM geodesic distance d^2(A, B), differentiable in B.

    B : [..., n, n] prediction (SPD, requires grad)
    A : [..., n, n] target (SPD, treated as constant)
    A_inv_sqrt : optionally precomputed A^{-1/2} (e.g. once per dataset)
    eig_floor : if given, eigenvalues of A^{-1/2} B A^{-1/2} are clamped from
        below before the log -- caps the barrier gradient 2 log(l)/l when the
        prediction is nearly singular.  Clamping acts on congruence-invariant
        quantities, so the SE(3)-invariance of the loss is preserved.
    """
    if A_inv_sqrt is None:
        with torch.no_grad():
            A_inv_sqrt = inv_sqrtm_spd(A)
    M = A_inv_sqrt @ B @ A_inv_sqrt
    M = 0.5 * (M + M.transpose(-1, -2))          # 수치 대칭화
    lam = torch.linalg.eigvalsh(M)               # eigenvalue-only: 안정한 backward
    if eig_floor is not None:
        lam = lam.clamp_min(eig_floor)
    return torch.log(lam).square().sum(-1)


def airm_dist(B, A=None, A_inv_sqrt=None, eig_floor=None):
    """AIRM geodesic distance (square root of airm_dist2)."""
    return airm_dist2(B, A, A_inv_sqrt, eig_floor).clamp_min(1e-30).sqrt()


def log_euclidean_dist2(B, A):
    """|| log B - log A ||_F^2.  Cheaper, but only invariant under ORTHOGONAL
    congruence -- NOT under general Ad_T^{-T} (p != 0), so unlike AIRM it is
    not SE(3)-invariant for this task.  Kept for comparison experiments.
    NOTE: uses eigh-based matrix log; backward has divided-difference terms
    that degrade near repeated eigenvalues."""
    def logm(X):
        lam, V = torch.linalg.eigh(X)
        return (V * lam.clamp_min(1e-30).log().unsqueeze(-2)) @ V.transpose(-1, -2)
    D = logm(B) - logm(A)
    return (D * D).sum((-2, -1))

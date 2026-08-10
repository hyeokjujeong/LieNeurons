"""[SHARED UTIL] se(3) utilities for the congruence-equivariance experiments.

STORAGE CONVENTION -- ANGULAR / MOMENT FIRST.  The whole repository uses

    twist   xi = [omega ; v]   x[..., 0:3] = omega (angular)
                               x[..., 3:6] = v     (linear)
    wrench  F  = [m ; f]       F[..., 0:3] = m     (moment)
                               F[..., 3:6] = f     (force)

matching core.lie_alg_util.HatLayer('se3') / vee_se3 and the report documents.
In this ordering the adjoint of T = (R, p) acting on twists is

    Ad_T = [  R    0 ]                  Ad_T^{-T} = [ R  p^ R ]
           [ p^ R  R ],                             [ 0   R   ]

with the coupling block LOWER LEFT for twists and UPPER RIGHT for wrenches, so
omega (twist slot 0:3) and f (wrench slot 3:6) are the translation-blind ones.
The Klein form Gram matrix is Q = [[0, I], [I, 0]] (so that
xi1^T Q xi2 = omega1 . v2 + omega2 . v1) and Q^{-1} = Q.

Consequently the 6x6 stiffness K = L L^T built from wrench factors has

    K[0:3, 0:3] = mm  (rotational)      K[3:6, 3:6] = ff  (translational).

HISTORY.  This module (and the rest of the pipeline) previously stored
LINEAR-FIRST, xi = [v; omega], F = [f; m], with the coupling blocks mirrored.
The two orders are conjugate by the block-swap permutation Pi = [[0, I], [I, 0]]:
Ad^[omega;v] = Pi Ad^[v;omega] Pi.  Q is numerically the same matrix in both
(it IS the swap), so the head Y = Q Z needed no adjustment.  Relative errors,
ranks, dimensions, signatures and eigenvalues are all invariant under
Pi-conjugation, so no reported NUMBER changed with the switch -- only which
block is called ff and which mm.  Datasets written before the switch are
converted on load; see data_loader/peg_hole_data_loader.py.
"""
import torch


def hat3(a):
    """a: [..., 3] -> [..., 3, 3] cross-product matrix."""
    zero = torch.zeros_like(a[..., 0])
    return torch.stack([
        torch.stack([zero, -a[..., 2], a[..., 1]], dim=-1),
        torch.stack([a[..., 2], zero, -a[..., 0]], dim=-1),
        torch.stack([-a[..., 1], a[..., 0], zero], dim=-1),
    ], dim=-2)


def random_SO3(gen=None, dtype=torch.float64):
    A = torch.randn(3, 3, generator=gen, dtype=dtype)
    Q_, _ = torch.linalg.qr(A)
    if torch.det(Q_) < 0:
        Q_[:, 0] = -Q_[:, 0]
    return Q_


def random_SE3(trans_scale=1.0, gen=None, dtype=torch.float64):
    R = random_SO3(gen, dtype)
    p = torch.randn(3, generator=gen, dtype=dtype) * trans_scale
    return R, p


def compose(T1, T2):
    """(R1,p1) * (R2,p2) = (R1 R2, R1 p2 + p1)."""
    R1, p1 = T1
    R2, p2 = T2
    return R1 @ R2, R1 @ p2 + p1


def adjoint(R, p):
    """6x6 adjoint on twists stored [omega; v]:

        omega -> R omega,    v -> R v + p x R omega,

    i.e. [[R, 0], [p^ R, R]] — the coupling sits LOWER LEFT, so the angular
    slot is the translation-blind one."""
    A = torch.zeros(6, 6, dtype=R.dtype, device=R.device)
    A[0:3, 0:3] = R
    A[3:6, 3:6] = R
    A[3:6, 0:3] = hat3(p) @ R
    return A


def adjoint_inv(R, p):
    """Closed form Ad_T^{-1} = Ad_{T^{-1}}, avoiding an LU solve whose error
    grows with cond(Ad_T) ~ (1 + ||p||)^2."""
    return adjoint(R.transpose(-1, -2), -R.transpose(-1, -2) @ p)


def coadjoint(R, p):
    """6x6 coadjoint Ad_T^{-T} acting on wrenches stored [m; f]:

        m -> R m + p x R f,        f -> R f,

    i.e. the block matrix [[R, p^ R], [0, R]] — the mirror image of adjoint():
    for twists the angular slot is translation-blind, for wrenches the force
    slot is.  Numerically adjoint_inv(R, p)^T (closed form, no solve)."""
    return adjoint_inv(R, p).transpose(-1, -2)


def klein_Q(dtype=torch.float64, device=None):
    """Gram matrix of the Klein form, i.e. the musical FLAT map

        Q : se(3) -> se(3)*,    twist |-> wrench,

    since a bilinear form on V is by definition a map V -> V*.  It intertwines
    the two representations in this direction:  Q Ad_T = Ad_T^{-T} Q.
    Use this when lowering a twist to a wrench (head B: Y = Q Z).
    """
    Q = torch.zeros(6, 6, dtype=dtype, device=device)
    Q[0:3, 3:6] = torch.eye(3, dtype=dtype, device=device)
    Q[3:6, 0:3] = torch.eye(3, dtype=dtype, device=device)
    return Q


def klein_Q_inv(dtype=torch.float64, device=None):
    """The musical SHARP map, inverse of klein_Q:

        Q^{-1} : se(3)* -> se(3),   wrench |-> twist,

    intertwining the other way:  Q^{-1} Ad_T^{-T} = Ad_T Q^{-1}.
    Use this when raising a wrench to a twist (model C input stage).

    Q is the block swap, so Q^2 = I and Q^{-1} is NUMERICALLY the same matrix.
    It is kept as a separate function because the two are different maps
    between different spaces, and experiment B4 shows what confusing the
    twist/wrench type costs (O(1) equivariance failure once p != 0).
    """
    return klein_Q(dtype=dtype, device=device)


def transform_cloud(P, R, p):
    """P: [B, N, 3] -> R P + p."""
    return P @ R.transpose(-1, -2) + p


def scaled_err(A, B, delta=1e-300):
    """Scale-free relative error ||A - B|| / max(||A||, ||B||)."""
    na, nb = torch.linalg.norm(A.flatten()), torch.linalg.norm(B.flatten())
    return (torch.linalg.norm((A - B).flatten()) / (torch.maximum(na, nb) + delta)).item()

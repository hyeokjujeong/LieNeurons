"""Why the se(3) Killing form must not be used for gating or normalization.

Backs the claims of section 5.1 of docs/pc_se3_congruence_report.md:

  (1) The Killing form of se(3) has rank 3; its radical is the translation
      ideal t = {(0, v)}.
  (2) Every Ad-invariant symmetric bilinear form on se(3) is
      xi^T M_{a,b} xi = a||w||^2 + 2b (w . v), which is DEGENERATE for b = 0
      and INDEFINITE of signature (3,3) for b != 0.  So se(3) admits no
      Ad-invariant inner product, and the VN-style "fold across a learned
      hyperplane" nonlinearity has no metric to fold against.
  (3) Consequently LNKillingRelu is the exact identity on t and its gate is
      completely blind to the v slot.
  (4) And the Killing normalizer 1/||w||^2 used by LNBatchNorm / LNMaxPool
      blows up on bracket outputs w1 x w2 as the two directions align.

Run from the repo root:
    python experiment/pc_se3_congruence/check_killing_degeneracy.py
"""
import sys

sys.path.append('.')

import torch

from core.lie_alg_util import HatLayer, killingform_se3
from core.lie_neurons_layers import LNKillingRelu

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

hat = HatLayer('se3')


def banner(msg):
    print("=" * 72)
    print(msg)
    print("=" * 72)


# ---------------------------------------------- (1) rank and radical
banner("(1) Killing form of se(3): rank and radical")
print("(Gram built from HatLayer('se3'), i.e. in the repo's [v; omega] storage;")
print(" rank and radical dimension do not depend on the slot ordering)")
E = torch.eye(6)
G = torch.stack([torch.stack([killingform_se3(hat(E[i]), hat(E[j])).squeeze()
                              for j in range(6)]) for i in range(6)])
rank = torch.linalg.matrix_rank(G).item()
print(f"Gram matrix rank                      : {rank} / 6")
print(f"radical dimension                     : {6 - rank}   (= t = {{(0, v)}})")

# ------------------------------- (2) no invariant inner product exists
banner("(2) every Ad-invariant symmetric form is degenerate or indefinite")
print("M_{a,b} = [[a I, b I], [b I, 0]] in ANGULAR-FIRST [omega; v] blocks")
print("xi^T M_{a,b} xi = a||w||^2 + 2b (w . v)")
print("(signature, rank and det are invariant under the block swap that takes")
print(" this to the repo's [v; omega] storage order, so both readings agree)\n")
for a, b in [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, -0.5), (-2.0, 0.3)]:
    M = torch.zeros(6, 6)
    M[0:3, 0:3] = a * torch.eye(3)                      # a * (w1 . w2)
    M[0:3, 3:6] = b * torch.eye(3)                      # b * (w1 . v2 + w2 . v1)
    M[3:6, 0:3] = b * torch.eye(3)
    ev = torch.linalg.eigvalsh(M)
    npos = int((ev > 1e-12).sum())
    nneg = int((ev < -1e-12).sum())
    nnul = int((ev.abs() <= 1e-12).sum())
    verdict = "DEGENERATE" if nnul else ("INDEFINITE" if npos and nneg else "definite")
    label = "  <- Killing" if b == 0.0 else ("  <- Klein" if a == 0.0 else "")
    print(f"  a={a:+5.1f} b={b:+5.1f} : signature (+{npos}, -{nneg}, null {nnul})"
          f"  det={torch.det(M):+.4f} (=-b^6: {-b ** 6:+.4f})"
          f"  {verdict}{label}")
print("\nNo (a, b) yields a definite form => se(3) has no Ad-invariant inner product.")

# ------------------------------------- (3) LNKillingRelu degeneracies
banner("(3) LNKillingRelu on se(3)")
relu = LNKillingRelu(8, algebra_type='se3')

x_t = torch.zeros(2, 8, 6, 5)
x_t[:, :, 0:3, :] = torch.randn(2, 8, 3, 5)             # omega = 0: pure translation
with torch.no_grad():
    y_t = relu(x_t)
print(f"pure-translation input (w = 0):  ||out - in|| = "
      f"{torch.linalg.norm(y_t - x_t):.3e}   (exact identity on t)")

x = torch.randn(2, 8, 6, 5)
x_pert = x.clone()
x_pert[:, :, 0:3, :] += torch.randn(2, 8, 3, 5) * 5.0   # perturb v slot only
with torch.no_grad():
    def gate(z):
        d = relu.learn_dir(z.transpose(1, -1)).transpose(1, -1).transpose(2, -1)
        return killingform_se3(hat(z.transpose(2, -1)), hat(d))
    g0, g1 = gate(x), gate(x_pert)
print(f"gate change under a v-only perturbation      : "
      f"{torch.linalg.norm(g0 - g1):.3e}   (gate cannot see v)")
print(f"fraction of gates on the identity branch     : {(g0 <= 0).double().mean():.2f}")

# ------------------------------- (4) Killing normalizer blows up
banner("(4) Killing normalizer 1/||w||^2 on bracket outputs w1 x w2")
w1 = torch.tensor([1.0, 0.0, 0.0])
for ang in [1e-1, 1e-3, 1e-6]:
    w2 = torch.tensor([1.0, ang, 0.0])
    w2 = w2 / w2.norm()
    kf = torch.cross(w1, w2, dim=0).pow(2).sum().item()
    print(f"  angle ~ {ang:.0e} rad : ||w1 x w2||^2 = {kf:.3e}   1/kf = {1.0 / kf:.3e}")
print("\nOnly the EPS clamp in LNBatchNorm keeps this finite.  Use the Lie bracket")
print("for nonlinearity; if an invariant scalar gate is needed, use the Klein")
print("pairing q^T Q k between DISTINCT channels -- never the self-form (which")
print("vanishes identically on Pluecker-lifted lines) and never as a divisor.")

# Congruence-Equivariant Tensor Prediction from Point Clouds with Lie Neurons

**Technical Report**

Code under test: `experiment/pc_se3_congruence/` (`se3_utils.py`, `encoders.py`, `models.py`, `verify.py`) and the completed `core/` layers (`lie_alg_util.py`, `lie_neurons_layers.py`).
Environment: Python 3.11, PyTorch 2.11.0+cu128, `torch.float64`, CPU. Each experiment block reseeds (0/1/2/3/4) so its numbers are reproducible in isolation. Total runtime of `verify.py`: 1.2 s.

---

## Abstract

We show, by construction and by float64-precision numerical verification with **random (untrained) weights**, that a network built exclusively from Lie Neurons primitives — `LNLinear` and `LNLieBracket` on $\mathfrak{se}(3)$ — composed with a Plücker-type point-cloud lifting satisfies three exact transformation laws under any rigid motion $T=(R,p)\in SE(3)$:

- **(A)** a *compliance-type* output $C(T\cdot P)=\mathrm{Ad}_T\,C(P)\,\mathrm{Ad}_T^{\top}$, obtained from a plain Gram head $C=ZZ^{\top}/C_{\text{out}}$;
- **(B)** a *stiffness-type* output $K(T\cdot P)=\mathrm{Ad}_T^{-\top}K(P)\,\mathrm{Ad}_T^{-1}$, obtained from the same backbone with a single Klein-form intertwiner $Y=QZ$ in the head;
- **(C)** for covector ($\mathfrak{se}(3)^{*}$) inputs — raised to twists by the sharp map $Q^{-1}$, processed, then lowered again by $Q$ — one-step equivariance and the two-step *cascade* property: if $T_1$ produces $K_1$, then $T_2T_1$ produces $K_2=\mathrm{Ad}_{T_2}^{-\top}K_1\mathrm{Ad}_{T_2}^{-1}$.

Two point-cloud $\to\mathfrak{se}(3)$ encoders are tested — the closed-form pairwise Plücker lifting and a learnable VN-DGCNN direction-field lifting — and we prove that they are two parameterizations of the *same* underlying map, the origin-referenced screw lifting $L_0(r,n)=(n,\ r\times n)$. All positive tests sit at the float64 round-off floor ($10^{-16}$–$10^{-12}$ across translation magnitudes $\lVert p\rVert\in\{0,1,10^2,10^4\}$), while five intentionally broken negative controls fail at $O(1)$ whenever $p\neq 0$, confirming the discriminative power of the test suite. Experiment C additionally settles whether the covector space carries a nonlinearity of its own: the Lie bracket applied directly to wrenches fails at $O(1)$ (it is equivariant for $\mathrm{Ad}$, not $\mathrm{Ad}^{-\top}$), yet the transported bracket $[F_1,F_2]_{*}=(f_1\times m_2-f_2\times m_1,\ f_1\times f_2)$ yields a congruence-equivariant pipeline that never constructs $Q$ — the dual's equivariant bilinear space having dimension 2, the same as $\mathfrak{se}(3)$'s (§4.4). As a prerequisite we completed the repository's $\mathfrak{se}(3)$ support (`killingform_se3` plus two latent bugs), so that `LNKillingRelu`, `LNBatchNorm`, `LNMaxPool` and `LNInvariant` no longer raise on $\mathfrak{se}(3)$. None of them is used here: since $\mathfrak{se}(3)$ admits no Ad-invariant inner product, Killing-based gating and normalization are ill-founded on it, and the Lie bracket — which needs no form at all — is the nonlinearity throughout.

---

## 0. Conclusion First

> **All three target laws hold structurally — for arbitrary weights — using only `LNLinear` + `LNLieBracket` in the backbone.**
>
> $$\text{(A)}\quad C(T\cdot P)=\mathrm{Ad}_T\,C(P)\,\mathrm{Ad}_T^{\top},\qquad
> \text{(B)}\quad K(T\cdot P)=\mathrm{Ad}_T^{-\top}K(P)\,\mathrm{Ad}_T^{-1},$$
> $$\text{(C)}\quad K\!\left(\mathrm{Ad}_{T_2T_1}^{-\top}W\right)=\mathrm{Ad}_{T_2}^{-\top}\,K\!\left(\mathrm{Ad}_{T_1}^{-\top}W\right)\mathrm{Ad}_{T_2}^{-1}.$$

| Question | Answer | Evidence |
|---|---|---|
| Can a Lie-Neurons (linear + bracket only) network output $C\mapsto \mathrm{Ad}_T C\mathrm{Ad}_T^{\top}$? | **Yes**, with a plain Gram head; no Klein form needed anywhere. | Exp. A: $\le 3.8\text{e-}12$ |
| Can the same backbone output $K\mapsto \mathrm{Ad}_T^{-\top}K\mathrm{Ad}_T^{-1}$? | **Yes**, iff the head first applies the Klein intertwiner $Y=QZ$. Dropping $Q$ fails at $O(1)$. | Exp. B: $\le 6.9\text{e-}12$; neg. control B4 |
| Given covectors, does the cascade law $T_2T_1 \Rightarrow K_2=\mathrm{Ad}_{T_2}^{-\top}K_1\mathrm{Ad}_{T_2}^{-1}$ hold? | **Yes**, exactly; it follows from one-step equivariance + the homomorphism property of $\mathrm{Ad}^{-\top}$. | Exp. C: $\le 1.3\text{e-}15$ |
| Does the Lie bracket work on covectors without $Q$? | **No** — it is equivariant for $\mathrm{Ad}$, not $\mathrm{Ad}^{-\top}$; passes at $p=0$, $O(1)$ error otherwise. `LNLinear` is unaffected. | Exp. C4/C4b |
| Then is there *any* nonlinearity native to $\mathfrak{se}(3)^{*}$? | **Yes, and it is essentially unique**: the transported bracket $[F_1,F_2]_{*}=(f_1\times m_2-f_2\times m_1,\ f_1\times f_2)$, giving a $Q$-free pipeline. But it exists *because* $Q$ makes $\mathfrak{g}\cong\mathfrak{g}^{*}$. | §4.4, Exp. C5–C7 |
| Are the Plücker lifting and the learnable lifting the same operation? | **Yes** — both are the origin-referenced screw lifting $L_0(r,n)=(n, r\times n)$; they differ only in where the direction $n$ comes from. | §3, Exp. L1/L2: $\le 2.8\text{e-}16$ |
| Was the repo's $\mathfrak{se}(3)$ support complete? | **No** — `killingform` had no se3 branch and two layers had latent bugs; now fixed and verified. | §5 |
| Should the $\mathfrak{se}(3)$ Killing form then be used for gating/normalization? | **No.** $\mathfrak{se}(3)$ admits no invariant inner product; the Killing branch is blind to $v$ and is the exact identity on $\mathfrak{t}$. Experiments A–C use zero Killing form (0 calls, runtime-verified). | §5, `check_killing_degeneracy.py` |

As in the companion report (`exp.md`), the entire judgement rests on the $p\neq 0$ region: every negative control passes perfectly at $p=0$ because $\mathrm{Ad}_{(R,0)}$ is orthogonal, and is exposed only once translations enter.

---

## 1. Problem Statement

Let $P=\{r_i\}_{i=1}^{N}\subset\mathbb{R}^3$ be a point cloud, acted on by $T\cdot P=\{Rr_i+p\}$. We verify three propositions, each for **arbitrary network weights** (all runs use random initialization; nothing is trained):

**(A) Twist–twist (compliance-type) tensor.**
$$f_A(T\cdot P)=\mathrm{Ad}_T\, f_A(P)\,\mathrm{Ad}_T^{\top},\qquad f_A(P)=ZZ^{\top}/C_{\text{out}} .$$
This is the transformation law of a type-$(2,0)$ tensor over twist space (e.g. a compliance $C=K^{-1}$), and it requires **no Klein form at all** — a relevant design point for the bracket-only program, where the Klein form is deliberately held back.

**(B) Wrench–wrench (stiffness-type) tensor.**
$$f_B(T\cdot P)=\mathrm{Ad}_T^{-\top} f_B(P)\,\mathrm{Ad}_T^{-1},\qquad f_B(P)=(QZ)(QZ)^{\top}/C_{\text{out}} .$$
Identical backbone; the Klein form enters **only in the head**, as the constant intertwiner from $\mathfrak{se}(3)$ to $\mathfrak{se}(3)^{*}$.

**(C) Covector input and cascade composition.** Assume wrench-valued features $W\in\mathbb{R}^{6\times C}$ are given (transforming as $W\mapsto \mathrm{Ad}_T^{-\top}W$), rather than a point cloud. Verify one-step equivariance of $f_C(W)=K$ and the cascade law: with $W_1=\mathrm{Ad}_{T_1}^{-\top}W_0 \Rightarrow K_1$ and $W_2=\mathrm{Ad}_{T_2T_1}^{-\top}W_0 \Rightarrow K_2$,
$$K_2=\mathrm{Ad}_{T_2}^{-\top}\,K_1\,\mathrm{Ad}_{T_2}^{-1}.$$

## 2. Conventions

Following `exp.md` and `pc_to_se3_mapping_en.pdf`, this report writes elements of $\mathfrak{se}(3)$ **angular-first**:

$$\xi=\begin{pmatrix}\omega\\ v\end{pmatrix}\in\mathbb{R}^6,
\qquad
\xi^{\wedge}=\begin{bmatrix}\hat\omega & v\\ 0 & 0\end{bmatrix}\in\mathbb{R}^{4\times4},
\qquad \hat a\,x=a\times x .$$

Wrenches $F=(m,f)\in\mathfrak{se}(3)^{*}$ are moment-first, so that $F^{\top}\xi$ is power. In this ordering every $6\times6$ block matrix below is written in $[\omega\,;\,v]$ blocks:

$$\mathrm{Ad}_T=\begin{bmatrix} R & 0\\ \hat p R & R\end{bmatrix},\qquad
Q=\begin{bmatrix} 0 & I_3\\ I_3 & 0\end{bmatrix},\qquad
\xi_1^{\top} Q\, \xi_2=\omega_1\!\cdot\! v_2+\omega_2\!\cdot\! v_1 .$$

Feature tensors are $[B, C, 6, N]$ (channels $C$, one twist per column). Three identities are used throughout, verified in Experiment 0:

- **(I-1)** homomorphism: $\mathrm{Ad}_{T_1}\mathrm{Ad}_{T_2}=\mathrm{Ad}_{T_1T_2}$;
- **(I-2)** Klein invariance: $\mathrm{Ad}_T^{\top} Q\, \mathrm{Ad}_T=Q$ (so $\mathrm{Ad}:SE(3)\to O(3,3)$);
- **(I-3)** intertwiner (**flat**, lowering): $Q\,\mathrm{Ad}_T\,Q^{-1}=\mathrm{Ad}_T^{-\top}$;
- **(I-4)** intertwiner (**sharp**, raising): $Q^{-1}\mathrm{Ad}_T^{-\top}Q=\mathrm{Ad}_T$, i.e. $Q^{-1}\mathrm{Ad}_T^{-\top}=\mathrm{Ad}_T\,Q^{-1}$ — (I-3) read backwards.

**A note on $Q$ versus $Q^{-1}$.** A bilinear form on $V$ *is* a map $V\to V^{*}$, so the Klein Gram matrix is the musical **flat**
$$Q:\ \mathfrak{se}(3)\longrightarrow\mathfrak{se}(3)^{*},\qquad \text{twist}\mapsto\text{wrench},$$
and its inverse is the **sharp** $Q^{-1}:\mathfrak{se}(3)^{*}\to\mathfrak{se}(3)$. Because $Q$ is the block swap, $Q^{2}=I_6$ and the two are the *same matrix numerically* — but they are maps between different spaces, and we write them distinctly. Head B lowers ($Y=QZ$, I-3); model C raises at its input ($X=Q^{-1}W$, I-4) and lowers again in its head. Experiment B4 measures what conflating the twist and wrench types costs: $O(1)$ failure as soon as $p\neq0$.

$\mathrm{Ad}_T^{-1}$ is always evaluated in closed form as $\mathrm{Ad}_{T^{-1}}$, $T^{-1}=(R^{\top},-R^{\top}p)$, to avoid conditioning loss at $\lVert p\rVert = 10^4$.

> **Code correspondence — stated once here and not repeated.** The code now uses the **same** order as this prose. `HatLayer('se3')` and `vee_se3` in `core/lie_alg_util.py` store $[\omega\,;\,v]$, i.e. `x[..., 0:3]` $=\omega$ and `x[..., 3:6]` $=v$; wrenches are stored $[m\,;\,f]$; and `experiment/pc_se3_congruence/se3_utils.py` builds $\mathrm{Ad}_T=\bigl[\begin{smallmatrix} R & 0\\ \hat pR & R\end{smallmatrix}\bigr]$ to match, with the coupling block in the lower left. Consequently $K$ has $K[0{:}3,0{:}3]=$ rotational (`mm`) and $K[3{:}6,3{:}6]=$ translational (`ff`).
>
> Until 2026-08-10 the code stored the opposite (linear/force first) order, so `se3_utils.adjoint` then read $\bigl[\begin{smallmatrix} R & \hat pR\\ 0 & R\end{smallmatrix}\bigr]$. The two orders are conjugate by the block-swap permutation $\Pi=\bigl[\begin{smallmatrix}0&I\\ I&0\end{smallmatrix}\bigr]$:
> $$\mathrm{Ad}^{[\omega;v]}_T=\Pi\,\mathrm{Ad}^{[v;\omega]}_T\,\Pi ,$$
> and $Q$ is numerically the same matrix in both orders (it *is* the swap), so $Y=QZ$ needed no adjustment. Every number reported here — relative errors, ranks, dimensions, signatures, eigenvalues — is invariant under $\Pi$-conjugation, so **no result in this document changed with the switch**; verified by re-running the full suite before and after. The metric names `err_rel_ff` / `err_rel_mm` also kept their physical meaning (translational / rotational), because the block slices were renamed together with the storage. Datasets written before the switch (`meta.json` version < 3) are converted on load.

---

## 3. Point Cloud → $\mathfrak{se}(3)$: One Lifting, Two Parameterizations

Both encoders instantiate the **origin-referenced screw (Plücker) lifting**

$$L_0(r,n) := \begin{pmatrix} n \\ r\times n\end{pmatrix}\in\mathfrak{se}(3),
\qquad \omega = n,\quad v = r\times n .$$

$L_0(r,n)$ is the Plücker coordinate vector of the line through the point $r$ with direction $n$: the direction sits in the angular slot, the moment about the origin in the linear slot.

### 3.1 Equivariance of $L_0$ (the only proof that matters)

Let the direction field be rotation-only: $n'(T\cdot P)=Rn(P)$ — true for any translation-invariant construction. With $r'=Rr+p$:

$$r'\times n' = (Rr+p)\times Rn = R(r\times n) + p\times(Rn) = R(r\times n)+\hat p R\, n .$$

Stacking angular-first,

$$L_0(r',n')=\begin{pmatrix} Rn \\ R(r\times n)+\hat pR\,n\end{pmatrix}
=\begin{bmatrix} R & 0\\ \hat pR & R\end{bmatrix}
\begin{pmatrix} n \\ r\times n\end{pmatrix}
=\mathrm{Ad}_T\,L_0(r,n). \qquad\blacksquare$$

The only two facts used are $Ra\times Rb=R(a\times b)$ and bilinearity of $\times$. The $\hat pR$ term generated by the moment is exactly the lower-left block of $\mathrm{Ad}_T$ — the block that couples the angular slot into the linear slot: **absolute position enters the feature only through the moment, and the moment is what carries the translation part of the representation.** Equivalently, the angular slot is translation-blind ($\omega'=R\omega$ with no $p$), which is why the direction field only has to be $SO(3)$-equivariant.

### 3.2 The two encoders

**(E1) Closed-form pairwise Plücker lifting** (no parameters). For each point $r_i$ and its $k$ nearest neighbors $r_j$:
$$\xi_{ij}=L_0(r_i,\ d_{ij}),\qquad d_{ij}=r_j-r_i .$$
Channel $c$ collects the rank-$c$ neighbor (distance rank is $SE(3)$-invariant); mean over $i$ gives $V\in\mathbb{R}^{6\times k}$.

**(E2) Learnable VN lifting.** Center the cloud ($x_i=r_i-c$, $c=$ centroid), run a small VN-DGCNN (EdgeConv ×2 + VN-linear + VN-LeakyReLU, $1\to 8\to 8\to C$) to get an $SO(3)$-equivariant direction field $n_i^{(ch)}$, lift **at the anchor**, pool, and transport back to the origin:
$$V=\mathrm{Ad}_{(I,c)}\ \frac{1}{N}\sum_i \bigl(n_i^{(ch)},\ (r_i-c)\times n_i^{(ch)}\bigr).$$

### 3.3 Equivalence of the two encoders (simple form)

Three one-line lemmas connect everything.

**Lemma 1 (two moment formulas are identical).**
$$r_i\times d_{ij} = r_i\times(r_j-r_i) = r_i\times r_j - \underbrace{r_i\times r_i}_{0} = r_i\times r_j .$$
So the "docs" convention $(p_j-p_i,\ p_i\times p_j)$ and the "exp/pdf" convention $(d_{ij},\ r_i\times d_{ij})$ are the *same map*, not merely similar. *Measured: $1.1\text{e-}16$ (Exp. L1).*

**Lemma 2 (anchor transport = origin referencing).** For any reference $c$:
$$\mathrm{Ad}_{(I,c)}\bigl(n,\ (r-c)\times n\bigr) = \bigl(n,\ c\times n+(r-c)\times n\bigr) = \bigl(n,\ r\times n\bigr) = L_0(r,n),$$
using only $\mathrm{Ad}_{(I,c)}(\omega,v)=(\omega,\ v+c\times\omega)$ and bilinearity. So (E2) with anchor transport is literally $\frac1N\sum_i L_0(r_i, n_i)$ with a learned $n_i$. *Measured: $3.6\text{e-}16$ (Exp. L2).*

**Lemma 3 (what happens without transport — negative control).** Dropping $\mathrm{Ad}_{(I,c)}$ leaves moments referenced to the *data-dependent* anchor, and since $x_i$ and $n_i$ are both rotation-only, the output transforms as $\mathrm{diag}(R,R)$ — the $SO(3)\times SO(3)$ collapse. It agrees with $\mathrm{Ad}_T$ only when $p=0$. *Measured: passes at $p=0$ ($1.4\text{e-}15$), fails at $O(1)$ for all $p\neq0$ (Exp. L5).*

**Summary.** Both encoders compute $\sum_i w_i\, L_0(r_i, n_i)$ with invariant weights; they differ only in the direction field:
$$\text{(E1)}:\ n_i = d_{ij}\ \text{(pairwise differences, } j = \text{rank-}c\text{ neighbor)},\qquad
\text{(E2)}:\ n_i = \mathrm{VN}(P)_i\ \text{(learned)}.$$
Equivariance of both is the single Theorem of §3.1. This is the precise sense in which the two documents (`docs/se3_equivariant_pointcloud.md` and `exp.md`/`pc_to_se3_mapping_en.pdf`) describe the same operation.

---

## 4. Architecture

$$P\ \xrightarrow{\ \text{Encoder (E1 or E2)}\ }\ V\in\mathbb{R}^{6\times 8}
\ \xrightarrow{\ \text{Backbone}\ }\ Z\in\mathbb{R}^{6\times 8}
\ \xrightarrow{\ \text{Head}\ }\ C \text{ or } K\in\mathbb{S}^6 .$$

### 4.1 Backbone — `LNLinear` + `LNLieBracket` only

Three blocks of `LNLinearAndLieBracket(algebra_type='se3')`, channels $8\to16\to16\to8$, exactly as shipped in `core/lie_neurons_layers.py`:

- `LNLinear`: $V\mapsto VW$ (channel mixing, bias-free). Since $\mathrm{Ad}_T$ acts on the left and $W$ on the right, $(\mathrm{Ad}_TV)W=\mathrm{Ad}_T(VW)$ — trivially equivariant. **Bias is forbidden** (negative control A4).
- `LNLieBracket`: $V\mapsto V+\mathrm{vee}\bigl([\hat d_2,\hat d_1]\bigr)$ with learned directions $d_1=VW_1,\ d_2=VW_2$. Equivariance is the automorphism property $[\mathrm{Ad}_T\xi_1,\mathrm{Ad}_T\xi_2]=\mathrm{Ad}_T[\xi_1,\xi_2]$; the residual term is equivariant by linearity. This is the **only nonlinearity** used in experiments A–C. No Killing ReLU, no batch norm, no pooling, no Klein-form gate appears in the backbone.

### 4.2 Heads

**Head A (Gram).** $C=ZZ^{\top}/C_{\text{out}}$. If $Z\mapsto \mathrm{Ad}_TZ$ then $C\mapsto \mathrm{Ad}_T C \mathrm{Ad}_T^{\top}$ — precisely target (A), with no extra structure. (This is the same algebra that makes $ZZ^\top$ the *wrong* type for a stiffness — see B4.)

**Head B (Klein intertwiner + Gram).** Lower the twist features to wrenches with the flat map, then form the Gram: $Y=QZ$, $K=YY^{\top}/C_{\text{out}}$. By (I-3), $Y\mapsto \mathrm{Ad}_T^{-\top}Y$, hence
$$K\mapsto \mathrm{Ad}_T^{-\top}\,K\,\mathrm{Ad}_T^{-1},$$
and $K$ is symmetric PSD by construction, SPD iff $\operatorname{rank}Z=6$ (guaranteed generically by $C_{\text{out}}=8\ge6$).

### 4.3 Model C — covector inputs

Design (details filled in as requested): the input is a batch of wrench features $W\in\mathbb{R}^{6\times C}$ transforming in the dual representation $W\mapsto \mathrm{Ad}_T^{-\top}W$ (physically: force/moment measurements, contact wrenches). The backbone of §4.1 is a *twist* network, so the wrenches must first be **raised** with the sharp map $Q^{-1}:\mathfrak{se}(3)^{*}\to\mathfrak{se}(3)$. By (I-4),

$$Q^{-1}\,\mathrm{Ad}_T^{-\top}\,Q=\mathrm{Ad}_T \quad\Longrightarrow\quad X:=Q^{-1}W\ \text{transforms as}\ X\mapsto\mathrm{Ad}_T X,$$

so the backbone is reused verbatim, and head B **lowers** back with the flat map $Q$:

$$f_C(W)=\mathrm{HeadB}\bigl(\mathrm{Backbone}(Q^{-1}W)\bigr),
\qquad
\mathfrak{se}(3)^{*}\xrightarrow{\ Q^{-1}\ }\mathfrak{se}(3)\xrightarrow{\ \text{Backbone}\ }\mathfrak{se}(3)\xrightarrow{\ Q\ }\mathfrak{se}(3)^{*}\xrightarrow{\ \text{Gram}\ }\mathbb{S}^6 .$$

$Q^{-1}$ and $Q$ are the same matrix numerically ($Q^{2}=I_6$), so this costs nothing at runtime, but writing them distinctly is what keeps the type bookkeeping honest — the round trip $\mathfrak{se}(3)^{*}\to\mathfrak{se}(3)\to\mathfrak{se}(3)^{*}$ is what makes the output a wrench–wrench tensor rather than a twist–twist one.

One-step equivariance follows by composing the three equivariant stages; the cascade law follows from one-step equivariance plus the homomorphism $\mathrm{Ad}_{T_2T_1}^{-\top}=\mathrm{Ad}_{T_2}^{-\top}\mathrm{Ad}_{T_1}^{-\top}$ (verified independently as C3). The experiment draws $W_0\sim\mathcal N(0,1)^{6\times 8}$, fixes $T_1$ with $\lVert p_1\rVert\sim1$, and sweeps $T_2$ over the translation scales.

### 4.4 Is there a nonlinearity native to $\mathfrak{se}(3)^{*}$? — dropping $Q$ entirely

The route above raises to $\mathfrak{se}(3)$ because the Lie bracket is a map $\mathfrak{g}\times\mathfrak{g}\to\mathfrak{g}$, equivariant for $\mathrm{Ad}$ and *not* for $\mathrm{Ad}^{-\top}$. It is worth asking whether that detour is necessary — whether the covector space carries a nonlinearity of its own.

**The coadjoint representation, in coordinates.** From $\mathrm{Ad}_T^{-1}=\mathrm{Ad}_{T^{-1}}$ and $R\,\widehat{R^{\top}p}=\hat pR$,

$$\mathrm{Ad}_T^{-\top}=\begin{bmatrix} R & \hat pR\\ 0 & R\end{bmatrix},
\qquad
\mathrm{Ad}_T^{-\top}\begin{pmatrix} m\\ f\end{pmatrix}
=\begin{pmatrix} Rm + p\times Rf\\ Rf\end{pmatrix},$$

the familiar wrench transformation law. Structurally it is the *mirror image* of the twist case: for twists the angular slot $\omega$ is translation-blind and $v$ absorbs the $\hat p$ coupling; for wrenches the **force** slot $f$ is translation-blind and the **moment** $m$ absorbs it. That mirror is exactly what $Q$ implements, and it is why (I-4) holds.

**Consequence — a native covector bracket exists, and it is essentially unique.** Since $Q$ is an isomorphism *of representations* $\mathfrak{se}(3)\to\mathfrak{se}(3)^{*}$, conjugation by $Q$ is a bijection between equivariant bilinear maps on $\mathfrak{se}(3)$ and on $\mathfrak{se}(3)^{*}$. Transporting the bracket gives a closed form containing no $Q$ at all:

$$[F_1,F_2]_{*}\ :=\ Q\bigl[\,Q^{-1}F_1,\ Q^{-1}F_2\,\bigr]
=\begin{pmatrix} f_1\times m_2 - f_2\times m_1\\ f_1\times f_2\end{pmatrix}
\quad\text{(as }(m,f)\text{)} . \tag{4.4}$$

Compare the twist bracket $[\xi_1,\xi_2]=(\omega_1\times\omega_2,\ \omega_1\times v_2-\omega_2\times v_1)$: the two formulas are the same expression with the roles of the slots exchanged, $\omega\leftrightarrow f$, $v\leftrightarrow m$. By Theorem 5.2 of `docs/se3_equivariant_pointcloud.md` the space of equivariant bilinear maps on $\mathfrak{se}(3)$ is 2-dimensional ($[\cdot,\cdot]$ and $N\circ[\cdot,\cdot]$), so the same is true on the dual, spanned by $[\cdot,\cdot]_{*}$ and $N_{*}=QNQ^{-1}:(m,f)\mapsto(f,0)$. *(Measured independently in C7: the equivariant bilinear space on $\mathfrak{se}(3)^{*}$ has dimension exactly 2.)*

So a fully $Q$-free covector pipeline exists — `LNLinear` + $[\cdot,\cdot]_{*}$ + Gram — and it is what **model C-native** implements. Note the honest reading: $Q$ has not been eliminated, it has been *compiled away* into (4.4). The operation exists precisely because $\mathfrak{se}(3)$ possesses a non-degenerate invariant form; a Lie algebra without one has $\mathfrak{g}\not\cong\mathfrak{g}^{*}$ and admits **no** equivariant bilinear operation on its dual whatsoever.

**Two negative controls isolate the failure.** Applying the ordinary twist bracket directly to wrench coordinates (model C-naive) treats $m$ as if it were $\omega$; since $\mathrm{Ad}$ and $\mathrm{Ad}^{-\top}$ agree only where $\mathrm{Ad}$ is orthogonal, it passes at $p=0$ and fails at $O(1)$ otherwise. By contrast `LNLinear` alone is untroubled on covectors: channel mixing acts on the *right* and therefore commutes with *any* left representation. The bracket, and only the bracket, is what needs the type to be right.

---

## 5. Completing the Repository's $\mathfrak{se}(3)$ Support

Before this work `core/` ran `LNLinear`, `LNLieBracket` and `LNLinearAndLieBracket` on $\mathfrak{se}(3)$, but every Killing-form-dependent layer raised. Three changes fix that. `killingform_se3` was added to `core/lie_alg_util.py`, returning $\omega_x\!\cdot\!\omega_d$ — a dedicated branch is needed because the naive trace $\mathrm{tr}(\hat x^{\top}\hat d)=2\,\omega_x\!\cdot\!\omega_d+v_x\!\cdot\! v_d$ is *not* Ad-invariant, so the $v$ term must be dropped. `LNLieBracketNoResidualConnect` had `HatLayer()` hard-coded to the sl(3) default (shape error 6 vs 8) and ignored `algebra_type` in its inner `LNKillingRelu`. `LNInvariant` failed to forward `algebra_type`. Post-fix equivariance with random weights (float64 at $\lVert p\rVert\sim100$; float32 via the pre-existing `docs/scripts/check_layers.py`):

| Layer (se3) | before | after (float64) | after (float32) |
|---|---|---|---|
| `LNKillingRelu` | `ValueError` | $2.1\text{e-}16$ | $3.8\text{e-}06$ |
| `LNLieBracketNoResidualConnect` | `RuntimeError` | $5.5\text{e-}16$ | $5.7\text{e-}07$ |
| `LNBatchNorm` (dim=4) | `ValueError` | $1.5\text{e-}16$ | $1.9\text{e-}06$ |
| `LNMaxPool` | `ValueError` | — | $0.0$ |
| `LNInvariant` (invariance) | `ValueError` | $4.3\text{e-}16$ | — |

These are bug fixes, not endorsements. None of the unlocked layers is used in experiments A–C — a runtime tripwire counts **0** `killingform` calls across every forward pass — because $\mathfrak{se}(3)$ admits no Ad-invariant inner product, so the VN-style "fold against a learned direction" that `LNKillingRelu` / `LNBatchNorm` / `LNMaxPool` implement has no metric to fold against. `killingform_se3` carries this caveat in its docstring, and `check_killing_degeneracy.py` reproduces the supporting numbers (rank 3 radical $\mathfrak{t}$; `LNKillingRelu` exactly the identity on $\mathfrak{t}$; the $\lVert\omega\rVert^{2}$ normalizer diverging on near-collinear bracket outputs). The Lie bracket, which needs no form at all, is the nonlinearity used throughout.

---

## 6. Verification Methodology

**Metric.** Scale-free relative error, immune to output magnitude:
$$e = \frac{\lVert f(T\cdot x)-\rho(T)f(x)\rVert_F}{\max\bigl(\lVert f(x)\rVert_F,\ \lVert\rho(T)f(x)\rVert_F\bigr)} ,$$
with $\rho(T)$ the target output representation of each experiment.

**Transformation sampling.** $R$ Haar-uniform (QR of a Gaussian matrix, det-corrected); $p\sim\mathcal N(0,s^2 I)$ with $s\in\{0,\ 1,\ 10^2,\ 10^4\}$; 5 trials per scale, maximum error reported. The $p=0$ column is a *control*, not evidence: every defect studied here is invisible at $p=0$.

**Data.** $B=2$ clouds of $N=64$ Gaussian points, $k=8$ neighbors; covariance features $W_0\sim\mathcal N(0,1)$; all weights random, seed 0, no training.

**Negative controls** (each breaks exactly one assumption): (L5) anchor lift without transport; (A4) bias added to the first `LNLinear`; (B4) head without the Klein intertwiner ($K=ZZ^{\top}$ tested against the congruence law); (B5) $K+\varepsilon I$ regularization, $\varepsilon=10^{-3}$; (C4) the twist Lie bracket applied directly to covector features.

**Rows that are not equivariance errors.** Six entries report a different kind of quantity and are labelled as such in §7: L1 and C6 are exact identities between two closed-form expressions, O5 is an involution check, B3 reports eigenvalues of $K$, L2 compares two encoder variants, and **C7 is a dimension count** rather than a test of any particular network.

C7 asks whether the covector nonlinearity (4.4) has any competitors. Write a general bilinear map $B:\mathfrak{se}(3)^{*}\times\mathfrak{se}(3)^{*}\to\mathfrak{se}(3)^{*}$ in coordinates: $6$ output components, each a bilinear form with $36$ coefficients, so $216$ unknowns. Equivariance under the coadjoint representation $\rho(T)=\mathrm{Ad}_T^{-\top}$ requires
$$B\bigl(\rho F_1,\ \rho F_2\bigr)-\rho\,B(F_1,F_2)=0,$$
which is **linear** in those 216 coefficients and yields 6 equations per sampled triple $(T;F_1,F_2)$. Stacking 120 random triples gives a $720\times216$ system; the dimension of its null space — computed by counting singular values below $10^{-8}\sigma_{\max}$ — is the dimension of the space of equivariant bilinear maps on the dual. The result, **2**, matches the known dimension on $\mathfrak{se}(3)$ itself (Theorem 5.2 of `docs/se3_equivariant_pointcloud.md`: $[\cdot,\cdot]$ and $N\circ[\cdot,\cdot]$), confirming that $[\cdot,\cdot]_{*}$ and $N_{*}$ already span everything and no further covector operation remains to be found. This mirrors the method of `docs/scripts/enumerate_se3.py`, applied to the dual instead of the algebra.

---

## 7. Results

All values are max scaled error over 5 trials, except the six rows noted in §6 (L1, L2, B3, C6, C7, O5) which report identities, eigenvalues or a dimension. **Bold** rows are negative controls (expected to fail for $p\ne 0$).

### 7.1 Experiment 0 — algebraic identities

**What.** The four matrix identities (I-1)–(I-4) plus the involution $Q^{2}=I_6$, checked directly on $6\times6$ matrices.

**Why.** Every equivariance proof in §3–§4 rests on these; a convention slip (wrong block order, wrong sign of $\hat p$, flat/sharp confusion) would silently invalidate everything downstream. Testing them separately localizes such a mistake here instead of letting it surface as an unexplained end-to-end failure.

**Setup.** No network and no data — only $\mathrm{Ad}_T$, $\mathrm{Ad}_{T}^{-1}$ (closed form $\mathrm{Ad}_{T^{-1}}$) and $Q$. Five random $T$ per translation scale.

| Check | $p=0$ | $\lVert p\rVert\sim1$ | $\sim10^2$ | $\sim10^4$ |
|---|---|---|---|---|
| O1 $\mathrm{Ad}_{T_2}\mathrm{Ad}_{T_1}=\mathrm{Ad}_{T_2T_1}$ | 1.5e-16 | 2.7e-16 | 4.5e-16 | 3.9e-16 |
| O2 $\mathrm{Ad}_T^{\top}Q\mathrm{Ad}_T=Q$ | 3.7e-16 | 6.0e-16 | 2.5e-14 | 2.9e-12 |
| O3 **flat** $Q\mathrm{Ad}_TQ^{-1}=\mathrm{Ad}_T^{-\top}$ (I-3) | 0.0 | 4.5e-16 | 8.4e-16 | 4.0e-16 |
| O4 **sharp** $Q^{-1}\mathrm{Ad}_T^{-\top}Q=\mathrm{Ad}_T$ (I-4) | 0.0 | 2.3e-16 | 4.8e-16 | 4.9e-16 |
| O5 $QQ^{-1}=I_6$ (involution) | 0.0 | — | — | — |

### 7.2 Experiment L — lifting layer

**What.** The point cloud $\to\mathfrak{se}(3)$ stage in isolation: the two closed-form moment identities of §3.3, and $\mathrm{Ad}$-equivariance of both encoders.

**Why.** The lifting is the only place where the $SE(3)$ action on points becomes an $\mathrm{Ad}$ action on features, so it is where translation information either enters correctly or is destroyed. Isolating it separates "the encoder is wrong" from "the backbone is wrong" in experiments A and B. L5 additionally quantifies the anchor-transport trap, the failure mode the companion documents identify as the easiest to miss.

**Setup.** $B=2$ clouds of $N=64$ Gaussian points ($\sigma=2$), $k=8$ neighbors. Plücker encoder: parameter-free, channel $c$ = the rank-$c$ neighbor. Learnable encoder: VN-DGCNN ($1\to8\to8\to8$, EdgeConv $\times2$ + VN-linear + VN-LeakyReLU) with random weights, run in three modes — `origin`, `anchor_transport`, and `no_transport` (L5) — sharing one state dict so the three differ *only* in the transport step.

| Check | $p=0$ | $\sim1$ | $\sim10^2$ | $\sim10^4$ |
|---|---|---|---|---|
| L1 $r_i\times d_{ij}=r_i\times r_j$ (Lemma 1) | 1.1e-16 | | | |
| L2 anchor transport = origin lift (Lemma 2) | 2.8e-16 | | | |
| L3 Plücker encoder equivariance | 5.3e-16 | 4.5e-16 | 3.3e-15 | 4.7e-13 |
| L4 learnable encoder equivariance | 2.1e-15 | 1.5e-15 | 4.6e-14 | 3.5e-12 |
| **L5 no anchor transport** | 2.0e-15 | **6.7e-01** | **1.0e+00** | **1.0e+00** |

### 7.3 Experiment A — $C(T\cdot P)=\mathrm{Ad}_T C\,\mathrm{Ad}_T^{\top}$

**What.** Whether a Lie Neurons network can output a twist–twist (compliance-type) tensor obeying the covariant congruence law.

**Why.** This is the cheaper of the two target laws: a plain Gram head $ZZ^{\top}$ already transforms as $\mathrm{Ad}\,(\cdot)\,\mathrm{Ad}^{\top}$, so if it works, the whole pipeline can be built with **no invariant form anywhere** — directly relevant to the bracket-only program, which deliberately withholds the Klein form. A3 isolates the backbone square of the commuting diagram so that an encoder fault cannot be mistaken for a backbone fault; A4 confirms the test can actually detect a broken network.

**Setup.** Encoder (Plücker or learnable) $\to$ 3 blocks of `LNLinearAndLieBracket(algebra_type='se3')`, channels $8\to16\to16\to8$ $\to$ head $C=ZZ^{\top}/C_{\text{out}}$. Random weights, no training. A3 feeds $\mathrm{Ad}_T V_0$ to the backbone directly rather than transforming the cloud. A4 clones A1's weights and adds a bias ($\sigma=0.5$) to the first `LNLinear`.

| Check | $p=0$ | $\sim1$ | $\sim10^2$ | $\sim10^4$ |
|---|---|---|---|---|
| A1 end-to-end, Plücker encoder | 8.1e-16 | 9.6e-16 | 3.4e-15 | 4.8e-13 |
| A2 end-to-end, learnable encoder | 3.4e-15 | 1.9e-15 | 3.9e-14 | 3.8e-12 |
| A3 backbone square alone | 3.3e-16 | 3.6e-16 | 3.9e-16 | 4.1e-16 |
| **A4 bias in first `LNLinear`** | **1.3e+00** | **1.1e+00** | **9.9e-01** | **9.9e-01** |

Verdict: **design (A) is realizable with linear + bracket layers only, and needs no Klein form.** Note A4 is the one control that fails even at $p=0$ — a bias breaks rotation equivariance too.

### 7.4 Experiment B — $K(T\cdot P)=\mathrm{Ad}_T^{-\top}K\,\mathrm{Ad}_T^{-1}$

**What.** Whether the *same* backbone as experiment A can output a wrench–wrench (stiffness-type) tensor, which obeys the contravariant congruence law instead.

**Why.** A stiffness maps twists to wrenches, so it is a tensor on the dual space and transforms by $\mathrm{Ad}^{-\top}(\cdot)\mathrm{Ad}^{-1}$, not $\mathrm{Ad}(\cdot)\mathrm{Ad}^{\top}$. The question is how much machinery this costs: the claim under test is that a **single constant matrix in the head** suffices, with the backbone untouched. B4 measures the price of getting the type wrong, and is the sharpest control in the report — $ZZ^{\top}$ is not merely inaccurate, it is a different tensor type. B5 checks a regularizer that looks harmless and is not.

**Setup.** Identical to A except the head: $Y=QZ$ (flat map, I-3) then $K=YY^{\top}/C_{\text{out}}$. B3 reports $\mathrm{eig}(K)$ to confirm the head yields a genuine PSD matrix, full-rank because $C_{\text{out}}=8\ge6$. B4 clones B1's weights and deletes $Q$ from the head. B5 keeps B1 intact and post-composes $K\mapsto K+\varepsilon I_6$, $\varepsilon=10^{-3}$.

| Check | $p=0$ | $\sim1$ | $\sim10^2$ | $\sim10^4$ |
|---|---|---|---|---|
| B1 end-to-end, Plücker encoder | 1.3e-15 | 8.0e-16 | 5.7e-15 | 3.5e-13 |
| B2 end-to-end, learnable encoder | 4.8e-15 | 6.3e-15 | 9.9e-14 | 6.9e-12 |
| B3 $K$ PSD / rank | min eig $1.1\text{e-}07\ \ge 0$, rank 6 | | | |
| **B4 head without Klein ($K=ZZ^{\top}$)** | 9.5e-16 | **1.0e+00** | **1.0e+00** | **1.0e+00** |
| **B5 $K+\varepsilon I$** | 9.2e-16 | **4.3e-01** | **6.0e-01** | **6.6e-01** |

Verdict: **one constant matrix $Q$ in the head converts design (A) into design (B).** B4 quantifies the type error of omitting it: $ZZ^{\top}$ transforms covariantly ($\mathrm{Ad}\,\cdot\,\mathrm{Ad}^{\top}$), which coincides with the required contravariant law only on the orthogonal subgroup $p=0$.

### 7.5 Experiment C — covector input, cascade

**What.** The setting where the input is already a batch of wrenches (e.g. measured contact forces/moments) rather than a point cloud. C1–C3 are three *different* statements, easily conflated:

| | statement | transforms | baseline compared against | network? |
|---|---|---|---|---|
| **C1** | $f(\mathrm{Ad}_T^{-\top}W_0)=\mathrm{Ad}_T^{-\top}f(W_0)\mathrm{Ad}_T^{-1}$ | one, $T$ | the **untransformed** output $f(W_0)$ | yes, 1 evaluation |
| **C2** | $f(\mathrm{Ad}_{T_2T_1}^{-\top}W_0)=\mathrm{Ad}_{T_2}^{-\top}K_1\mathrm{Ad}_{T_2}^{-1}$ | two, $T_1$ then $T_2$ | the **already-transformed** output $K_1=f(\mathrm{Ad}_{T_1}^{-\top}W_0)$ | yes, 2 evaluations |
| **C3** | $\mathrm{Ad}_{T_2T_1}^{-\top}=\mathrm{Ad}_{T_2}^{-\top}\mathrm{Ad}_{T_1}^{-\top}$ | two | — (pure matrix identity) | **no** |

C1 is per-transform equivariance measured from the origin of the chain. C2 is the property actually wanted in use: *given the estimate $K_1$ already produced in frame $T_1$, does moving a further $T_2$ transform $K_1$ by congruence?* — the baseline is $K_1$, and $W_0$ never appears on the right-hand side. C3 removes the network entirely and asks only whether the dual action is a group homomorphism.

**Why.** Logically $\text{C1}+\text{C3}\Rightarrow\text{C2}$: apply C1 with input $\mathrm{Ad}_{T_1}^{-\top}W_0$ and transform $T_2$, then rewrite $\mathrm{Ad}_{T_2}^{-\top}\mathrm{Ad}_{T_1}^{-\top}$ as $\mathrm{Ad}_{T_2T_1}^{-\top}$ using C3. So C2 is not independent evidence — it is the end-to-end check that the two halves compose in the right *order*, which is exactly where a convention slip would land.

C3 is worth isolating because it is not automatic. Transposition reverses order, and so does inversion; in $\mathrm{Ad}^{-\top}$ the two reversals cancel,
$$(\mathrm{Ad}_{T_2}\mathrm{Ad}_{T_1})^{-\top}=\mathrm{Ad}_{T_2}^{-\top}\mathrm{Ad}_{T_1}^{-\top},$$
so the dual action is a genuine left representation. Had the dual been modelled by $\mathrm{Ad}_T^{\top}$ or $\mathrm{Ad}_T^{-1}$ alone — each a plausible-looking choice — it would be an *anti*-homomorphism, C3 would fail, and C2 would fail with it while C1 still passed. C3 is the dual counterpart of O1.

C4–C7 then ask the structural question this experiment was extended to answer: is the raise-process-lower detour *necessary*, or does $\mathfrak{se}(3)^{*}$ carry a nonlinearity of its own?

**Setup.** Input $W_0\sim\mathcal N(0,1)^{2\times8\times6\times1}$, no point cloud involved. Model C = $Q^{-1}$ (sharp) $\to$ the same 3-block twist backbone $\to$ head B. For the cascade, $T_1$ is drawn once at $\lVert p_1\rVert\sim1$ and $T_2$ is swept over the translation scales. C4–C7 compare four $Q$-free variants against it, all on the same input: linear-only, twist bracket applied to wrenches, the native covector bracket (4.4), and a direct enumeration of the equivariant bilinear space on the dual (method in §6).

| Check | $p=0$ | $\sim1$ | $\sim10^2$ | $\sim10^4$ |
|---|---|---|---|---|
| C1 one-step: $K(\mathrm{Ad}_T^{-\top}W)=\mathrm{Ad}_T^{-\top}K(W)\mathrm{Ad}_T^{-1}$ | 5.0e-16 | 6.0e-16 | 5.3e-16 | 5.3e-16 |
| C2 cascade: $K_2=\mathrm{Ad}_{T_2}^{-\top}K_1\mathrm{Ad}_{T_2}^{-1}$ | 1.3e-15 | 9.9e-16 | 1.0e-15 | 1.3e-15 |
| C3 $\mathrm{Ad}_{T_2T_1}^{-\top}=\mathrm{Ad}_{T_2}^{-\top}\mathrm{Ad}_{T_1}^{-\top}$ | 5.1e-16 | 3.0e-16 | 6.1e-16 | 3.7e-16 |

Verdict: **the cascade property is exact.** C1/C2 show no error growth with $\lVert p\rVert$ at all, unlike A/B — the covector pipeline never multiplies features by $\hat p$-bearing matrices internally (the two $Q$-conjugations are permutations), so the round-off amplification seen in the encoders is absent.

**Dropping $Q$ entirely (§4.4).** Can the covector space host the nonlinearity itself?

| Check | $p=0$ | $\sim1$ | $\sim10^2$ | $\sim10^4$ |
|---|---|---|---|---|
| C4b `LNLinear` only, on covectors, no $Q$ | 4.0e-16 | 4.3e-16 | 5.1e-16 | 5.0e-16 |
| **C4 twist bracket applied to covectors, no $Q$** | 5.4e-16 | **4.2e-01** | **1.0e+00** | **1.0e+00** |
| C5 native covector bracket $[\cdot,\cdot]_{*}$, $Q$-free end to end | 4.7e-16 | 4.2e-16 | 5.9e-16 | 7.8e-16 |
| C6 $[\cdot,\cdot]_{*}=Q\,[\,Q^{-1}\cdot,\,Q^{-1}\cdot\,]$ | **0.0** (bit-identical) | | | |
| C7 dimension of the equivariant bilinear space on $\mathfrak{se}(3)^{*}$ | **2** (same as on $\mathfrak{se}(3)$) | | | |

Three things are separated here.

1. **The linear layer is innocent (C4b).** `LNLinear` is right multiplication by a channel matrix, which commutes with *any* left representation, so it is equivariant on $\mathfrak{se}(3)^{*}$ with no modification. The problem was never the linear part.
2. **The bracket is the culprit, and it fails exactly as predicted (C4).** $[\cdot,\cdot]$ is equivariant for $\mathrm{Ad}$, not for $\mathrm{Ad}^{-\top}$; applying it to wrench coordinates passes at $p=0$ (5.4e-16, because $\mathrm{Ad}_{(R,0)}$ is orthogonal so the two representations coincide) and fails at $O(1)$ the moment translation enters. This is the same signature as L5/B4/B5, and it is the direct confirmation of the conjecture that motivated this experiment.
3. **But the covector space is not barren (C5–C7).** The transported bracket (4.4) is a genuine closed-form nonlinearity on $\mathfrak{se}(3)^{*}$, and a pipeline built from `LNLinear` $+\ [\cdot,\cdot]_{*}$ $+$ Gram satisfies the congruence law with **no $Q$ constructed anywhere** — errors flat at $\sim5\text{e-}16$ across all translation scales. C6 confirms it is bit-identical to the $Q$-sandwich, and C7 confirms by direct enumeration that the equivariant bilinear space on the dual has dimension 2, exactly matching $\mathfrak{se}(3)$ — so (4.4) and $N_{*}$ exhaust the options and nothing new was missed.

The honest summary: **"$Q$-free" is a statement about the implementation, not about the mathematics.** $Q$ is what makes $\mathfrak{se}(3)\cong\mathfrak{se}(3)^{*}$ as representations, and it is that isomorphism — not any independent structure on the dual — that supplies the covector nonlinearity. On a Lie algebra with no invariant non-degenerate form there would be no equivariant bilinear operation on the dual at all, and the conjecture behind this experiment would hold without qualification.

### 7.6 Reading the error growth

Positive tests grow from $10^{-16}$ to $10^{-13\ldots-12}$ as $\lVert p\rVert$ goes $0\to10^4$ — four orders of translation, roughly four orders of error: linear round-off accumulation through $\hat p$-dependent matrix products, not a structural violation. Structurally broken systems (L5, B4, B5) sit at $O(1)$ *independently* of $\lVert p\rVert$. The two regimes are separated by ten orders of magnitude; there is no ambiguity in the verdicts.

---

## 8. Discussion

**Where the Klein form is — and is not — needed.** The bracket-only program of `docs/se3_equivariant_pointcloud.md` holds the Klein form back from the *backbone* (and must: lifted lines have zero pitch, so $K(\xi,\xi)=0$ makes Klein self-normalization a division by zero). This experiment shows the compliance-type output (A) keeps the entire pipeline Klein-free, while the stiffness-type output (B) needs $Q$ exactly once, as a constant head intertwiner — a role in which degeneracy of inputs is irrelevant since $Q$ is never inverted against a feature. The two uses are orthogonal to the concerns in either source document, and consistent with both.

Consequently, recommendation #1 of `docs/se3_equivariant_pointcloud.md` §6.6 ("implement `killingform_se3`; it unlocks normalization and gating") should be read as **superseded on the gating/normalization point** (§5): that route does not exist. The function itself remains the correct implementation of the Killing form and the two bug fixes stand; only the recommended *use* changes.

**What experiment C actually settles.** In its one-step and cascade form (C1–C3) it is a pure representation-theory statement, and with the raise/lower pair it inherits all of A/B's machinery. The sharper question is C4–C7: *does the covector space carry its own nonlinearity, or is the detour through $\mathfrak{se}(3)$ forced?* The answer is a genuine both-ways result. The Lie bracket **cannot** be used on covectors as-is — it is a map $\mathfrak{g}\times\mathfrak{g}\to\mathfrak{g}$, equivariant for $\mathrm{Ad}$ and not $\mathrm{Ad}^{-\top}$, and C4 measures the cost at $O(1)$ once $p\neq0$ (with the usual $p=0$ blind spot). But the dual is not barren: $[\cdot,\cdot]_{*}$ of (4.4) is a closed-form equivariant bilinear operation on $\mathfrak{se}(3)^{*}$, and C5 runs an entire congruence-equivariant pipeline without constructing $Q$ once. C7 shows by enumeration that the dual's equivariant bilinear space has dimension 2 — the same as $\mathfrak{se}(3)$'s — so nothing beyond $[\cdot,\cdot]_{*}$ and $N_{*}$ is hiding there.

The structural reading matters more than the implementation win. Everything on the dual is available *because* the Klein form makes $\mathfrak{se}(3)\cong\mathfrak{se}(3)^{*}$ as representations; $Q$ is not eliminated by (4.4), it is compiled into it. So the intuition behind the conjecture — "brackets do not live on covector spaces" — is correct in general and fails here only because $\mathfrak{se}(3)$ is one of the algebras where $\mathfrak{g}$ and $\mathfrak{g}^{*}$ happen to be isomorphic. Practically this means a covector-input model is free to work in whichever space is convenient, provided the bracket formula matches the space it is applied in.

**Still open.** The nontrivial extension is *mixed* input (point cloud **and** covectors) where twist features and wrench features must interact. There the natural pairing is $F^{\top}\xi$ — the invariant scalar of wrench–twist duality, i.e. virtual power — which would supply exactly the data-dependent gating the linear+bracket backbone currently lacks, and unlike the Klein self-form it does not vanish on Plücker-lifted lines.

**Limitations.** (i) All results are structural (random weights); expressivity/trainability is untested — in particular the bracket-only backbone's $v\!\to\!\omega$ blockage and pitch-blindness (§5–6 of the companion analysis) are unaffected by these results. (ii) The Plücker encoder's channel-by-neighbor-rank scheme assumes no distance ties. (iii) $K$'s minimum eigenvalue, while nonnegative, is not bounded away from zero; training-time conditioning may require the *equivariant* regularizer $K+\varepsilon M(P)$ noted in `exp.md` §3.4, never $\varepsilon I$ (B5).

---

## Appendix A. Reproduction

```bash
conda activate lieneurons
python experiment/pc_se3_congruence/verify.py                    # ~1.2 s, writes results.json
python experiment/pc_se3_congruence/check_killing_degeneracy.py  # backs section 5
python docs/scripts/check_layers.py                              # regression check of core layers
```

`verify.py` installs a tripwire on `core.lie_alg_util.killingform` and asserts at the end that the count is zero, so the claim "experiments A–C use no $\mathfrak{se}(3)$ Killing form" is re-checked on every run rather than asserted in prose.

Files:

| File | Contents |
|---|---|
| `experiment/pc_se3_congruence/se3_utils.py` | $\mathrm{Ad}$, $\mathrm{Ad}^{-1}$ (closed form), $Q$, samplers, metric |
| `experiment/pc_se3_congruence/encoders.py` | Plücker encoder, minimal VN stack, learnable lift (3 modes) |
| `experiment/pc_se3_congruence/models.py` | backbone (LNLinear+LNLieBracket), heads A/B, model C, covector algebra ($[\cdot,\cdot]_{*}$, `CovectorBackbone`, `ModelCNative`), negative controls |
| `experiment/pc_se3_congruence/verify.py` | experiments 0, L, A, B, C (incl. C4–C7 covector study); Killing-form tripwire; writes `results.json` |
| `experiment/pc_se3_congruence/check_killing_degeneracy.py` | §5: rank/radical, no-invariant-inner-product enumeration, `LNKillingRelu` degeneracies, normalizer blow-up |
| `core/lie_alg_util.py` | **modified**: `killingform_se3` + dispatch branch, with a docstring warning against gating/normalization use |
| `core/lie_neurons_layers.py` | **modified**: two `algebra_type` propagation bugs fixed |

## Appendix B. Relation to the Source Documents

- `docs/se3_equivariant_pointcloud.md`: supplies the lifting (§2), the classification results motivating "linear + bracket only", and TODO items #1/#3/#4 — all three completed here (§5).
- `exp.md` / `pc_to_se3_mapping_en.pdf`: supplies the anchor-transport construction, the negative-control methodology, and the translation-sweep protocol, all reused here; experiment B reproduces its congruence result with a *different backbone* (Lie Neurons linear+bracket instead of LinearMix/QGramGate/BracketLayer — in particular, with no Klein-form gating anywhere before the head).

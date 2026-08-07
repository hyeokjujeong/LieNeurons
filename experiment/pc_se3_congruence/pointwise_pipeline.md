# Pointwise Wrench Set-Aggregation + Late Second-Moment Stiffness 실험 보고서

**실험일:** 2026-08-07  
**W&B project:** `adjoint_equivariant_network/pc-se3-congruence`  
**정밀도/장치:** `torch.float64`, CPU 및 CUDA  
**상태:** 구조 검증 + **full GPU 학습 완료**. §8.3–§8.5(Phase 1–3)를
`run_pointwise_gpu_experiments.sh`로 실행했고, §5.5가 그 결과다 (18 run, 약 72분).
남은 것은 정확도 ablation(§8.6)과 seed 분산(§7.2)이다.  
**목적:** neighbor 축을 backbone 이전에 집합 축약으로 없애고 point 축을 second moment
직전까지 유지하면, 대칭 물체의 rank collapse와 등거리 tie의 permutation 불변성 붕괴가
동시에 해결되는지 검증한다.

> **문서 규약.** 본문(§0–§10)은 **기본 설정 하나만** 서술한다. 구현이 지원하는 대체
> 경로는 부록 B, 설계 과정에서 제기된 질문과 그 답은 부록 A, 긴 유도는 부록 C에 있다.

---

## 0. 결론

> **이웃 축약을 rank가 아니라 거리로 라벨링하고, point 축을 Gram까지 유지하면,
> 선행 두 실패(대칭에서의 rank collapse, 등거리 tie에서의 permutation 붕괴)가
> 동시에 사라진다. 그리고 이 구성에서 encoder는 학습 파라미터가 하나도 필요 없다.**

미학습 모델의 구조 검증(Phase 1)과 full recipe 학습(Phase 2–3, 150 epoch · 4096 샘플 ·
CUDA · 9 케이스 × 2 타깃)에서 다음을 얻었다.

- centro($\eta=0$)·$C_2$·tetra·iid 전부에서 `rank_pred = 6`, $\lambda_{\min}\sim10^{-3}$
- equivariance 오차 $\le5.8\times10^{-15}$, point permutation 오차 $\le1.1\times10^{-15}$
- 15개 아키텍처 변형 전부 동일한 구조 지표
- 정육면체 격자(정확한 tie 다수)에서 permutation 오차 $3.01\times10^{-16}$.
  같은 격자에서 선행 rank-channel 모델은 $1.92\times10^{-1}$이고, jitter $10^{-8}$로
  tie를 깨면 즉시 회복한다
- **realizability(teacher 타깃): 9개 케이스 전부 val $d\le1.5\times10^{-3}$**,
  ff/mm 블록 상대오차 $\le5.5\times10^{-4}$, train과 val이 사실상 같다.
  smoke(0.012) 대비 약 8–10배 개선이며 대칭 데이터셋과 iid 사이에 격차가 없다
- 학습 18 run 전부에서 `rank_pred = 6`, `clamped = 0`,
  `equiv_err_final` $\le8.9\times10^{-15}$ — gradient step이 구조를 훼손하지 않는다
- analytic(`kernel`) 타깃은 val $d$ 0.297–1.718에서 **평탄화**하며 train과 val이 붙어
  있다. 과적합도 학습 부족도 아닌 **모델 클래스 불일치의 바닥**이고, §7.1이 예고한
  값이다. 절대값으로 표현력을 판단하지 않는다
- encoder 학습 파라미터 **0개** (전체 18,697개)

따라서 선행 실패의 직접 원인은 다음 두 설계 선택이었다는 해석과 일치한다.

1. edge wrench를 point 축에 대해 먼저 평균하여 permutation gauge를 없앤 것
2. **이웃 rank를 채널 index로 사용한 것** — compact kernel이나 adaptive radius가
   아니다 (§5.3에서 분리 확인)

단, 학습 수치는 여전히 **단일 seed**이고 seed 간 분산을 측정하지 않았다. 그리고
analytic 타깃은 raw wrench의 이차식이고 본 모델은 latent covector의 second moment이므로
**matched target이 아니다** (§7.1). 정확도 ablation도 아직 하지 않았다 (§7.3).

---

## 1. 기존 문제

### 1.1 Global vector mean에 의한 rank collapse

선행 Plücker encoder는 edge wrench를 point 축에 대해 먼저 평균했다.

$$
\mu_c=\frac1N\sum_i w_{i,c}\in\mathfrak{se}(3)^*.
$$

Cloud가 proper symmetry group $H\subset SE(3)$를 가지면, 임의의 등변 global vector는
고정 부분공간

$$
\operatorname{Fix}_H(A)=\{x\in\mathfrak{se}(3)^*:\ A_hx=x\ \ \forall h\in H\}
$$

안에 갇힌다. Centro-symmetric cloud에서는 $f$-채널이 항등적으로 0이 되어
$\operatorname{rank}K\le3$, $C_2$ cloud에서는 $\operatorname{rank}K_{ff}\le1$이다.
이는 표현론적 제약이므로 채널 수를 늘리거나 lift를 learnable하게 바꿔도 복구되지 않는다.

### 1.2 Rank-channel과 LN backbone의 tie 붕괴

후속 구성 `WrenchEdgeEncoder(graph='kernel')`는 point 축을 살렸지만 **채널 $c$를
$c$번째 최근접 이웃**으로 정의했다. Second moment만 취하면 채널 합이 순서에 무관해
문제가 없으나, LN backbone이 채널을 섞는 순간 등거리 이웃의 rank 교환에 대해 불변하지
않게 된다. 이 실패는 `models.py`의 주석에 이미 명시되어 있었으나 정량화되지 않았다.

정확한 tie는 예외가 아니라 규칙이다. 격자, 정다면체 orbit 등 대칭 물체에서는 "몇 번째
최근접 이웃"이라는 라벨이 애초에 정의되지 않는다.

### 1.3 가설

> Neighbor 축은 **집합**으로 취급해 backbone 이전에 없애고, point 축은 Gram까지
> 유지한다. 1.1이 사라지는 이유는 point 축이 permutation gauge를 보존하기 때문이고,
> 1.2가 사라지는 이유는 채널 라벨이 더 이상 rank가 아니기 때문이다.

---

## 2. 제안한 pointwise 방법

$$
P
\xrightarrow{\text{tie-safe graph + Plücker lift}} W^{\mathrm{edge}}
\xrightarrow{\text{집합 축약}} X^{(0)}
\xrightarrow{\text{LN block}\times3} X^{(3)}
\xrightarrow{\text{factor head}} (L,\ K)
$$

| 단계 | 연산 | 출력 shape ($B{=}64,N{=}128,k{=}64$) | 파라미터 |
|---|---|---|---:|
| 그래프 | 반경 $r_i$, 창 $\phi(q_{ij})$ | $[64,128,64]$ | 0 |
| lift | $w_{ij}=[\,p_j-p_i\,;\,p_i\times(p_j-p_i)\,]$ | $[64,128,64,6]$ | 0 |
| 집합 축약 | 고정 거리 shell 가중 평균 | $[64,8,6,128]$ | **0** |
| block $\times3$ | LN-Linear + covector bracket + Klein gate | $[64,16,6,128]$ | 15,376 |
| head | factor 사영 + $\beta$ + 전역 scale | $[64,6,1024]$, $[64,6,6]$ | 3,321 |

$k=64$ 축은 집합 축약에서 사라지고, $N=128$ 축은 마지막 Gram까지 살아남는다.

### 2.1 표기와 축 규약

Neighbor 수와 stiffness가 같은 글자를 쓰지 않도록 고정한다.

| 기호 | 의미 | 기본값 | weight sharing 규칙 |
|---|---|---|---|
| $B$ | batch | — | 동일 모델 |
| $N$ | point 수 | 128 | 모든 point가 동일 파라미터 공유. point-index별 파라미터 금지 |
| $k$ | point당 이웃 후보 수 | 64 | **unordered set.** rank별 파라미터 금지 |
| $C_\ell$ | layer $\ell$ 채널 수 | $8\to16\to32\to16$ | channel별 weight 상이 허용, 자유로운 mixing 허용 |
| $H$ | factor 채널 수 | 8 | factor별 파라미터 허용, 모든 point에서 공유 |
| $6$ | coadjoint 표현 축 | — | **LN-등변 연산만 허용.** 임의의 $6\times6$ mixing 금지 |
| $K$ | $6\times6$ stiffness | — | — |

세 번째와 여섯 번째 줄이 이 설계의 모든 제약을 만든다. $k$축에는 집합 연산만,
$6$축에는 등변 연산만 허용된다. 학습 가능한 자유도는 그 사이(채널 축)와 불변 스칼라
위에만 존재한다 (§2.7).

Tensor layout은 다음과 같다.

$$
P\in\mathbb R^{B\times N\times3},\quad
W^{\mathrm{edge}}\in\mathbb R^{B\times N\times k\times6},\quad
X^{(\ell)}\in\mathbb R^{B\times C_\ell\times6\times N},
$$
$$
Z\in\mathbb R^{B\times H\times6\times N},\qquad
K\in\mathbb R^{B\times6\times6}.
$$

`[B, C, 6, N]`은 repo의 `LNLinear`, `covector_bracket`, `klein_gram`이 이미 소비하는
layout이고 이 layer들은 trailing 축에 대해 pointwise다. 따라서 backbone을 point마다
적용하기 위해 `[BN, C, 6, 1]`로 reshape할 필요가 없다 — 기존 구현이 그대로 pointwise
backbone이 된다. 선행 계획서가 제안한 $(B,N)$ 병합은 불필요했다.

### 2.2 Tie-safe local graph

**Support 반경.** 밀도 보정된 전역 스케일을 쓴다.

$$
r_i=r(P)=\alpha\,\sigma(P)\Bigl(\frac{k_{\rm target}}{N}\Bigr)^{1/3},
\qquad
\sigma(P)=\Bigl(\tfrac1N\sum_i\|p_i-\bar p\|^2\Bigr)^{1/2},
$$

기본값 $\alpha=1.15$, $k_{\rm target}=16$. $\sigma(P)$는 RMS 반경으로
permutation-invariant, rigid-invariant이며 $P$에 대해 매끄럽다. $(k_{\rm target}/N)^{1/3}$은
3차원 밀도 보정이므로 $N$이 바뀌어도 평균 degree가 $k_{\rm target}$ 근처에 머문다
(§5.7에서 측정: $N=48/128/512$에 대해 9.6/13.5/15.3). $N$은 cloud의 상수이지
순서통계량이 아니므로 이 보정도 tie-safe하다.

**Compact window.** 정규화 거리 $q_{ij}=\|p_j-p_i\|/r_i$에 C2 Wendland window를 쓴다.

$$
\phi(q)=
\begin{cases}
(1-q)^4(1+4q), & 0\le q<1,\\
0, & q\ge1.
\end{cases}
$$

$\phi(1)=\phi'(1)=0$이므로 (유도는 부록 C.3) edge가 support를 드나들 때 값과 gradient가
모두 연속이다. 경계에 정확히 놓인 등거리 shell은 양쪽 모두 0을 기여한다.

**후보 집합과 truncation.** 실제로는 최근접 $k$개 후보만 materialize한다. 거리는
`torch.cdist(..., compute_mode='donot_use_mm_for_euclid_dist')`로 계산하는데, mm 항등식
$\|a\|^2-2a\!\cdot\!b+\|b\|^2$은 tie 처리가 의존하는 정확한 대칭 $d_{ij}=d_{ji}$를 잃기
때문이다. Support 안에 후보보다 많은 점이 들어오면 잘려나가고, 그 순간 이웃 집합이
기하만의 함수가 아니게 된다. 이것이 그래프에 남은 **유일한** tie 경로이므로

$$
\text{truncation\_frac}=\Pr_i\bigl[\,d_{i,(k)}<r_i\,\bigr]
$$

를 항상 계산해 매 epoch 로그·W&B에 남기고, $0.01$을 넘으면 프로세스당 한 번
`RuntimeWarning`을 발생시킨다. 기본 설정은 $N=48/128/512$ 전부에서 이 값이 0이다 (§5.7).

기본 경로가 그래프에서 받는 것은 $q_{ij}$와 $\phi(q_{ij})$ 둘뿐이고 후자는 전자의
함수다. 즉 **edge 하나당 스칼라 하나**다. 선택 경로가 쓰는 13차원 edge 불변량은 부록 B.2.

### 2.3 Plücker lift와 coadjoint 법칙

$$
w_{ij}=\begin{bmatrix}f_{ij}\\ m_{ij}\end{bmatrix}
=\begin{bmatrix}p_j-p_i\\ p_i\times(p_j-p_i)\end{bmatrix}\in\mathfrak{se}(3)^*.
$$

$T=(R,p)$에 대해 $f'=R(p_j-p_i)=Rf$이고

$$
m'=(Rp_i+p)\times R(p_j-p_i)
=R\bigl(p_i\times(p_j-p_i)\bigr)+p\times Rf
=Rm+p\times Rf,
$$

따라서

$$
w_{ij}(T\cdot P)=A_T\,w_{ij}(P),
\qquad
A_T=\operatorname{Ad}_T^{-\top}=\begin{bmatrix}R&0\\ \hat pR&R\end{bmatrix}.
$$

저장 순서는 $[f;m]$이며 **force slot이 translation에 blind**하다. 물리적으로 $w_{ij}$는
$p_i$를 지나 $p_j$를 향하는 직선을 따르는 순수 힘의 wrench이며, Klein 행렬 $Q$는
코드 어디에도 등장하지 않는다.

### 2.4 이웃 집합에서 채널로

이 단계가 하는 일은 하나다: **$k$축을 없애고 $C_0$축을 만든다.** 설계 원칙은

> 채널 라벨은 이웃 집합의 **연속** 함수여야 한다.

rank 라벨("$c$ = $c$번째 최근접 이웃")은 등거리 tie에서 정의되지 않는다. 거리 라벨
("$c$ = 정규화 거리 $c_c$ 근처의 이웃들")은 항상 연속이다. 후자를 가장 단순한 형태로
실현한다.

$$
X^{(0)}_{i,c}=\sum_{j\in\mathcal N(i)}a_{ij,c}\,w_{ij},
\qquad
a_{ij,c}=\frac{\phi(q_{ij})\,\rho_c(q_{ij})}{\sum_{l}\phi(q_{il})\,\rho_c(q_{il})},
$$

$$
\rho_c(q)=\exp\!\Bigl[-\Bigl(\tfrac{q-c_c}{w}\Bigr)^{2}\Bigr],
\qquad
c_c=\operatorname{linspace}(0,1,C_0),\quad w=\tfrac{1}{C_0-1},\quad C_0=8 .
$$

채널 $c$는 "정규화 거리 $c_c$ 근처 껍질에 있는 이웃 wrench들의 가중 평균"이다.
$\rho_c$는 고정이며 학습 파라미터가 아니다.

**왜 가중치를 학습하지 않는가.** 뒤따르는 LN-Linear $W\in\mathbb R^{C_0\times C_1}$와
합성하면 선형성에 의해

$$
\sum_c W_{c'c}\Bigl(\sum_j\phi(q_{ij})\rho_c(q_{ij})w_{ij}\Bigr)
=\sum_j \phi(q_{ij})
\underbrace{\Bigl(\sum_c W_{c'c}\rho_c(q_{ij})\Bigr)}_{\in\ \mathrm{span}\{\rho_c\}}
w_{ij},
$$

즉 **학습된 radial kernel이 사라진 것이 아니라 한 layer 뒤로 옮겨간 것**이다.
$\mathrm{span}\{\rho_c\}$ 안의 임의 가중 함수가 그대로 복원되므로 encoder에 가중치
MLP를 둘 이유가 없다. 실측으로도 제거했을 때 평균 성능이 나빠지지 않는다 (§5.6).

**정규화 분모만은 이 논리로 넘어가지 않는다.** 비율은 shell들의 선형결합이 아니므로
LN-Linear가 만들어낼 수 없다. 그래서 설계 규칙은 다음과 같다.

> LN 연산이 흡수할 수 있는 것은 전부 뒤로 넘기고, 흡수할 수 없는 것만 encoder에 남긴다.

정규화가 하는 일은 dense한 영역의 anchor가 단지 이웃이 많다는 이유로 더 큰 feature를
갖는 것을 막는 것이다. 측정된 효과는 tetra에서 $d$ $1.530\to1.218$ (§5.6).

**왜 encoder에 bracket을 두지 않는가.** $X^{(0)}$은 $w\to-w$에 대해 홀수 parity이므로,
짝수 parity 채널 $[X^{(0)}U,\ X^{(0)}V]_*$를 concat하는 구성을 생각할 수 있다. 두 이유로
기본 설정에서 제외한다.

- **중복이다.** Block 0이 이미 bracket을 계산한다:
  $Y^{(0)}=X^{(0)}W_0+[(X^{(0)}W_0)V_0,\ (X^{(0)}W_0)U_0]_*$.
  여기서 $W_0U_0$와 $W_0V_0$는 $C_0\to C_1$의 임의 사상이므로, encoder bracket이 만들
  수 있는 모든 채널을 block 0이 이미 포함한다. 위의 설계 규칙이 그대로 적용된다.
- **parity 동기가 성립하지 않는다.** $[X^{(0)}U,\ X^{(0)}V]_*$는 $X^{(0)}$으로 만들어지므로
  $X^{(0)}$이 사라지는 국소 대칭에서는 함께 사라진다. 짝수 parity를 실제로 살리려면
  raw edge에서 분리되지 않는 이중합을 해야 하며, 그것이 부록 B.3의 `pairwise`다.

실측으로도 제거가 손해가 아니다 — analytic 타깃 네 데이터셋 전부에서 근소하게
낫다 (§5.6). 결과적으로

$$
\underbrace{[64,128,64,6]}_{\text{edge wrench}}
\ \longrightarrow\
\underbrace{[64,8,6,128]}_{X^{(0)},\ \text{학습 파라미터 }0}
$$

이고, **encoder는 수학적 타입만 맞추는 고정 lift**가 된다.

### 2.5 Pointwise LN block

블록 $\ell$은 다음을 계산한다. 모든 $W_\ell,U_\ell,V_\ell$과 MLP는 point 간 공유되고,
data-dependent 계수 $g$만 point마다 다르다.

$$
Y_i^{(\ell)}=X_i^{(\ell)}W_\ell
+\bigl[X_i^{(\ell)}V_\ell,\ X_i^{(\ell)}U_\ell\bigr]_*,
\qquad
X_i^{(\ell+1)}=g_i^{(\ell)}\odot Y_i^{(\ell)} .
$$

**Covector bracket.** $\mathfrak{se}(3)^*$ 위의 유일한(스칼라배 제외) 등변 bilinear map:

$$
[F_1,F_2]_*=\bigl(f_1\times f_2,\ \ f_1\times m_2-f_2\times m_1\bigr).
$$

Residual 형태로 쓰고 $V_\ell$을 0으로 초기화한다. AIRM 목적함수 앞에서 무작위 이차
사상은 feature scale이 잡히기 전에 불안정하다. $U_\ell$은 무작위로 남겨 분기가 계속
gradient를 받게 하고, 한 step 뒤 $V_\ell\ne0$이 되면 $U_\ell$도 풀린다 (단위 테스트가
이 두 단계를 검사한다).

**Klein-form gate.** 채널 사영 $U=XR$, $V=XS$ ($P=8$)로 불변 스칼라를 만들어 채널을
변조한다.

$$
S_i=\operatorname{diag}\bigl(U_i^\top Q^{-1}V_i\bigr)\in\mathbb R^{P},
\qquad
\varsigma(s)=\operatorname{sign}(s)\log(1+|s|),
$$
$$
s_{\rm global}=\frac1N\sum_i\varrho\bigl(\varsigma(S_i)\bigr)\in\mathbb R^{16},
\qquad
g_{i,c}=1+\tanh\Bigl(\mathrm{MLP}\bigl(\varsigma(S_i),\,s_{\rm global}\bigr)_c\Bigr).
$$

설계상 중요한 세 가지가 있다.

1. **gate는 채널의 6성분 전체에 곱한다.** $A_T$가 block-triangular($f$가 $m$으로
   섞임)이므로 force/moment slot에 서로 다른 gate를 주면 등변성이 깨진다.
2. **불변량을 분모로 쓰지 않는다.** Klein form은 signature $(3,3)$의 indefinite
   form이라 null cone이 존재한다. $1+\tanh(\cdot)$의 bounded gate만 쓴다.
3. **global context는 스칼라의 평균**이다. 전역 등변 *벡터*를 만들지 않으므로 §1.1의
   stabilizer 제약을 되살리지 않는다. $\varsigma$는 단조 함수이므로 눌러도 여전히
   불변량이고, feature scale이 드리프트해도 MLP가 포화하지 않는다.

### 2.6 Late second-moment head

$$
Z_i=X_i^{(3)}W_{\rm head}\in\mathbb R^{6\times H},
\qquad
\beta_{i,h}=\operatorname{softplus}\Bigl(r_\theta\bigl(\varsigma(S_i),s_{\rm global}\bigr)_h\Bigr)>0,
$$

$$
K(P)=\frac{e^{g}}{NH}\sum_{i=1}^{N}\sum_{h=1}^{H}\beta_{i,h}\,z_{i,h}z_{i,h}^{\top}.
$$

정규화 $Z(P)=NH$는 선행 synthetic 타깃이 $Nk$로 나누는 것과 짝이 맞는다.

$e^g$는 학습되는 전역 스칼라 하나다. 불변량이므로 구조를 소모하지 않지만, AIRM에서
순수 scale 불일치 $s$가 $\sqrt6|\log s|$만큼 기여하므로 (부록 C.2) 이 handle이 없으면
최적화가 $\beta$ MLP의 bias만으로 scale을 맞춰야 한다. $\beta$의 마지막 layer는 weight
0, bias $\operatorname{softplus}^{-1}(1)$로 초기화해 $\beta\equiv1$에서 출발한다.

Head는 factor 행렬을 **먼저** 내고 $K$를 그것으로부터 만든다.

$$
L_\theta(P)=\Bigl[\sqrt{\tfrac{e^g\beta_{i,h}}{NH}}\;z_{i,h}\Bigr]_{(i,h)}
\in\mathbb R^{6\times NH},
\qquad
K=L_\theta L_\theta^{\top}.
$$

반대 방향(등변 $K$의 Cholesky)은 basis 순서에 의존하므로 등변이 아니다.
`model.factors(P)`가 $L$을 돌려준다.

### 2.7 파라미터 배치

권장 스케줄 $C_0=8\to16\to32\to16$, $H=8$ 기준 총 18,697개.

| 모듈 | 파라미터 | 구성 |
|---|---:|---|
| `set_encoder` | **0** | 고정 거리 shell 8개 — 학습 파라미터 없음 |
| `blocks.0` ($8\to16$) | 4,096 | LN-Linear 128 + bracket 512 + gate 3,456 |
| `blocks.1` ($16\to32$) | 6,800 | LN-Linear 512 + bracket 2,048 + gate 4,240 |
| `blocks.2` ($32\to16$) | 4,480 | LN-Linear 512 + bracket 512 + gate 3,456 |
| `head` | 3,321 | factor 사영 128 + 불변 사영 256 + $\beta$ MLP 2,936 + $g$ 1 |
| **합계** | **18,697** | |

역할별로 나누면 이 배치가 §2.1 축 규약의 직접적 귀결임이 드러난다.

| 역할 | 위치 | 파라미터 | 비중 |
|---|---|---:|---|
| 불변 스칼라 → 스칼라 MLP | block 3개의 gate, head의 $\beta$·context MLP | 13,064 | 70% |
| 6-벡터에 작용하는 등변 선형 | `LNLinear`, bracket 방향 $U_\ell,V_\ell$, gate/head 채널 사영 | 5,632 | 30% |
| 전역 스칼라 | head의 $g$ | 1 | — |
| **encoder 전체** | — (고정 shell lift) | **0** | — |

$6$축에는 등변 연산만 허용되므로 **MLP는 6-벡터에 직접 작용할 수 없다.** 따라서 MLP는
"불변 스칼라를 받아 스칼라를 내는" 자리에만 놓일 수 있고, 이 모델에서 그 자리는 gate
$g_{i,c}$와 factor weight $\beta_{i,h}$ 둘뿐이다. Encoder에 있던 두 후보 — edge 가중치
$a_{ij,c}$와 bracket 사영 — 는 뒤따르는 LN 연산이 흡수할 수 있어 전부 제거되었다.
gate와 $\beta$는 곱셈이 채널별로 다른 비선형 위치에 있어 뒤로 넘길 수 없다.

학습 자유도는 세 종류다: ① 등변 채널 혼합, ② 불변 스칼라 gate·weight, ③ 전역 스칼라 하나.

### 2.8 Equivariance, permutation gauge, PSD

**단계별 근거.** 각 항목은 §5.1의 단위 테스트에 대응한다.

| 단계 | 근거 |
|---|---|
| 그래프 | $r(P)$, $q_{ij}$, $\phi$ 전부 거리만의 함수 → rigid-invariant, relabeling-covariant |
| lift | §2.3 |
| 집합 축약 | 불변 스칼라 $a_{ij,c}$ $\times$ coadjoint 벡터의 **집합 합** |
| LN-Linear | 채널 축만 혼합: $A_T(XW)=(A_TX)W$ |
| covector bracket | $\mathfrak{se}(3)^*$ 위 유일한 등변 bilinear map |
| Klein gate | 불변 스칼라 $\times$ **6성분 전체** |
| head | $\sum\beta zz^\top$, $\beta$ 불변 |

Klein pairing $\langle a,b\rangle=f_a\!\cdot\!m_b+m_a\!\cdot\!f_b$가 불변인 것은 $A_T$를
대입했을 때 교차항이

$$
Rf_a\!\cdot\!(p\times Rf_b)+(p\times Rf_a)\!\cdot\!Rf_b
=\det[Rf_a,p,Rf_b]+\det[p,Rf_a,Rf_b]=0
$$

으로 상쇄되기 때문이다. force pairing $f_a\!\cdot\!f_b$도 $f\mapsto Rf$이므로 불변이며,
부록 B.4에서 옵션으로 사용한다.

**Permutation gauge — 이 설계의 핵심.** Factor 행렬 $L_\theta(P)\in\mathbb R^{6\times NH}$에
대해, 강체변환 $T$가 점 순열 $\pi_T$를 유도할 때 각 factor가
$z_{\pi_T(i),h}(T\cdot P)=A_Tz_{i,h}(P)$, $\beta_{\pi_T(i),h}(T\cdot P)=\beta_{i,h}(P)$를
만족하므로

$$
L_\theta(T\cdot P)=A_T\,L_\theta(P)\,\bigl(\Pi_T\otimes I_H\bigr).
$$

$\Pi_T\otimes I_H$가 orthogonal이므로 Gram에서 소거되어

$$
K(T\cdot P)
=A_TL_\theta(\Pi_T\otimes I_H)(\Pi_T\otimes I_H)^\top L_\theta^\top A_T^\top
=A_T\,K(P)\,A_T^\top .
$$

Global vector pooling을 먼저 하면 factor 축이 $H$ 하나로 줄어 gauge $\Pi_T$가 사라지고,
각 벡터가 stabilizer에 의해 **직접 고정**되어야 한다 — 그것이 §1.1이다. Point index를
유지하면 대칭은 factor를 고정하는 대신 **서로 permutation**시킬 수 있다.

> late second moment의 핵심은 "2차"라는 점보다, point-factor gauge를 Gram까지 유지한다는 점이다.

**Tie-safety.** Anchor $i$의 이웃 목록을 재배열하는 $\Pi$에 대해, $X^{(0)}_i=A_i^\top W_i$로
쓰면 $A_i\in\mathbb R^{k\times C_0}$의 원소 $(A_i)_{jc}=a_{ij,c}$는 $q_{ij}$의 함수다.
재배열은 $W_i\mapsto\Pi W_i$와 **동시에** $A_i\mapsto\Pi A_i$를 유발하므로

$$
(\Pi A_i)^\top(\Pi W_i)=A_i^\top\Pi^\top\Pi W_i=A_i^\top W_i .
$$

학습 가능한 **고정** 행렬 $M\in\mathbb R^{k\times C_0}$을 썼다면 $M$은 $\Pi$를 따라가지
않으므로 이 논증이 성립하지 않는다. 그 $j$가 곧 rank이며, 그것이 §1.2다. 두 이웃이
정확히 등거리이면 $q_{ij}=q_{il}$이므로 $A_i$의 두 행이 동일하고, 따라서 tie가 있어도
위 식이 성립한다.

**PSD와 해석.**

$$
\xi^\top K\xi=\frac{e^g}{NH}\sum_{i,h}\beta_{i,h}\,(z_{i,h}^\top\xi)^2\ \ge\ 0 .
$$

Analysis map $\mathcal A_{\theta,P}(\xi)=L_\theta^\top\xi$를 두면 $K=\mathcal A^*\mathcal A$이고,
가상 측정 $y_{i,h}=z_{i,h}^\top\xi+\varepsilon_{i,h}$,
$\varepsilon_{i,h}\sim\mathcal N(0,\beta_{i,h}^{-1})$의 Fisher information과 같다.
Plücker lift는 기본 국소 virtual-work 측정, LN-Linear는 측정 채널의 혼합, covector
bracket은 새 등변 측정 방향의 생성, Klein gate는 geometry-dependent 측정 정밀도, late
second moment는 그 모든 측정의 Fisher information 합성이며 $K^{-1}$은 국소 twist 추정
공분산, 즉 compliance다.

### 2.9 계산 복잡도

float64, $B$=batch, $N$=points, $k$=candidates, $C$=channels.

| 항목 | 크기 | $B{=}64,N{=}128,k{=}64,C{=}32$ |
|---|---|---|
| 거리 행렬 $[B,N,N]$ | $8BN^2$ | 8.4 MB |
| edge wrench $[B,N,k,6]$ | $48BNk$ | 25 MB |
| backbone activation $[B,C,6,N]$ (layer당) | $48BCN$ | 12.6 MB |

$N=512$면 activation은 layer당 50 MB, 거리 행렬은 134 MB가 된다. 지배적 비용은
$O(N^2)$ 거리 행렬이며 $N\gtrsim10^3$에서는 spatial hash로 교체해야 한다.

edge 텐서는 encoder 안에서만 존재한다. $k$축을 backbone까지 끌고 가면 activation에
$\times k$가 붙어 $N=512,k=32$에서 layer당 1.5 GiB가 된다. 이것이 §2.4의 축약을
backbone 이전에 두는 실용적 이유다.

---

## 3. 구현

| 파일 | 변경 내용 |
|---|---|
| `pointwise_graph.py` | 신규. tie-safe graph, Wendland window, edge 불변량, truncation 진단·경고 |
| `pointwise_models.py` | 신규. 집합 encoder / message passing / Klein gate / block / late Gram head / 전체 모델 |
| `verify_pointwise.py` | 신규. 구조 검증 A–F + 변형 표 |
| `run_pointwise_suite.py` | 신규. verify / teacher / analytic / ablation phase runner |
| `blockage_bench.py` | `--encoder pointwise`, `--target-graph teacher`, `--pw-*` CLI, graph 지표 로깅 |
| `data_synth.py` | `lattice_clouds` (exact-tie 벤치마크) 추가 |
| `test/test_pc_pointwise_pipeline.py` | 신규. 구조 테스트 35종 |

재사용한 repo 자산은 `core.lie_neurons_layers.LNLinear`,
`models.{covector_bracket, klein_gram}`, `train.affine_invariant_d`,
`metrics.{block_metrics, f_signal, init_wandb}`이다. 데이터셋·타깃·W&B·진단은
`blockage_bench.py`를 통해 선행 실험과 동일하게 쓴다.

---

## 4. 실험 설정

### 4.1 전체 object suite

| 이름 | 정의 | 검사하는 실패 모드 |
|---|---|---|
| `iid` | 이방성 Gaussian blob, 임의 SE(3) pose | 대칭 없음 (기준선) |
| `centro` | $P=\{\pm a_i\}+\eta\cdot$noise | §1.1 — $\eta=0$에서 $\operatorname{rank}\le3$ |
| `c2` | 단일 $C_2$ 축 대칭 | §1.1 — $\operatorname{rank}K_{ff}\le1$ |
| `tetra` | $A_4$ orbit (12개, $-I$ 없음) | §1.1 + 구조적 거리 tie |
| `lattice` | $n^3$ 정육면체 격자 (본 실험 신규) | **§1.2** — 대량의 정확한 tie |
| `fiber` | 같은 $f$-요약, 다른 타깃의 cloud 쌍 | 선행 진단의 회귀 가드 |

`lattice_clouds(jitter=·)`는 tie를 연속적으로 깨서 near-tie 곡선을 그릴 수 있다.

### 4.2 타깃

| `--target-graph` | 정의 | 역할 |
|---|---|---|
| `teacher` | 같은 클래스의 고정 난수 pointwise 모델 | **realizability.** 타깃이 모델 클래스 안에 있음이 보장되므로 잔차를 표현력 부족으로 오해할 수 없다 |
| `kernel` | `contact_spring_kernel_K` — compact-kernel edge second moment | 선행 실험과 동일한 analytic 타깃 |
| `knn`/`all` | hard-kNN / all-pairs contact spring | 선행 비교군 |

읽는 순서가 중요하다. analytic 타깃은 **raw** wrench의 edge-level second moment이고 본
모델은 **latent** covector의 second moment다. 두 함수족이 일치하지 않으므로 `kernel`
타깃의 잔차 하한에는 타깃 불일치가 섞여 있다. 반드시 teacher를 먼저 읽는다 (§7.1).

### 4.3 손실

Affine-invariant Riemannian metric:

$$
d(K_{gt},K)
=\bigl\|\log\bigl(K_{gt}^{-1/2}KK_{gt}^{-1/2}\bigr)\bigr\|_F
=\Bigl(\sum_i\log^2\lambda_i(A)\Bigr)^{1/2},
\quad A=L_{gt}^{-1}KL_{gt}^{-\top},
$$

$K_{gt}=L_{gt}L_{gt}^\top$ (Cholesky). 고윳값만 쓰므로 autograd가 $1/(\lambda_i-\lambda_j)$
고유벡터 항을 타지 않는다. 이 거리는 congruence-invariant이므로 등변 모델과 합성하면
translation-consistent 목적함수가 된다.

### 4.4 공통 hyperparameter

| 항목 | smoke (§5.6) | full recipe (§5.5, §8) |
|---|---|---|
| epoch / batch / lr | 60 / 32 / $10^{-3}$ cosine | 150 / 64 / $10^{-3}$ cosine ($\eta_{\min}=10^{-5}$) |
| $n_{\rm train}$ / $n_{\rm val}$ | 256 / 64 | 4096 / 512 |
| $N$ | 48 | 128 (iid는 32/128/512, tetra는 120) |
| radius mode | `global_scale`, $\alpha=0.75$ | `density_scaled`, $\alpha=1.15$, $k_{\rm target}=16$ |
| `candidate_k` | 24 | 64 |
| 채널 / $H$ | $8\to16\to32\to16$ / 8 | 동일 |
| 파라미터 | 18,697 (encoder 0) | 동일 |
| seed (model/data/teacher) | 0 / 100 / 7 | 동일 |
| 장치 | CPU | CUDA |

### 4.5 지표

매 epoch stdout과 W&B에 기록한다.

| 지표 | 의미 | 경보 조건 |
|---|---|---|
| `train_d` / `val_d` | AIRM 거리 | — |
| `graph_truncation_frac` | support 안 이웃이 top-$k$에서 잘림 | $>0$ → 설정 수정 후 재실행 |
| `graph_mean_degree` / `graph_max_degree` | 실제 이웃 수 | degree $<3$이면 국소 정보 부족 |
| `rank_pred` | $K$의 수치적 rank | $<6$ → rank collapse |
| `lam_min` | 최소 고윳값 중앙값 | $10^{-10}$ 근처면 가짜 full rank |
| `clamped` | AIRM 고윳값 guard 발동 수 | $>0$ → 조건수 문제 |
| `err_rel_ff`/`fm`/`mm` | 블록별 상대오차 | ff만 크면 force 계보 소실 |
| `f_signal` | encoder force 채널 norm | 0으로 붕괴하면 §1.1 재발 |
| `equiv_err_init/final` | 학습 전후 등변성 | $>10^{-10}$ → 구조 훼손 |

---

## 5. 결과

> §5.5는 §8.3–§8.5를 그대로 실행한 **full GPU 결과**(150 epoch · 4096 샘플 · CUDA)이고,
> §5.6은 그보다 앞선 CPU smoke의 encoder 비교다. 두 표는 설정이 다르므로 절대값을
> 섞어 읽지 않는다. 두 경우 모두 **단일 seed**다.

### 5.1 단위 테스트

`pytest -q test/test_pc_pointwise_pipeline.py` → **35 passed**. 검사 항목:

- Wendland 창이 $q=1$에서 값과 gradient 모두 0인지
- 그래프 support가 rigid 변환에 불변이고, 후보를 의도적으로 좁히면 truncation이
  실제로 검출되는지
- Edge wrench의 coadjoint 법칙 (§2.3)
- Klein pairing과 force pairing의 불변성 (§2.8)
- **21개 구성 변형 각각에 대해** congruence 등변성 + point permutation 불변성 + 대칭성
- $K=LL^\top$ 정확 일치, PSD
- centro·$C_2$·tetra에서 rank 보존
- 격자에서 정확 tie 불변성, 그리고 **rank-channel 모델이 같은 격자에서 실패한다는
  회귀 가드** (이 테스트가 통과해 버리면 §5.3의 비교가 의미를 잃는다)
- 기본 pool이 encoder 파라미터를 갖지 않는다는 것, shell 가중치가 support 밖에서
  정확히 0이고 안에서 양수 합을 갖는다는 것
- zero-init 분기가 한 step 뒤 gradient를 받는지

### 5.2 구조 검증 — 대칭에서의 rank 보존

미학습 모델, seed 0, 4 샘플. 상단은 소형 검증 설정, 하단은 §8 권장 설정이다.

$N=48$, `candidate_k=24`, `global_scale` $\alpha=0.75$ (CPU):

| 데이터셋 | equivariance | permutation | rank | $\lambda_{\min}$ | mean deg | trunc |
|---|---|---|---|---|---|---|
| iid | 1.07e-15 | 4.76e-16 | 6.00 | 9.62e-04 | 9.6 | 0.016 |
| centro $\eta=0$ | 3.79e-15 | 3.15e-16 | 6.00 | 2.60e-03 | 9.2 | 0.000 |
| c2 $\eta=0$ | 1.26e-15 | 4.36e-16 | 6.00 | 3.61e-03 | 7.4 | 0.000 |
| tetra $\eta=0$ | 1.06e-15 | 4.15e-16 | 6.00 | 3.74e-03 | 6.1 | 0.000 |

$N=128$, `candidate_k=64`, `density_scaled` $\alpha=1.15$, $k_{\rm target}=16$ (CUDA):

| 데이터셋 | equivariance | permutation | rank | $\lambda_{\min}$ | mean deg | trunc |
|---|---|---|---|---|---|---|
| iid | 1.40e-15 | 3.78e-16 | 6.00 | 8.23e-04 | 11.5 | **0.000** |
| centro | 1.42e-15 | 5.98e-16 | 6.00 | 1.55e-03 | 13.0 | **0.000** |
| c2 | 1.60e-15 | 5.27e-16 | 6.00 | 2.02e-03 | 12.9 | **0.000** |
| tetra | 1.27e-15 | 4.09e-16 | 6.00 | 1.26e-03 | 8.9 | **0.000** |

정확한 proper symmetry에서도 $\operatorname{rank}=6$이 유지되고 $\lambda_{\min}$이 수치
잡음보다 훨씬 크다. §1.1이 제거되었다.

### 5.3 Tie 원인 규명

$3\times3\times3$ 정육면체 격자(27점, 거리 tie가 대량으로 존재). Phase 1 출력(CUDA):

| 모델 | permutation 오차 | equivariance 오차 |
|---|---|---|
| **pointwise (본 실험)** | **3.01e-16** | 5.73e-15 |
| rank-channel (`WrenchSecondMomentModel` + LN backbone) | **1.92e-01** | 2.71e-01 |

Jitter로 tie를 연속적으로 깨면:

| jitter | pointwise | rank-channel |
|---|---|---|
| 0 | 2.81e-16 | 1.65e-01 |
| $10^{-8}$ | 6.27e-16 | 2.64e-16 |
| $10^{-4}$ | 6.81e-16 | 2.89e-16 |
| $10^{-2}$ | 1.01e-15 | 2.28e-16 |
| $10^{-1}$ | 4.80e-16 | 2.93e-16 |

앞선 CPU 실행($N=48$, `global_scale`)에서도 같다 (pointwise 5.70e-16,
rank-channel 2.56e-01). 격자의 mean degree는 4.0, trunc 0이다.

rank-channel 모델은 tie가 $10^{-8}$만 깨져도 즉시 machine precision으로 회복한다. 즉
실패는 kernel의 매끄러움이나 반경 정의가 아니라 **정확한 tie에서 "$c$번째 최근접
이웃"이라는 채널 정체성이 정의되지 않는 것**에서 온다.

이는 선행 계획서의 진단을 한 단계 좁힌 결과다. 계획서는 adaptive radius $r_i=d_{i,(k)}$를
유력한 원인으로 지목했으나, §5.4의 변형 표에서 `knn_adaptive`/`knn_shell` 반경을 쓴
pointwise 모델도 격자에서 machine precision을 유지한다. $d_{i,(k)}$는 연속이고
permutation-invariant한 순서통계량이며 pooling이 집합 합이기 때문이다.

### 5.4 구성 변형의 구조 지표

Phase 1 출력(CUDA, $N=128$, `candidate_k=64`, `density_scaled` $\alpha=1.15$).
iid에서 equivariance, $C_2$에서 permutation·rank, 격자에서 tie 불변성.
각 행은 기본 설정에서 한 군데씩만 바꾼 것이다 (부록 B).

| variant | equiv | perm(c2) | perm(tie) | rank(c2) | params |
|---|---|---|---|---|---|
| **default** (고정 shell, bracket 없음) | 1.19e-15 | 5.00e-16 | 5.77e-16 | 6.00 | 18,697 |
| separable-encoder-bracket | 1.59e-15 | 5.02e-16 | 4.43e-16 | 6.00 | 18,953 |
| pairwise-encoder-bracket | 1.88e-15 | 5.45e-16 | 5.46e-16 | 6.00 | 18,909 |
| no-backbone-bracket | 1.63e-15 | 1.06e-15 | 5.07e-16 | 6.00 | 15,625 |
| no-gate | 1.81e-15 | 8.62e-16 | 1.03e-15 | 6.00 | 7,545 |
| full-gram-gate | 2.36e-15 | 5.04e-16 | 4.15e-16 | 6.00 | 18,220 |
| no-global-context | 1.63e-15 | 7.77e-16 | 4.85e-16 | 6.00 | 13,385 |
| message-passing | 2.03e-15 | 5.08e-16 | 8.45e-16 | 6.00 | 26,497 |
| attention-pool | 1.82e-15 | 5.46e-16 | 2.12e-16 | 6.00 | 20,465 |
| sum-pool | 1.38e-15 | 4.94e-16 | 8.82e-16 | 6.00 | 20,465 |
| knn-adaptive-radius | 1.13e-15 | 4.82e-16 | 2.69e-16 | 6.00 | 18,697 |
| knn-shell-radius | 1.22e-15 | 4.76e-16 | 2.51e-16 | 6.00 | 18,697 |
| density-scaled-radius | 1.18e-15 | 4.96e-16 | 5.84e-16 | 6.00 | 18,697 |
| uniform-beta | 1.77e-15 | 6.60e-16 | 5.78e-16 | 6.00 | 15,505 |
| force-invariant | 1.87e-15 | 6.06e-16 | 8.64e-16 | 6.00 | 20,745 |

이 표는 구조만 말한다. 모든 변형이 등변성·permutation·rank를 똑같이 만족한다는 뜻이지
정확도가 같다는 뜻이 아니다. 정확도 기여는 §5.6의 pool 비교를 제외하면 측정하지 않았다.

### 5.5 Full GPU 학습 결과 (Phase 2–3)

`bash experiment/pc_se3_congruence/run_pointwise_gpu_experiments.sh`, §4.4의 full recipe,
CUDA, float64, 18,697 파라미터. 표준 object suite 9 케이스 × 2 타깃 = 18 run,
합계 약 72분 (케이스당 89–153초, $N=512$만 약 17분).

#### Phase 2 — realizability (`--target-graph teacher`)

| dataset | $N$ | train $d$ (ep0 → ep149) | val $d$ | ff | mm | fm | rank | $\lambda_{\min}$ | equiv final |
|---|---|---|---|---|---|---|---|---|---|
| centro $\eta=0$ | 128 | 0.218 → **0.00143** | 0.00146 | 5.3e-04 | 5.5e-04 | — | 6.0 | 2.59e-03 | 1.22e-15 |
| centro $\eta=0.02$ | 128 | 0.194 → **0.00080** | 0.00079 | 2.7e-04 | 2.5e-04 | 3.4e-03 | 6.0 | 2.61e-03 | 2.13e-15 |
| centro $\eta=0.1$ | 128 | 0.196 → **0.00074** | 0.00073 | 1.9e-04 | 2.3e-04 | 1.4e-03 | 6.0 | 2.73e-03 | 1.38e-15 |
| centro $\eta=0.5$ | 128 | 0.184 → **0.00140** | 0.00132 | 3.5e-04 | 4.0e-04 | 1.6e-03 | 6.0 | 3.27e-03 | 3.60e-15 |
| c2 $\eta=0$ | 128 | 0.181 → **0.00115** | 0.00111 | 3.0e-04 | 3.7e-04 | 1.3e-03 | 6.0 | 2.57e-03 | 1.14e-15 |
| tetra $\eta=0$ | 120 | 0.181 → **0.00120** | 0.00121 | 4.1e-04 | 5.2e-04 | — | 6.0 | 2.42e-03 | 2.00e-15 |
| iid | 32 | 0.225 → **0.00096** | 0.00090 | 2.1e-04 | 2.3e-04 | 3.2e-04 | 6.0 | 2.14e-03 | 8.71e-16 |
| iid | 128 | 0.168 → **0.00111** | 0.00106 | 2.9e-04 | 3.1e-04 | 5.2e-04 | 6.0 | 1.52e-03 | 1.24e-15 |
| iid | 512 | 0.165 → **0.00089** | 0.00088 | 2.6e-04 | 2.7e-04 | 5.2e-04 | 6.0 | 6.77e-04 | 2.00e-15 |

모든 run에서 `clamped = 0`, `graph_truncation_frac = 0` ($N=512$만 3.7e-04).

`fm` 열의 "—"는 **측정 불가**를 뜻한다. centro $\eta=0$과 tetra는 타깃의 $K_{fm}$ 블록이
정확히 0이고($\|K_{fm}\|=1.3\times10^{-19}$, $5.0\times10^{-19}$ vs $\|K_{ff}\|\approx
5.7\times10^{-3}$), `err_rel_fm`은 이 0을 분모로 쓰므로 로그의 1.4e+12 같은 값은 의미가
없다. 두 데이터셋에서는 ff·mm와 AIRM 거리만 읽는다. $\eta$가 커지면 분모가 살아나
$\eta=0.02/0.1/0.5$에서 3.4e-3 / 1.4e-3 / 1.6e-3으로 정상화된다.

읽을 것은 세 가지다.

1. **9개 케이스 전부 val $d\le1.5\times10^{-3}$이고 train과 val이 사실상 같다.** 선행
   아키텍처가 구조적으로 실패하던 centro $\eta=0$·$C_2$·tetra가 iid와 구별되지 않는다.
   smoke의 0.008–0.012 대비 약 8–10배 개선이며, 이번에는 수렴한 값이다 (ep120에서
   이미 0.002–0.004, ep149까지 단조 감소).
2. ff와 mm가 같은 크기로 함께 줄어든다. 선행 실패의 특징이던 "ff만 큰" 패턴이 없다.
3. $N=32/128/512$ 전부에서 동일한 수준으로 수렴한다. $\lambda_{\min}$만 $N$에 따라
   6.8e-04까지 내려가는데, 이는 밀도가 높아지면서 Wendland 창의 개별 기여가 작아지는
   것이고 rank는 6으로 유지된다.

§8.4의 판정 기준(val $d\le$ smoke 수준, rank 6, `clamped`$=0$,
`equiv_err_final`$<10^{-12}$, trunc$\approx0$)을 모두 통과한다. 따라서 Phase 3을 읽어도
된다.

#### Phase 3 — analytic 타깃 (`--target-graph kernel`)

| dataset | $N$ | train $d$ (ep0 → ep149) | val $d$ | ff | mm | rank | $\lambda_{\min}$ | equiv final |
|---|---|---|---|---|---|---|---|---|
| centro $\eta=0$ | 128 | 1.390 → 0.717 | 0.729 | 0.264 | 0.395 | 6.0 | 2.65e-03 | 1.84e-15 |
| centro $\eta=0.02$ | 128 | 1.392 → 0.713 | 0.732 | 0.263 | 0.388 | 6.0 | 2.66e-03 | 6.03e-15 |
| centro $\eta=0.1$ | 128 | 1.384 → 0.701 | 0.717 | 0.233 | 0.345 | 6.0 | 2.67e-03 | 4.02e-15 |
| centro $\eta=0.5$ | 128 | 1.317 → 0.672 | 0.670 | 0.198 | 0.289 | 6.0 | 2.56e-03 | 8.88e-15 |
| c2 $\eta=0$ | 128 | 1.406 → 0.730 | 0.756 | 0.238 | 0.367 | 6.0 | 2.71e-03 | 2.13e-15 |
| tetra $\eta=0$ | 120 | 0.998 → 0.517 | 0.544 | 0.178 | 0.253 | 6.0 | 3.36e-03 | 1.75e-15 |
| iid | 32 | 2.576 → 1.708 | 1.718 | 0.640 | 0.760 | 6.0 | 7.58e-04 | 7.49e-15 |
| iid | 128 | 1.367 → 0.695 | 0.708 | 0.208 | 0.264 | 6.0 | 1.15e-03 | 1.97e-15 |
| iid | 512 | 0.900 → **0.293** | 0.297 | 0.077 | 0.101 | 6.0 | 1.16e-03 | 3.24e-15 |

`clamped = 0`, trunc = 0 ($N=512$만 3.7e-04)로 Phase 2와 동일하다.

이 표의 val $d$는 **수렴한 바닥**이다. ep30에서 이미 최종값의 10% 안에 들어오고
(예: centro $\eta=0$ 0.782 → 0.729) 이후 120 epoch 동안 거의 움직이지 않으며, train과
val의 차이가 5% 내외다. 즉 최적화가 덜 된 것도 과적합도 아니라 **모델 클래스가 타깃을
포함하지 않는 데서 오는 바닥**이고, 이는 §7.1이 미리 명시한 상황이다. Phase 2가 같은
설정에서 $10^{-3}$까지 내려간다는 사실이 이 해석을 뒷받침한다 — 같은 옵티마이저·같은
epoch에서 타깃만 바꾸면 수백 배 차이가 난다.

바닥의 크기 자체는 정보를 준다.

- **$N$에 대해 단조 감소한다.** iid에서 1.718 / 0.708 / **0.297** ($N=32/128/512$).
  analytic 타깃은 edge 단위 second moment이므로 점이 촘촘해질수록 latent covector의
  point 단위 second moment로 근사하기 쉬워진다.
- **대칭이 깨질수록 근소하게 낮아진다.** centro $\eta=0\to0.5$에서 0.729 → 0.670.
- 부록 C.2가 보이듯 전역 scale이 $e^{\pm1}$만 틀려도 $d\approx2.45$이므로,
  $d\approx0.3$–$0.7$은 scale이 아니라 방향(고유벡터) 오차가 지배한다는 뜻이다.

선행 `tensor_kernel_experiment_report.md` §0의 val $d = 3.76\times10^{-4}\sim
6.09\times10^{-4}$와 나란히 놓을 때 주의한다. 그쪽은 **matched target**(모델과 타깃이
같은 compact-kernel second-moment 법칙)이고 이쪽은 아니므로, 직접 비교되는 것은 val $d$가
아니라 rank(둘 다 6) · equivariance(둘 다 $10^{-15}$ 대) · clamp 안정성(둘 다 0)이다.
표현력을 같은 축에서 비교하려면 Phase 2의 $\le1.5\times10^{-3}$이 대응하는 수치다.

### 5.6 Encoder 단순화 — 고정 shell vs 학습된 attention (smoke)

§2.4의 주장을 직접 측정한다. §4.4의 **smoke** 설정, analytic(`kernel`) 타깃, val $d$.
이 표만 §5.5보다 앞선 CPU 실행이며, 절대값을 §5.5와 섞어 읽지 않는다.

| dataset | 고정 shell (정규화 X) | **고정 shell + 정규화 (기본)** | 학습 attention |
|---|---|---|---|
| encoder 파라미터 | **0** | **0** | 1,768 |
| centro $\eta=0$ | **1.257** | 1.293 | 1.452 |
| c2 $\eta=0$ | **1.375** | 1.400 | 1.387 |
| tetra $\eta=0$ | 1.530 | **1.218** | 1.552 |
| iid | 1.539 | 1.442 | **1.200** |
| 평균 | 1.425 | **1.338** | 1.398 |

가운데 열이 현재 기본 설정이다. `attention`은
separable bracket과 함께 측정했다. **encoder bracket을 켜면** centro 1.318 / c2 1.401 /
tetra 1.231 / iid 1.448, 평균 1.350으로 네 데이터셋 전부 근소하게 나쁘다 — §2.4의
중복성 논증과 일치한다.

세 가지를 읽을 수 있다.

1. 학습된 가중 MLP를 encoder에서 제거해도 평균 성능이 나빠지지 않는다 (1.398 → 1.338).
2. **정규화는 예외다.** 정규화가 있으면 tetra에서 $1.530\to1.218$로 크게 낫고 iid에서도
   앞선다. 비율은 shell의 선형결합이 아니라 LN-Linear가 만들 수 없기 때문이며, 그래서
   이것만 encoder에 남겼다. 반대로 centro·c2에서는 정규화 없는 쪽이 근소하게 낫다.
3. `attention`이 확실히 앞서는 곳은 iid(1.200)뿐이다. 국소 밀도가 균일하지 않은
   cloud에서 soft degree와 밀도 대비 길이에 반응할 수 있는 것이 이유로 보이나, 이는
   단일 seed·smoke 규모의 추정이다.

결과적으로 encoder 파라미터가 1,896 → **0개**가 되었다. `attention`과 encoder bracket은
부록 B의 플래그로 되돌아간다.

**iid 격차는 아직 재확인되지 않았다.** §8.6대로 이번 라운드는 Phase 1–3만 돌렸고
`--phase ablation`을 실행하지 않았으므로, full recipe에서 `attention`이 iid에서 앞서는지는
미측정이다 (§9-1).

### 5.7 Graph degree/truncation 보정

이방성 Gaussian cloud 16개, $k_{\rm target}=16$:

| $N$ | mode | $\alpha$ | cand | mean deg | max deg | trunc |
|---|---|---|---|---|---|---|
| 48 | global_scale | 0.75 | 32 | 8.3 | 23 | 0.000 |
| 128 | global_scale | 0.75 | 32 | 21.8 | 32 | **0.359** |
| 128 | global_scale | 0.75 | 64 | 25.2 | 63 | 0.000 |
| 128 | global_scale | 0.50 | 32 | 9.5 | 32 | 0.007 |
| 512 | global_scale | 0.75 | 64 | 55.0 | 64 | **0.697** |
| 48 | **density_scaled** | **1.15** | 48 | 9.6 | 25 | 0.000 |
| 128 | **density_scaled** | **1.15** | 48 | 13.5 | 44 | 0.000 |
| 512 | density_scaled | 1.15 | 48 | 15.3 | 48 | 0.008 |
| 512 | **density_scaled** | **1.15** | **64** | 15.3 | 60 | **0.000** |

GPU 실험 전에 반드시 알아야 할 실무 사항이다. 전역 스케일 반경은 $N$이 커지면 support가
함께 커져 $N=128$에서 truncation이 0.36까지 오른다. §2.2의 밀도 보정
($\alpha=1.15$, $k_{\rm target}=16$, `candidate_k=64`)이 $N=48/128/512$ 전부에서 평균
degree 9.6/13.5/15.3, truncation 0.000을 준다. 이것이 기본값을 밀도 보정으로 정한 이유다.

### 5.8 회귀 검증

- `pytest -q test/` → **40 passed** (기존 tensor 5 + 신규 35)
- `run_tensor_kernel_suite.py --quick --phase sanity` → analytic 정확 일치 유지,
  equivariance $\sim3\times10^{-15}$
- `--encoder plueck / learnable / bracket`, `--dataset fiber` 경로 정상 동작하며
  $C_2$에서 rank 2 collapse도 그대로 재현 — 선행 진단이 훼손되지 않았다
- CUDA 경로 확인 완료: 구조 검증(§5.2, §5.3)과 **full 학습 18 run**(§5.5) 모두 CUDA·
  float64에서 실행되었고, 학습 전후 equivariance가 전 run에서 $10^{-15}$ 대를 유지한다

---

## 6. 결과 해석

### 6.1 Rank collapse 해결

정확한 proper symmetry(centro $\eta=0$, $C_2$, $A_4$)에서 $\operatorname{rank}K=6$,
$\lambda_{\min}\sim10^{-3}$이다. 원인은 §2.8의 permutation gauge다 — 대칭이 factor를
고정하는 대신 서로 permutation시킬 수 있으므로 $\operatorname{Fix}_H$ 제약이 개별
factor에 걸리지 않는다.

### 6.2 Tie 실패 원인의 분리

격자에서 $5.7\times10^{-16}$ vs $2.6\times10^{-1}$, 그리고 jitter $10^{-8}$에서의 즉각
회복이 원인을 rank-as-channel로 좁힌다. compact kernel도 adaptive radius도 아니다.
그래프 쪽에 남은 tie 경로는 candidate truncation 하나뿐이므로 상시 감시한다.

### 6.3 최적화 가능성

같은 클래스 teacher 타깃에서 9개 케이스 전부 val $d\le1.5\times10^{-3}$로 수렴하고,
학습 전후 등변성이 $10^{-15}$ 대를 유지하며 `clamped = 0`이다. gradient step이 구조를
훼손하지 않는다. 대칭 데이터셋(centro $\eta=0$, $C_2$, tetra)과 비대칭 iid의 수렴값이
구별되지 않는다는 점이 §6.1과 짝을 이룬다 — rank가 살아 있으니 최적화도 막히지 않는다.

analytic 타깃의 잔차는 이와 별개다. 같은 옵티마이저·같은 epoch에서 타깃만 바꾸면
$10^{-3}$이 $10^{-1}$대가 되고, 그 값이 30 epoch 이후 움직이지 않으며 train/val이
붙어 있다. 세 조건이 함께 성립하면 남은 설명은 표현력(모델 클래스가 타깃을 포함하지
않음)뿐이다. 이것이 §7.1을 측정으로 확인한 것이다.

### 6.4 Encoder는 학습될 이유가 없다

§2.4의 논증과 §5.6의 측정이 일치한다. encoder에 있던 두 학습 후보(가중치 MLP, bracket
사영)는 모두 뒤따르는 LN 연산이 흡수할 수 있어 제거되었고, 남은 것은 정규화 하나뿐이다.

> Encoder는 수학적 타입을 맞추는 고정 lift이고, 학습 파라미터를 갖지 않는다.

이로써 학습은 전부 LN 블록의 등변 채널 혼합과 불변 스칼라 gate에서 일어난다.

### 6.5 선행 계획서 대비 수정된 판단

| 계획서의 서술 | 본 실험의 결론 |
|---|---|
| backbone을 point마다 적용하려면 $[BN,C,6,1]$로 reshape | 불필요. `[B,C,6,N]`이 이미 pointwise다 (§2.1) |
| adaptive radius $r_i=d_{i,(k)}$가 tie 실패의 유력한 원인 | 아니다. 원인은 rank-as-channel이며 adaptive radius도 tie-safe하다 (§5.3, §5.4) |
| 이웃 축약에 learned attention/set-aggregation 필요 | 고정 shell + 정규화로 충분하다 (§2.4, §5.6) |
| 방법 B의 이중합은 $O(k^2)$ | 계수가 분리되면 $O(k)$이며, 그 경우 블록의 bracket과 중복이라 아예 불필요하다 (§2.4, 부록 B.3) |

---

## 7. 이 실험이 증명하지 않는 것

### 7.1 Matched target이 아니다

analytic 타깃은 raw wrench의 edge-level second moment이고 본 모델은 latent covector의
second moment다. bracket 한 번만 거쳐도 lift에 대해 bilinear이고 gate까지 지나면 비다항
함수가 되므로, 두 함수족은 일치하지 않는다. `kernel` 타깃의 잔차 하한에는 타깃 불일치가
섞여 있으며, 그 절대값으로 표현력을 판단해서는 안 된다.

### 7.2 단일 seed다

§5.5는 full recipe로 수렴했지만 `model_seed=0`, `data_seed=100`, `teacher_seed=7`
하나뿐이다. seed 간 분산을 측정하지 않았으므로 케이스 사이의 작은 차이
(예: teacher에서 centro $\eta=0$의 0.00143 vs $\eta=0.1$의 0.00074)는 유의하다고 볼 수
없다. 유의하게 읽을 수 있는 것은 자릿수 차이뿐이다. §5.6은 여전히 60 epoch·256 샘플의
CPU smoke다.

### 7.3 정확도 ablation을 하지 않았다

§5.4는 구조 지표만이다. gate·backbone bracket·message passing이 정확도에 얼마나
기여하는지는 측정하지 않았다.

### 7.4 Second moment의 비단사성

출력은 21 자유도뿐이므로 $P\mapsto K$가 injective일 수 없다. 필요한 것은 shape
injectivity가 아니라 stiffness-relevant sufficiency이며, 실제 데이터에서는 $K$가
재질·내부 구조·경계 조건에도 의존하므로 $P$만으로 결정되지 않을 수 있다.

### 7.5 Full rank는 보장값이 아니다

$\operatorname{rank}K=\dim\operatorname{span}\{\sqrt{\beta_{i,h}}z_{i,h}\}$이므로, 본
실험이 보인 것은 대칭에 의한 **가짜** rank deficiency가 사라졌다는 것이지 어떤 입력에서도
full rank가 나온다는 것이 아니다.

### 7.6 Sampling density 전이와 실측 stiffness

`--pw-radius-mode fixed`와 `--pw-normalize beta`로 학습한 뒤 $N$을 바꿔 평가하는 실험,
그리고 실제 stiffness 데이터에 대한 검증은 아직 하지 않았다.

---

## 8. 재현 방법

모든 명령은 repo root에서 `lieneurons` 환경으로 실행한다. §8.3–§8.5는 아래 한 줄로
묶여 있고, 2026-08-07에 이 스크립트로 실행한 결과가 §5.2·§5.3·§5.5다.

```bash
conda activate lieneurons
bash experiment/pc_se3_congruence/run_pointwise_gpu_experiments.sh
```

`DEVICE`, `WANDB_MODE`, `VERIFY_OUT` 환경변수로 장치·로깅·출력 경로를 바꿀 수 있다.

### 8.1 사전 점검

```bash
cd /PublicSSD/jhri626/LieNeurons
conda run -n lieneurons python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
conda run -n lieneurons python -m pytest -q test/test_pc_pointwise_pipeline.py
```

### 8.2 그래프 설정 확인

기본값은 §5.7에서 보정한 조합이며 `run_pointwise_suite.py`가 이미 이를 쓴다.

```
--pw-radius-mode density_scaled --pw-radius-alpha 1.15 --pw-target-k 16 --pw-candidates 64
```

`blockage_bench.py`를 직접 쓸 때는 명시해야 한다. 다른 $N$이나 다른 cloud 분포를
쓴다면 학습 전에 degree/truncation부터 확인한다.

```bash
conda run -n lieneurons python experiment/pc_se3_congruence/verify_pointwise.py \
  --device cuda --n-points 128 --candidates 64 \
  --radius-mode density_scaled --radius-alpha 1.15 --target-k 16
```

`trunc` 열이 0이 아니면 `--pw-candidates`를 올리거나 `--pw-radius-alpha`를 내린다.

### 8.3 Phase 1 — 구조 검증 (수 분)

```bash
conda run -n lieneurons python experiment/pc_se3_congruence/verify_pointwise.py \
  --device cuda --full \
  --n-points 128 --candidates 64 \
  --radius-mode density_scaled --radius-alpha 1.15 --target-k 16 \
  --out experiment/pc_se3_congruence/pointwise_verify_results.json
```

판정 기준: equiv $<10^{-12}$, perm $<10^{-12}$, rank $=6$, trunc $=0$, 그리고 격자에서
pointwise perm $<10^{-12}$이면서 rank-channel perm $\gtrsim10^{-2}$.
**결과: 통과** (§5.2–§5.4, `pointwise_verify_results.json`).

### 8.4 Phase 2 — Realizability (teacher 타깃)

analytic보다 먼저 돌린다. 표준 object suite(centro $\eta\in\{0,0.02,0.1,0.5\}$, $C_2$,
tetra, iid $N\in\{32,128,512\}$) 9개 케이스가 각각 별도 W&B run으로 기록된다.

```bash
conda run -n lieneurons python experiment/pc_se3_congruence/run_pointwise_suite.py \
  --phase teacher --recipe full --device cuda --wandb-mode online
```

판정 기준: 모든 대칭 데이터셋에서 val $d$가 smoke 수준(0.012) 이하로 내려가고,
`rank_pred`$=6$, `clamped`$=0$, `equiv_err_final`$<10^{-12}$,
`graph_truncation_frac`$=0$. 어느 하나라도 어긋나면 analytic 결과는 읽을 필요가 없다.
**결과: 통과** — 9 케이스 전부 val $d\le1.5\times10^{-3}$ (§5.5).

### 8.5 Phase 3 — Analytic 타깃 (같은 suite)

```bash
conda run -n lieneurons python experiment/pc_se3_congruence/run_pointwise_suite.py \
  --phase analytic --recipe full --device cuda --wandb-mode online
```

선행 tensor-kernel 실험과 같은 타깃, 같은 object suite이므로
`tensor_kernel_experiment_report.md`의 수치와 나란히 놓을 수 있다. 판정 기준은 rank 6,
`clamped`$=0$, `equiv_err_final`$<10^{-12}$, trunc $=0$을 유지하면서 val $d$가 수렴할
것이며, **val $d$의 절대값으로 표현력을 판단하지 않는다** (§7.1).
**결과: 통과** — 구조 지표는 Phase 2와 동일하고 val $d$는 0.297–1.718에서 평탄화한다
(§5.5).

### 8.6 Ablation은 이번 라운드에서 제외했다

`run_pointwise_suite.py --phase ablation`이 준비되어 있으나(17종 × 9 케이스 = 153 run),
이번 목표는 선행 실험과 같은 축에서 비교 가능한 수치를 얻는 것이므로 Phase 1–3만
돌렸다. Phase 2–3이 통과했으므로 다음 라운드는 여기서 시작한다 (§9-1). seed 3개로 좁혀
돌리고, 특히 §5.6의 `attention` iid 격차를 full recipe에서 재확인한다.

```bash
conda run -n lieneurons python experiment/pc_se3_congruence/run_pointwise_suite.py \
  --phase ablation --ablation-target kernel --recipe full --device cuda \
  --ablations default,no-gate,no-backbone-bracket,rank-channel-baseline
```

### 8.7 단일 케이스와 인자 전달

```bash
conda run -n lieneurons python experiment/pc_se3_congruence/blockage_bench.py \
  --dataset c2 --encoder pointwise --method covector \
  --target-graph teacher --recipe full --device cuda \
  --pw-radius-mode density_scaled --pw-radius-alpha 1.15 --pw-target-k 16 \
  --pw-candidates 64 --pw-channels 8 16 32 16 --pw-factors 8 \
  --wandb-mode online
```

`run_pointwise_suite.py`에서는 `--` 뒤 인자가 `blockage_bench.py`로 그대로 전달된다.

```bash
conda run -n lieneurons python experiment/pc_se3_congruence/run_pointwise_suite.py \
  --phase teacher --recipe full --device cuda --wandb-mode online \
  -- --epochs 60 --batch 128 --lr 2e-3
```

OOM 대응 순서는 `--batch` 축소 → message passing off 또는 `--pw-msg-channels` 축소 →
`--pw-bracket none`(기본)으로 복귀 → `--pw-candidates` 축소(단 truncation 확인)다.

### 8.8 단위 테스트와 구조 검증

```bash
conda run -n lieneurons python -m pytest -q test/test_pc_pointwise_pipeline.py
conda run -n lieneurons python experiment/pc_se3_congruence/verify_pointwise.py --full
```

### 8.9 결과를 문서에 반영하는 법

2026-08-07 라운드에서 아래를 모두 수행했다. 다음 라운드에서도 같은 순서를 따른다.

1. §5.2–§5.4를 Phase 1 출력으로 교체. `pointwise_verify_results.json`은 Phase 1이
   덮어쓴다.
2. §5.5를 Phase 2·3의 full 결과 표(9 케이스 × 2 타깃)로 교체.
3. §0, 머리말 상태 줄, `README.md` §6.5의 수치를 갱신.
4. analytic 결과는 `tensor_kernel_experiment_report.md` §0(val $d$ 3.76e-4 ~ 6.09e-4,
   rank 6, equiv 2e-15~8e-15)과 나란히 놓되, 그쪽은 matched-target 검증이고 이쪽은
   아니므로 직접 비교되는 것은 val $d$가 아니라 rank·equivariance·clamp 안정성임을
   명시한다.
5. `err_rel_fm`은 centro $\eta=0$·tetra에서 분모가 0이므로 표에 그대로 옮기지 않는다
   (§5.5).

---

## 9. 다음 권장 실험

1. **Ablation의 정량화** (Phase 1–3 통과로 해금됨). §5.4는 구조 지표만이다. 동일 타깃·
   seed 3개로 val $d$를 비교해 gate / backbone bracket / message passing / pool 방식의
   기여를 분리한다. teacher 타깃이 $10^{-3}$까지 내려가므로 이제 차이를 볼 해상도가
   있다. §5.6의 `attention` iid 격차 재확인이 여기 포함된다 (§8.6의 명령).
2. **Analytic 잔차의 분해.** §5.5 Phase 3의 바닥이 iid에서 $N$에 대해 단조 감소한다
   (1.718 → 0.708 → 0.297). $N\in\{1024,2048\}$까지 늘려 바닥이 0으로 가는지, 아니면
   유한한 값으로 수렴하는지 본다. 전자면 잔차는 이산화 오차이고 후자면 함수족 차이다.
   같은 run에서 AIRM 거리를 scale 성분과 방향 성분으로 분해해 기록한다 (부록 C.2).
3. **Candidate·반경 sweep.** $k_{\rm target}\in\{8,16,32\}$와 $\alpha$ 격자에서
   정확도·runtime·$\lambda_{\min}$·truncation을 함께 본다. $N=512$에서
   `graph_truncation_frac`이 3.7e-04로 0이 아니므로 `--pw-candidates`도 함께 올린다.
4. **Seed 분산.** §5.5는 단일 seed다 (§7.2). seed 3–5개로 teacher 타깃을 다시 돌려
   케이스 간 차이가 seed 잡음보다 큰지 확인한다.
5. **Near-tie 학습 곡선.** `lattice_clouds(jitter=·)`로 학습해 tie 부근에서 rank-channel
   모델과의 val $d$ 차이를 정량화한다. §5.3은 미학습 모델 기준이다.
6. **Sampling density 전이.** `--pw-radius-mode fixed`와 `--pw-normalize beta`로 학습한 뒤
   $N$을 바꿔 평가한다. 물리 stiffness로 가려면 필요한 성질이다.
7. **Mismatched target.** hard-kNN / radius / FEM / 실측 stiffness 타깃에 대해 latent
   second moment가 얼마나 근사하는지. 단 exact symmetry에서는 hard-kNN 타깃 자체의
   등변성부터 검사해야 한다.
8. **General wrench 확장.** pure couple와 nonzero-pitch screw feature를 추가해
   pure-force Plücker cone의 표현 제약을 측정한다.
9. **Sparse neighbor backend.** $O(N^2)$ 거리 행렬을 spatial hash / radius query로 교체.
   $N\gtrsim10^3$에서 지배적 비용이 된다.

---

## 10. 최종 판단

이번 결과는 다음 설계 원칙을 지지한다.

$$
\boxed{
\text{tie-safe local graph}
\;\longrightarrow\;
\text{거리로 라벨링한 집합 축약}
\;\longrightarrow\;
\text{point 축을 유지한 LN backbone}
\;\longrightarrow\;
\text{late Gram}
}
$$

두 가지가 핵심이다. 첫째, **point 축을 Gram까지 유지하는 것**은 메모리 선택이 아니라
대칭이 factor를 permutation할 수 있게 하는 구조적 조건이다. 둘째, **채널 라벨은 이웃
집합의 연속 함수여야 한다** — rank는 tie에서 정의되지 않으며, 그것이 선행 실패의
실제 원인이었다.

부수적으로 얻은 결론은 encoder에 관한 것이다.

$$
\boxed{
\text{Encoder는 수학적 타입을 맞추는 고정 lift이고, 학습 파라미터를 갖지 않는다.}
}
$$

LN 연산이 흡수할 수 있는 것(가중치, bracket)은 전부 뒤로 넘기고, 흡수할 수 없는
것(정규화)만 남긴다. 이 규칙으로 encoder 파라미터가 1,896에서 0으로 줄었고 성능은
오히려 근소하게 좋아졌다.

Full GPU 실험(§5.5)은 이 구조적 결론을 학습 쪽에서도 확인한다. 같은 클래스 타깃에서는
9개 케이스 전부 val $d\le1.5\times10^{-3}$로 수렴하고, 대칭 데이터셋과 iid의 수렴값이
구별되지 않으며, 150 epoch 학습 후에도 rank 6과 $10^{-15}$ 등변성이 유지된다. 구조를
보존하는 것과 학습되는 것 사이에 상충이 없다.

다만 남은 것이 둘이다. **단일 seed**이므로 케이스 사이의 작은 차이는 유의하지 않고
(§7.2), analytic 타깃의 잔차 0.30–1.72는 표현력 부족이 맞다는 것까지만 확인했을 뿐
그 크기를 이산화 오차와 함수족 차이로 분해하지 못했다 (§9-2). 정확도 ablation도
아직이다 (§7.3, §9-1).

---

## 부록 A. 설계 문답

설계 과정에서 제기된 질문과, 그에 답하기 위해 수행한 측정을 기록한다.

### A.1 이웃 축 $k$를 그냥 $C_0$처럼 쓰면 안 되는가?

안 된다 — 정확히는, 채널 라벨을 rank로 두면 안 된다. $k\to C_0$ 변환 자체가 문제가
아니라, 채널 라벨이 이웃 집합의 불연속 함수인지 연속 함수인지가 문제다.

| | 계수 출처 | 순열 $\Pi$에서 | tie에서 |
|---|---|---|---|
| rank-channel | 슬롯 index $j$ (학습 파라미터) | 따라가지 않음 → 깨짐 | "몇 번째"가 정의 안 됨 |
| 고정 shell (본 설계) | 거리 $q_{ij}$ (파라미터 0) | 따라감 → 불변 | 두 행이 동일 |

측정값은 §5.3이다: 격자에서 rank-channel $2.6\times10^{-1}$, 본 모델 $5.7\times10^{-16}$.

### A.2 $k\to C_0$가 선형 행렬이고 거리만 쓰기 때문에 permutation invariant한 것인가?

행렬인 것은 맞지만 **고정 행렬이 아니라 데이터로부터 만들어지는 행렬**이다.
$X_i=A_i^\top W_i$에서 $(A_i)_{jc}=a_{ij,c}$는 그 edge의 거리로 계산되며 학습
파라미터가 아니다. 이웃 순서를 바꾸면 $W_i\mapsto\Pi W_i$인데 $A_i$도 같이 움직여서

$$
(\Pi A_i)^\top(\Pi W_i)=A_i^\top\Pi^\top\Pi W_i=A_i^\top W_i .
$$

학습 가능한 고정 행렬 $M\in\mathbb R^{k\times C_0}$이었다면 $M$은 따라 움직이지 않으므로
깨진다. 그 $j$가 곧 rank다. Tie까지 커버되는 이유는 $q_{ij}=q_{il}$이면 $A_i$의 두 행이
완전히 동일하기 때문이다 (§2.8).

### A.3 Edge 불변량 13차원은 과한 것 아닌가?

기본 경로에서는 13차원이 아예 생성되지 않는다.

| 설정 | `EdgeInvariants` 인스턴스 |
|---|---|
| **기본** (고정 shell pool, message passing off) | **0개** |
| `attention` pool | 1개 |
| message passing on | 3개 |

기본 경로가 그래프에서 받아 쓰는 것은 $q_{ij}$ 하나뿐이고 $\phi(q_{ij})$는 그 함수다
(§2.2). 13차원은 선택 경로에만 남아 있으며, 거기서도 정보량 기준으로는 네 개만 독립이다.

| 성분 | 개수 | 새 정보인가 |
|---|---|---|
| $q_{ij}$ | 1 | ✔ |
| $\log(1+d_{ij})$ | 1 | ✔ cloud의 절대 scale ($q$는 $r_i$로 나눠 없앴다) |
| $\nu_i=\sum_l\phi(q_{il})$ (soft degree) | 1 | ✔ anchor 국소 밀도 |
| $d_{ij}/\bar d_i$, $\bar d_i=\frac{\sum_l\phi(q_{il})d_{il}}{\nu_i}$ | 1 | ✔ 밀도 대비 길이 |
| $\phi(q_{ij})$ | 1 | ✘ $q$의 함수 |
| $\rho_t(q_{ij})$, $t=1..n_{\rm rbf}$ | 8 | ✘ $q$의 함수 — MLP가 급격한 radial 의존성을 쉽게 학습하도록 넣은 basis 확장 |

### A.4 가중치 MLP 없이 어떻게 동작하는가? 모델에 MLP가 없는가?

MLP가 사라진 것은 encoder에서뿐이다. 모델 전체에는 MLP가 8개, 13,064 파라미터
(전체의 70%)로 남아 있다 (§2.7).

없어도 되는 이유는 §2.4의 흡수 논증이다. 고정 shell로 축약한 뒤 LN-Linear가 채널을
섞으면 $\mathrm{span}\{\rho_c\}$ 안의 임의 radial 가중 함수가 복원되므로, 학습된 radial
kernel은 사라진 것이 아니라 한 layer 뒤로 옮겨간 것이다.

남은 MLP가 gate $g_{i,c}$와 factor weight $\beta_{i,h}$에만 있는 것도 우연이 아니다.
$6$축에는 등변 연산만 허용되므로 MLP는 6-벡터에 직접 작용할 수 없고, "불변 스칼라를
받아 스칼라를 내는" 자리에만 놓일 수 있다.

### A.5 Encoder의 bracket은 필요한가?

필요 없어서 제거했다. 두 가지 이유가 §2.4에 있다: (i) block 0이 이미 같은 bracket을
계산하므로 중복이고, (ii) $X^{(0)}$으로 만들어지므로 $X^{(0)}$이 사라지는 국소 대칭에서는
함께 사라져 원래의 parity 동기를 만족하지 못한다.

측정값도 제거를 지지한다 (§5.6). teacher 타깃 val $d$는 제거 0.011/0.010/0.012/0.009,
유지 0.017/0.021/0.018/0.012이고, analytic 타깃도 네 데이터셋 전부 제거한 쪽이 낫다.
결과적으로 encoder 파라미터가 0이 되었다.

---

## 부록 B. 구성 옵션 전체

본문은 기본 경로만 서술했다. 아래는 한 군데씩만 바꾼 변형이다. 구조 지표는 모두
동일했고(§5.4), 정확도 기여는 §5.6의 pool 비교를 제외하면 측정하지 않았다.

### B.1 이웃 축약 (`--pw-pool`)

| 값 | 식 | encoder 파라미터 |
|---|---|---|
| **`basis_mean`** (기본) | §2.4 — 고정 shell + shell별 정규화 | 0 |
| `basis` | 정규화 없이 $a_{ij,c}=\phi(q_{ij})\rho_c(q_{ij})$ | 0 |
| `attention` | $a_{ij,c}=\dfrac{\phi(q_{ij})e^{\eta_{ij,c}-\mu_{i,c}}}{\sum_l\phi(q_{il})e^{\eta_{il,c}-\mu_{i,c}}}$, $\eta$는 13차원 불변량의 MLP | 1,768 |
| `sum` / `mean` | $a=\phi\,\Phi_\theta(s)$ (mean은 soft degree로 나눔) | 1,768 |

`attention`이 기본 경로보다 더 보는 것은 soft degree와 밀도 대비 길이, 그리고 그
의존성의 비선형성이다. 정규화는 기본값이 이미 갖고 있다. 대신 구현 세부가 둘 붙는다.

1. **window $\phi$가 정규화 분모 안에 있어야 한다.** 밖에 두면 support 경계를 드나드는
   edge가 $\exp\eta$만큼 분모를 $O(1)$ 흔들어, $\phi(1)=0$으로 확보한 경계의 매끄러움이
   깨진다.
2. **shift $\mu_{i,c}$는 support 안에서만 최대를 잡아야 한다.** 밖의 logit이 shift를
   정하면 실제 weight가 전부 underflow한다. 구현은 support 밖을 `finfo.min/4`로 채워
   최대를 잡고 $(\eta-\mu)$를 0 이하로 clamp하므로, support가 빈 anchor에서도
   $0\cdot\infty$ 대신 정확히 0이 나온다.

### B.2 Edge 불변량 (`--pw-rbf`)

`attention`/`sum`/`mean` pool과 message passing이 쓰는 $n_{\rm rbf}+5$차원 벡터. 성분과
독립성은 부록 A.3의 표를 참조.

### B.3 Encoder bracket (`--pw-bracket`)

| 값 | 식 | 비용 |
|---|---|---|
| **`none`** (기본) | 1차 모멘트 채널만 — encoder 파라미터 0 | $O(k)$ |
| `separable` | $[X^{(0)}U,\ X^{(0)}V]_*$. bilinearity로 $\sum_{j,l}a_{ij,c_1}a_{il,c_2}[w_{ij},w_{il}]_*=[X^{(0)}_{i,c_1},X^{(0)}_{i,c_2}]_*$ 이므로 $O(k^2)$가 아니라 $O(k)$지만, **block 0의 bracket과 중복이다** (§2.4) | $O(k)$ |
| `pairwise` | 분리되지 않는 이중합 (부록 C.1) | $O(k^2)$ |

`pairwise`의 실질적 차이는 국소 대칭에서 드러난다. Neighborhood가 국소적으로
antipodal이면 모든 1차 모멘트가 사라지고 그것으로 만든 separable bracket도 함께
사라지지만, pairwise 항은 $(j,l)\to(-j,-l)$에서 부호가 두 번 뒤집혀 짝수 parity로
살아남는다.

### B.4 Gate와 head

| 플래그 | 기본 | 대안 |
|---|---|---|
| `--pw-gate` | `projected` ($S_i\in\mathbb R^P$) | `full` — $S_i=X_i^\top Q^{-1}X_i\in\mathbb R^{C\times C}$ 전체 Gram을 행 단위로 gate ($O(C^2)$) / `none` — gate 제거 (18,697 → 7,545) |
| `--pw-no-global-context` | context 사용 | 제거 — gate·head가 국소 불변량만 봄 |
| `--pw-no-bracket-layers` | bracket 사용 | 제거 — backbone이 LN-Linear + gate만 |
| `--pw-normalize` | `nh` ($Z=NH$) | `beta` ($Z=\sum\beta$, sampling 밀도 불변) / `one` ($Z=1$, 접촉량에 비례) |
| `--pw-beta` | `learned` | `uniform` ($\beta\equiv1$) |
| `--pw-force-invariant` | off | on — gate·head 불변량에 $f_c\!\cdot\!f_d$ 추가. force slot이 translation-blind이므로 진짜 불변량이며 등변 projection $N_*:(f,m)\mapsto(f,0)$에서 유도된다. 기본 off는 framework note의 Klein gate를 그대로 두기 위해서다 |

### B.5 Message passing (`--pw-message-passing`, 기본 off)

켜면 각 블록 앞에 다음이 붙는다.

$$
M_i^{(\ell)}=\frac{1}{\nu_i}\sum_j\gamma^{(\ell)}_{ij}X_j^{(\ell)}W_m^{(\ell)},
\qquad
\widetilde X_i^{(\ell)}=X_i^{(\ell)}+M_i^{(\ell)},
$$
$$
\gamma^{(\ell)}_{ij}=\phi(q_{ij})\,
\Gamma_{\theta_\ell}\Bigl(s_{ij},\ \varsigma(\langle U_i,V_i\rangle),\
\varsigma(\langle U_j,V_j\rangle),\ \varsigma(\langle U_i,V_j\rangle)\Bigr).
$$

$\gamma$가 불변 스칼라이므로 $M$은 $X_j$의 coadjoint 법칙을 물려받는다. $k$축을 다시
만드는 유일한 지점이므로 메모리를 지배한다: gather가 $[B,C_{\rm msg},6,N,k]$이고
$B{=}64,C_{\rm msg}{=}8,N{=}128,k{=}64$에서 201 MB다.

### B.6 반경 정의 (`--pw-radius-mode`)

| 값 | $r_i$ | 성격 |
|---|---|---|
| **`density_scaled`** (기본) | §2.2 | $N$이 변해도 평균 degree 유지 |
| `global_scale` | $\alpha\sigma(P)$ | degree가 $N$에 따라 증가 (§5.7 주의) |
| `fixed` | $r_0$ | 물리 길이. sampling density 전이 실험용 |
| `knn_adaptive` | $d_{i,(k_s)}$ | 연속·permutation invariant지만 전체 프로파일이 순서통계량 하나에 묶임 |
| `knn_shell` | $d_{i,(k_s)}+\epsilon$, hard weight | 등거리 shell 전체 포함 (degree $\ne k_s$ 허용) |

---

## 부록 C. 유도 상세

### C.1 Pairwise bracket의 Levi-Civita 축약

계수를 pair 불변량의 함수로 두면

$$
X^{\rm br}_{i,c}=\sum_{j,l}b_{ijl,c}\,[w_{ij},w_{il}]_*,
\qquad
b_{ijl,c}=\phi(q_{ij})\phi(q_{il})\,
\Psi_{\theta,c}\bigl(q_{ij},q_{il},\cos\theta_{jl},q_{ij}q_{il}\bigr),
$$

$\cos\theta_{jl}=\hat f_{ij}\!\cdot\!\hat f_{il}$. $[\cdot,\cdot]_*$가 $(j,l)$에 반대칭
이므로 $b$의 반대칭 성분만 기여한다.

이중합을 6-벡터로 만들지 않는다. $(u\times v)_x=\varepsilon_{xyz}u^yv^z$이므로 모멘트
텐서를 먼저 만든다.

$$
T^{yz}_{i,c}[a,a']=\sum_{j,l}b_{ijl,c}\,a^y_{ij}\,a'^z_{il}\in\mathbb R^{3\times3}
\ \Longrightarrow\
\sum_{j,l}b_{ijl,c}\,(a_{ij}\times a'_{il})_x=\varepsilon_{xyz}T^{yz}_{i,c}[a,a'] .
$$

Force slot은 $\varepsilon(T[f,f])$이고, moment slot은

$$
\sum_{j,l}b\,(f_j\times m_l-f_l\times m_j)
=\varepsilon\bigl(T[f,m]\bigr)+\varepsilon\bigl(T[m,f]\bigr),
$$

둘째 항의 부호는 $\varepsilon_{xyz}T^{yz}[m,f]=\sum b\,(m_j\times f_l)_x=-\sum b\,(f_l\times m_j)_x$
에서 나온다. 이 경로에서 최대 중간 텐서는 $[B,N,k,C_b,3]$이지만, 계수 텐서
$[B,N,k,k,C_b]$가 비용을 지배하므로 $C_b\le4$부터 시작한다.

### C.2 AIRM의 scale 민감도

$K=sK_{gt}$이면 $A=L_{gt}^{-1}(sK_{gt})L_{gt}^{-\top}=sI$이므로 고윳값이 전부 $s$이고

$$
d=\Bigl(\sum_{i=1}^{6}\log^2 s\Bigr)^{1/2}=\sqrt6\,|\log s| .
$$

$s=e^{\pm1}$이면 $d\approx2.45$로, §5.5의 잔차 대역과 같은 크기다. 이것이 §2.6의 전역
스케일 파라미터를 별도로 둔 이유다 — 이 방향의 오차는 표현력과 무관하며, 그것만으로
손실 대부분을 설명할 수 있다.

### C.3 Wendland 창의 경계 성질

$\phi(q)=(1-q)^4(1+4q)$에 대해

$$
\phi(1)=0,\qquad
\phi'(q)=-4(1-q)^3(1+4q)+4(1-q)^4=-20q(1-q)^3
\ \Longrightarrow\ \phi'(1)=0 .
$$

값과 1차 도함수가 모두 소멸하므로, edge가 support 경계를 넘을 때 forward와 backward
모두 $O(1)$ 점프가 없다. 경계에 정확히 놓인 등거리 shell은 양쪽 모두 0을 기여하므로
top-$k$ tie 처리가 결과를 바꾸지 못한다.

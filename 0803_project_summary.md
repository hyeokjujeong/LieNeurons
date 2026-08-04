# Adjoint-Equivariant Stiffness Learning — 종합 정리 (2026-08-03)

발표 자료(`0803_blockage_experiments.pptx`) 발전용 소스 문서. 구성: ① 구현 개요 →
② baseline(train_report) 세팅·결과 → ③ 결과 분석과 원인 규명 → ④ 개선 구현 →
⑤ 추가 실험 세팅 → ⑥ 결과·분석 → ⑦ 추후 방향.

---

## 0. 배경 — 데이터 쌍은 어떻게 만들고, 지표는 무엇을 재는가

### 0.1 (point cloud, K) 쌍의 생성 — 합성 supervised regression

실측 강성 데이터가 아직 없으므로, **cloud가 주어지면 K가 결정되는 "정답 함수"를
정해 두고** 쌍 $(P_i,\ K_{gt}(P_i))$를 합성한다. 현 단계의 질문이 "실제 강성을
맞추는가"가 아니라 "**알려진 등변 함수를 이 모델 클래스가 흡수할 수 있는가**"
(표현력·최적화·등변성 검증)이기 때문에 합성 라벨이 적합하다.

1. **Cloud 샘플링**: 비등방 Gaussian 점들(표본마다 다른 모양)을 랜덤 SE(3) pose에
   배치. 시나리오 실험은 이 단계만 교체 (centro/c2/tetra/iid 생성기).
2. **라벨 $K_{gt}(P)$** — cloud의 결정론적 함수, 두 종류:
   - **analytic (contact-spring)** — 물리 모형: 각 점 $i$와 kNN 이웃 $j$ 쌍을
     "접촉 스프링"으로 본다. 스프링은 접촉선 방향의 순수 힘 wrench
     $w_{ij} = (d_{ij},\ r_i \times d_{ij})$를 만들고, 가까울수록 큰 가중치
     $e^{-\lVert d\rVert^2/2\sigma^2}$를 받는다. 선형 스프링들의 강성은 rank-1
     기여의 합이라는 고전 결과 그대로:
     $$K_{gt}(P) = \frac{1}{Nk} \sum_{ij} e^{-\lVert d_{ij}\rVert^2 / 2\sigma^2}\;
     w_{ij}\, w_{ij}^{\top}$$
     직관: "점들이 스프링으로 이어진 물체를 원점에서 밀고 비틀 때 느껴지는 6×6
     강성". 리프트가 coadjoint로 변환하고 가중치가 invariant라 **라벨이 정확히
     congruence-equivariant** — 흠결 없는 등변 함수다.
   - **teacher** — 같은 아키텍처의 frozen random 네트워크 출력을 라벨로. 물리적
     의미는 없지만 "정답이 모델 클래스 안에 있음"이 보장되는 통제군 — 최적화
     경로만 분리 검증한다.

### 0.2 지표 읽는 법 (friendly)

| 지표 | 무엇을 재나 | 직관적 눈금 |
|---|---|---|
| `train_d`, `val_d` | 예측 $K$와 정답 $K$의 SPD-거리 (AIRM, §1.3) | **고유값이 몇 배 어긋났나의 로그 척도**: $d{=}1$ ≈ 평균 1.5배, $d{=}2.3$ ≈ 2.6배 어긋남. baseline 정체 2.55 ≈ 강성이 방향 평균 ~2.8배 어긋난 상태 |
| `err_rel_ff/fm/mm` | $K$의 6×6을 3×3 블록별 상대오차로 분해 | 물리 대응 — **ff = 병진 강성**(밀었을 때 힘), **mm = 회전 강성**(비틀었을 때 모멘트), fm = 결합. "어느 물리 응답을 못 맞추는지"의 국소화. ff=1.000이면 그 블록 예측이 사실상 0. ⚠️ 분모(타깃 블록 norm)가 작으면 폭발 — 대칭 근방 fm이나 작은 N에서의 큰 값은 지표 결함 |
| `rank_pred` | 예측 $K$의 rank | 정상 6. **3이면 병진 강성 쪽 절반이 통째로 죽은 것** (rank-collapse 정리의 서명) |
| `f_signal` | 인코더 방향채널($f_c$)의 평균 크기 | **ff-블록을 배울 원재료의 양** — 차단 때문에 ff로 가는 정보는 이것뿐이다. 0이면 학습 원천 불가(대칭), 작으면 노이즈로 학습(큰 N) |
| `equiv_err` | 학습 후 $K(T{\cdot}P)$ vs $\mathrm{Ad}^{-{\top}}K\,\mathrm{Ad}^{-1}$ 잔차 | ~1e-15가 정상 (구조적 성질). $O(1)$이면 등변성 자체가 붕괴 — tetra의 kNN tie 문제가 이렇게 감지됨 |
| `clamped`, floor 47.86 | loss의 고유값 가드 발동 수 / 해석적 하한 | rank-collapse로 pencil 고유값 3개가 정확히 0 → 각각 $\log(10^{-12})$에 clamp → $d \ge \sqrt{3} \cdot 27.6 = 47.86$. **loss가 이 값에 붙어 있으면 "학습이 안 되는 게 아니라 수학적으로 불가능"하다는 서명** |

---

## 1. 현재 구현 개요

### 1.1 문제 설정

Point cloud $P$로부터 stiffness $K \in S^{6}_{++}$를 예측한다. $K$는 좌표계 변환
$T=(R,p)\in SE(3)$에 대해 **congruence**로 변환한다 (similarity가 아님):

$$K(T\cdot P) = \mathrm{Ad}_T^{-\top} K(P)\, \mathrm{Ad}_T^{-1}, \qquad
\mathrm{Ad}_T^{-\top}: f \mapsto Rf,\quad m \mapsto Rm + p\times Rf$$

wrench $(f, m)$에서 force는 translation-blind, moment가 $p$-결합을 흡수한다
(평행축 정리의 구조). 네트워크는 factor $L$을 출력하고 $K = LL^\top/C$로 조립 —
SPD by construction이며 congruence는 equivariance에서 자동으로 따라온다.
(역방향 $K \to \text{Cholesky} \to L$은 기저 순서 의존이라 equivariant하지 않음.)

### 1.2 두 Method와 완전 동치성

| | Method 1 (vector) | Method 2 (covector) |
|---|---|---|
| 인코더 | twist Plücker 리프트 $\xi = (n,\, r{\times}n)$, $\mathrm{Ad}$-equivariant | wrench 리프트 $W = (r{\times}n,\, n)$, $\mathrm{Ad}^{-\top}$-equivariant |
| 백본 | LNLinear + twist bracket | LNLinear + covector bracket $[F_1,F_2]_* = (f_1{\times}m_2 - f_2{\times}m_1,\ f_1{\times}f_2)$ |
| head | Klein $Y = QZ$ 후 Gram (여기서 타입 전환) | $Q$ 없이 바로 Gram |

두 방법은 Klein form $Q$(블록 스왑)의 conjugation으로 층별 정확 대응:
$\text{CovBackbone}(QV) = Q\,\text{Backbone}(V)$, $(QZ)(QZ)^\top = Q\,ZZ^\top Q$.
**실측: 같은 seed → 가중치 텐서 완전 동일 → 출력 차 2.5e-16, 학습 궤적까지 매
epoch 일치.** 이후 모든 실험은 method 선택과 무관하다.

### 1.3 Loss — AIRM geodesic distance

$$d(K_{gt}, K_{pred}) = \bigl\lVert \log(K_{gt}^{-1/2} K_{pred} K_{gt}^{-1/2}) \bigr\rVert_F
= \Bigl(\sum_i \log^2 \lambda_i\Bigr)^{1/2}$$

- SPD manifold의 affine-invariant 계량이 주는 minimal geodesic distance
  (Cartan–Hadamard: geodesic 유일). $\lambda_i$ = pencil의 일반화 고유값.
- **congruence 불변** $d(WAW^\top, WBW^\top) = d(A,B)$ $\forall W \in GL(6)$ →
  equivariant 모델과 합성 시 **학습 loss가 SE(3)-invariant** (실측 3.4e-13) —
  frame 정규화·회전 증강 불필요.
- Gradient 안전성: 거리가 고유값만의 함수 → `eigvalsh`-only 미분 경로로
  $1/(\lambda_i - \lambda_j)$ 고유벡터 항 원천 회피. naive matrix-log는 float32
  근접축퇴(gap 1e-7)에서 NaN, 본 구현은 ~5e-7 유지. 경계 barrier
  $\partial d^2/\partial\lambda = 2\log\lambda/\lambda$는 `eig_floor`로 제어
  (invariant 양에만 작용해 등변성 보존).
- Log-Euclidean·Bures-Wasserstein은 직교 congruence만 불변 → 부적합.

### 1.4 인코더 동작 상세 — Plücker lift + mean pooling, 그리고 문제의 근원

인코더는 세 단계다 (`encoders.py`):

1. **kNN 그래프**: 각 점 $i$의 $k$개 최근접 이웃을 거리 순위로 정렬 — 거리가
   SE(3)-invariant라 그래프·순위가 invariant (단, 거리 동률이 없다는 가정).
2. **쌍별 Plücker lift**: 점 $i$와 rank-$c$ 이웃의 차이 $d_{i,c}$에 대해
   $w_{i,c} = (f, m) = (d_{i,c},\ r_i \times d_{i,c})$ — "접촉선을 따라 작용하는
   순수 힘의 wrench". 한 줄 증명으로 $\mathrm{Ad}^{-\top}$-equivariant.
3. **mean pooling**: 채널 $c$별로 점 축에 대해 평균
   $$f_c = \frac{1}{N}\sum_i d_{i,c}, \qquad m_c = \frac{1}{N}\sum_i r_i \times d_{i,c}
   \qquad (\text{cloud} \to k\text{개의 요약 wrench})$$
   mean은 선형이라 equivariance가 공짜이고 순열 불변 — 하지만 **백본이 시작되기
   전에 압축이 끝난다**는 뜻이기도 하다.

**문제 — parity(홀짝) 분석.** 점반사 $P \to -P$에서 $d \to -d$ (**홀수**),
$m = r{\times}d \to (-r){\times}(-d) = m$ (**짝수**). mean은 선형이므로:

- $f_c$ = 홀수 함수의 평균 → 대칭 cloud에서 **정확히 상쇄** (rank-collapse의 재료 ①)
- 대칭이 없어도 $\lvert f_c \rvert \sim N^{-1/3}\cdot N^{-1/2} = N^{-5/6}$ 감쇠
  (kNN 거리 축소 × 평균 상쇄) — 조밀한 cloud일수록 ff-신호가 잡음화
- 반면 타깃의 $K_{ff} = \sum w\, dd^{\top}$는 $dd^{\top}$가 **짝수**(2차 모멘트)라
  대칭에서도 $O(1)$로 생존 — 모델이 못 따라가는 정확한 지점

blockage(§3.1) 때문에 $m$-측의 풍부한 정보로 이 손실을 보충할 수도 없으므로,
"1차 모멘트 mean pooling + bracket-only 백본"의 조합이 rank-collapse를 만든다.

### 1.5 주요 코드 (experiment/pc_se3_congruence/)

`encoders.py`(twist/wrench Plücker + VN 학습형), `models.py`(두 백본, GramGate,
GateBackbone, DualBackbone, heads), `spd_loss.py`(AIRM), `train.py`(baseline 학습),
`data_synth.py`(cloud 샘플링, contact-spring GT, 시나리오 생성기),
`blockage_bench.py`(시나리오×비선형성 벤치, wandb), `metrics.py`(블록 진단·wandb).

---

## 2. Baseline 실험 (train_report.md) — 세팅과 결과

### 2.1 세팅

| 항목 | 내용 |
|---|---|
| 아키텍처 | PlueckerEncoder(k=16) → [LNLinear+LN-Bracket]×5 → Klein head (채널 16-64-128-128-64-32, ~120k) |
| GT analytic | contact-spring $K = \sum k(\lVert d\rVert)\, w w^\top$, $w = (d, r{\times}d)$ — 구성상 정확히 equivariant한 "모델 밖" 함수 |
| GT teacher | frozen random 동일 아키텍처 — 실현 가능성 통제군 |
| 데이터 | 비등방 Gaussian cloud, N=128, train/val 4096/512, $\lVert p\rVert \sim 1$ |
| 학습 | Adam 1e-3 + cosine→1%, batch 64, 150 epochs, float64, grad-clip 1.0, 단일 시드 |

설계 의도: teacher = "최적화가 되는가", analytic = "모델 밖 등변 함수를 흡수하는가"의 분리.

### 2.2 결과

| target | train $d$ (ep1→150) | val $d$ | 판정 |
|---|---|---|---|
| teacher | 0.384 → **0.00207** (186×) | 0.00218 (gap 없음) | 최적화 경로 건강 |
| analytic | 6.63 → **1.55** (단조) | **2.55 정체** (ep~40 최저 2.45 후 미세 상승) | 표현력/표본 한계 |

- 학습 후에도 equivariance 2.7e-15, loss 불변성 1.2e-15 — 등변성은 학습과 직교.
- 고유값 클램프 0회 / NaN 없음 (9,600 steps).
- 부수 성과: `vee_*` dtype 잠복버그 발견·수정 (전역 default dtype으로 zeros 생성 →
  조용한 float32 다운캐스트 → 등변성 바닥 1e-8 고정).
- analytic 정체의 원인 후보(리포트 기재): ① bracket-only 백본의 구조 제약(blockage)
  ② 인코더 mean-pool 압축 ③ 표본 수 — **미분리 상태로 남김**.

---

## 3. 결과 분석 → 원인 규명 (이론)

### 3.1 Bracket Blockage — 병진 성분이 회전 슬롯에 영향 불가

bracket의 구조:

$$[\xi_1,\xi_2] = (\underbrace{\omega_1{\times}\omega_2}_{v \text{ 미등장}},\ \omega_1{\times}v_2 - \omega_2{\times}v_1)
\qquad
[F_1,F_2]_* = (f_1{\times}m_2 - f_2{\times}m_1,\ \underbrace{f_1{\times}f_2}_{m \text{ 미등장}})$$

- $\mathfrak{se}(3) = \mathfrak{so}(3) \ltimes \mathbb{R}^3$: 병진부가 **ideal** →
  몫사영 $\pi(\omega,v) = \omega$가 Lie 준동형 → LNLinear(채널 우측곱, 슬롯
  보존)와 bracket 모두 $\pi$와 가환 → **백본 전체가 가환**: $f$-계보는 입력
  $f$-슬롯만으로 닫힌다.
- 실측: $m$ 교란 → $f$-슬롯 출력 변화 **비트 단위 0.0** (다층 전체). $f \to m$
  역방향은 정상 전파 (일방향성).
- **Linear로도 탈출 불가**: equivariant slot-mixing map의 공간(commutant)을 수치로
  계산하면 정확히 2차원 $\{I,\ N: f \to m\}$ — $m \to f$ intertwiner는 존재하지
  않는다 (ideal 구조와 모순).
- 결과: **$K_{ff}$ 블록이 인코더 mean 방향벡터 $\{f_c\}$만의 함수로 타입 고정**.
  "정보 손실"이 아니라 출력 관측가능성 문제 ($f{\to}m$ 개방 + skip으로 내부 표현은
  정보 보존 — 그러나 그 블록으로 꺼낼 수 없음).

### 3.2 Rank-Collapse 정리와 실패 계층

- **정리**: 중심대칭 cloud($P = -P$)에서 $f_c = \frac1N\sum_i d_i^{(c)} = 0$ 정확히
  → 어떤 가중치·게이트로도 $K_{ff} \equiv 0$, $K_{fm} \equiv 0$, rank$(K) \le 3$.
  AIRM loss에는 **해석적 하한** $d \ge \sqrt3\,\lvert\log\varepsilon_{clamp}\rvert = 47.86$.
- **실패 계층** (중심대칭은 극단일 뿐):
  - $C_2$ 축 하나(기계부품에 흔함): $f_c \in \mathrm{Fix}(\Gamma)$ →
    rank$(K_{ff}) \le 1$ (고정 부분공간 논증)
  - 거울면: 법선 방향 blind (rank ≤ 2)
  - $A_4$ orbit: Fix=\{0\}이나 구조적 kNN tie가 개입 (tie-잡음 지배)
  - **대칭이 없어도**: $\lvert f_c\rvert \sim N^{-1/3}\cdot N^{-1/2} = N^{-5/6}$
    통계 소멸 (실측 $N^{-0.8}$) — 조밀한 스캔일수록 ff-신호가 잡음에 가라앉음
- ⇒ baseline analytic 정체의 유력 주원인 = **ff-표현력**, 근원은 인코더.

---

## 4. 개선 구현 — Invariant Gram Gate와 Dual Branch

- 스칼라(invariant)는 표현 제약 밖 → $m$-정보를 $f$-계수에 곱셈 주입하는
  **유일하게 남은 통로**. 일반형(GramGate):
  $$\nu(X)_c = x_c\,\phi\bigl(\mathrm{MLP}(S_{c,:})\bigr), \qquad
  S = X^\top Q^{-1} X \ (\text{invariant Klein-Gram})$$
  rank-1 bilinear $q_c^\top Q k_c$ (Coadjoint PDF 식 11)는 특수케이스.
- **설계 함정 (실측 발견)**: 슬롯별 gating은 $\mathrm{Ad}^{-\top}$가
  block-triangular라 $p \neq 0$에서 equivariance 0.48로 파괴 → **채널 전체
  (6성분)에 곱해야** 정확히 성립. 게이트는 bounded($1{+}\tanh$)만 — $Q$가
  indefinite라 null cone에서 $Q$-norm 정규화는 ill-posed.
- **DualBackbone**: bracket ∥ gate 병렬 브랜치 → 채널 concat → bracket 층 병합
  (Lie Neurons 논문 Appendix C, Fig. 5 패턴). bracket의 방향 생성력 + gate의
  $m{\to}f$ 주입 결합 의도.
- 세 백본 × 두 method 전 조합 equivariance 실측 ~1e-15 검증 후 학습 투입.

## 5. 추가 실험 세팅 — 실패 시나리오 벤치마크 (blockage_bench)

| 데이터셋 | 구성 | 이론 예측 |
|---|---|---|
| centro($\eta$) | $\{\pm a_i\} + \eta\cdot$noise | $\eta{=}0$: rank≤3, 하한 47.86 / $\eta$로 난이도 보간 |
| c2 | 180° 회전축 하나 (중심대칭 아님) | rank$(K_{ff})\le1$ |
| tetra | $A_4$ orbit (−I 없음) | tie-잡음 지배 |
| iid(N) | 대칭 없음, N=32/128/512 | $\lvert f_c\rvert \sim N^{-5/6}$ |

- **쌍 생성**: cloud 샘플링만 시나리오 생성기로 교체하고, **라벨은 전 시나리오
  공통으로 contact-spring $K_{gt}(P)$** (teacher는 벤치에서 미사용). 핵심 설계:
  contact-spring의 $K_{ff} = \sum w\, dd^{\top}$는 2차 모멘트라 대칭에서도 $O(1)$
  full-rank로 살아있고($dd^{\top}$는 $d\to-d$에 짝수), 붕괴하는 것은 모델의 도달
  가능 집합뿐(mean $f_c$는 홀수라 상쇄) — 실패를 온전히 모델 클래스에 귀속 가능.
- 레시피 = baseline 동일(§2.1), 데이터셋·비선형성(bracket/gate/dual)만 교체.
- 로깅(wandb `adjoint_equivariant_network/pc-se3-congruence`): $d$, **블록별 상대오차
  ff/fm/mm**(실패 국소화), rank$(K_{pred})$, $f$-signal(인코더 방향채널 norm — ff-계보의
  유일 입력), 학습 후 equivariance.
- eval-only `fiber`: 같은 $f$-요약·다른 타깃 쌍에서 모델 ff-예측 동일함 검증
  (실측: fc 격차 3e-17, 타깃 격차 1.73, 모델 격차 2e-36).

## 6. 벤치마크 결과 및 분석

### 6.1 대칭 계열 — 이론 예측의 정확한 재현

| dataset | bracket | gate | dual | 예측 |
|---|---|---|---|---|
| centro $\eta{=}0$ (N 32–512) | 47.89 / ff 1.000 / rank 3 | 동일 | 동일 | 하한 47.86 ✓ |
| c2 | 55.28 / ff 0.97 / rank 2 | 55.28 | 55.27 | rank$(K_{ff})\le1$ ✓ |
| tetra | $d$ 3.77, **equiv 1.0 ✗** | $d$ 20.7, equiv 1.1 ✗ | $d$ 3.48, equiv 1.0 ✗ | tie → 등변성 붕괴 |

- centro/c2: 세 비선형성이 자릿수까지 동일하게 하한 고정 — **게이트로도 불가
  ($g\cdot0=0$)** 3중 적중. 학습이 하는 일은 mm-잔차 다듬기뿐.
- **tetra 발견**: 학습 후 equivariance가 $O(1)$로 붕괴 — kNN tie가 순위 채널의
  불변성 자체를 깬다. 가중치가 아니라 **데이터 도메인 한계**: 고대칭 입력에서
  rank-채널 인코더는 등변성을 잃는다 → tie-robust 인코더 필요.

### 6.2 $\eta$·N sweep — 게이트의 실효는 "conditioning"

val_d (괄호: ff 오차):

| 데이터 | bracket | gate | dual |
|---|---|---|---|
| centro $\eta$=0.02 | 3.43 (0.63) | 3.35 (0.68) | 3.37 (0.58) |
| centro $\eta$=0.1 | 2.95 (0.58) | **2.52** (0.59) | 2.59 (0.51) |
| centro $\eta$=0.5 | 2.71 (0.48) | **2.32** (0.56) | 2.46 (0.52) |
| iid N=32 | 3.30 | 2.90 (0.95) | **2.75** (0.71) |
| iid N=128 (=baseline 세팅) | 2.57 (0.51) | **2.20** (0.59) | 2.26 (**0.47**) |
| iid N=512 | 2.18 (0.48) | 2.06 (0.57) | **2.04** (**0.44**) |

- **gate**: val 개선(최대 −0.44)하지만 **ff 오차는 오히려 소폭 상승** — 출처는
  mm 오차의 급감 (5.3→1.6, 2.2→0.86, 2.8→0.77). 게이트의 가치 = "차단 해소"가
  아니라 "invariant 채널 conditioning".
- **dual (최종 결과)**: iid에서 **유일하게 ff 오차를 낮춤** (bracket 0.51→0.47,
  0.48→0.44) + N=512에서 전체 최고 val **2.04**, N=32에서도 2.75로 최고 —
  bracket의 방향 생성력과 gate의 $m{\to}f$ 계수 주입이 진짜-신호 데이터에서
  시너지를 냄. 단 centro($f$-방향이 노이즈)에서는 과적합 열세 — 데이터 성격에
  따라 gate/dual 우위가 갈린다.
- ff 개선폭은 ~8%로 제한적 — ff-병목의 근본 해소는 여전히 인코더 몫.
- 재현성: iid N=128 bracket **2.57 ≈ baseline 2.55** — 하네스 정합 확인.
- 지표 결함 식별: 블록 상대오차의 분모 붕괴 (대칭 근방 fm 타깃 $\sim\eta$ → 오차
  129; iid N=32 ff 68.6) — 실패가 아니라 지표 문제, 분모를 $\lVert K_{gt}\rVert_F$
  전체로 통일 필요.

### 6.3 종합 해석

1. baseline analytic 정체(2.55)의 원인후보 ① blockage → **주원인으로 승격**:
   같은 레시피에서 ff-블록만 일관 정체(0.48–1.0), 대칭 극한에서 정확히 1.0,
   해석적 하한과 실측 일치.
2. 실험 구도 완성: teacher(실현가능 극한, 0.002) ↔ centro $\eta{=}0$(실현불가
   극한, 47.86) — 그 사이를 $\eta$(대칭 파편)·N(점 밀도)로 연속 보간.
3. ff-방향의 재료는 인코더 $\{f_c\}$뿐 — **방향을 새로 만들 수 있는 것은 인코더
   개선뿐**.

## 7. 추후 개선 방향

1. **[본질] 인코더 보강 — 2차 모멘트 도입.** mean pooling(1차 모멘트)이 홀수
   함수라 대칭·대수(N)에서 죽는 것이 근원이므로(§1.4), **pooling에 들어가는 항의
   parity를 짝수로 바꾸는** 두 가지 설계:

   **(A) Pooling-전 bracket (wrench 채널 유지 — 기존 백본에 바로 연결).**
   점 $i$에서 서로 다른 이웃 rank의 리프트끼리 bracket을 먼저 취하고 평균한다:
   $$f'_{(a,b)} = \frac{1}{N}\sum_i \bigl[\,w_{i,a},\ w_{i,b}\,\bigr]_*
   \qquad \text{f-출력} = \frac1N\sum_i d_{i,a}\times d_{i,b}$$
   $d \to -d$에서 $d_a{\times}d_b$는 부호가 **두 번** 뒤집혀 짝수 — 대칭에서
   상쇄되지 않는다. 기하적으로 $d_a{\times}d_b$는 국소 표면 법선 방향
   (pseudo-vector). bracket 출력은 여전히 coadjoint-equivariant wrench이므로
   출력 형식이 $\mathbb{R}^{6\times C'}$ 그대로라 **백본·헤드 수정 없이 인코더
   채널만 추가**하면 된다 (기존 1차 채널과 병행: 홀수+짝수 채널 혼합).

   **(B) 2차 모멘트 head (matrix-valued — 타깃을 클래스 안에 포함).**
   채널별 Gram $S_c = \frac1N\sum_i w_{i,c} w_{i,c}^{\top}$은 congruence-covariant
   ($S \mapsto A S A^{\top}$)이므로, invariant 게이트 가중으로 직접 조립:
   $$K(P) = \sum_c \mathrm{softplus}\bigl(g_c(\text{invariants})\bigr)\; S_c$$
   — SPD by construction + 정확히 equivariant. **contact-spring 타깃이 문자
   그대로 이 클래스의 원소**($g$ = 거리 가중)라 analytic 실험의 실현 가능성이
   보장된다. 대신 백본과의 결합엔 matrix-feature 층이 필요 (Coadjoint PDF의
   factorization 프레임과 접속).

   acceptance test 준비 완료: centro $\eta{=}0$에서 **하한(47.86) 소멸 + f-signal
   회복**이면 성공 — (A)는 짝수 채널이 살아남으므로 통과가 예측되고, 게이트는
   불통과가 이미 확인됨. iid N sweep에서 $N^{-5/6}$ 감쇠가 사라지는지도 부수 판정.

   **실측 결과 (2026-08-04, `BracketPlueckerEncoder` 구현·실행, 60 epochs):**

   | dataset | 기존 (전 구성 공통) | encoder=bracket | 판정 |
   |---|---|---|---|
   | centro $\eta{=}0$ | **47.89 고착** (하한), rank 3, f-sig 0 | **val 1.98**, rank 6, f-sig 0.085 | ✅ **하한 돌파 — acceptance PASS** |
   | c2 | 55.28 고착, rank 2 | **55.28 고착**, rank 2 (f-sig 0.078이나 축에 갇힘) | ✅ 차등 예측 적중 — proper 대칭은 벡터 채널로 불가 (stabilizer) |
   | iid N=512 | ff 0.475 (bracket) / 0.443 (dual, 150ep) | val 2.13, **ff 0.412** (60ep) | ✅ 역대 최저 ff 오차, 회귀 없음 |

   해석: 짝수-parity 채널이 improper 대칭($-I$)의 상쇄를 정확히 우회해 rank-collapse가
   해소된다. c2가 그대로인 것은 실패가 아니라 **이론의 추가 검증**: $C_2$는 proper
   회전이라 $SE(3)$ stabilizer가 모든 equivariant 벡터 채널을 축에 가둔다 — 이건
   matrix-채널(설계 B) 또는 late pooling만이 풀 수 있다. (centro run의 블록 상대오차
   폭발은 분모 붕괴 지표 결함 — AIRM $d$가 신뢰 지표.)
2. **[견고성] tie-robust 인코더** — tetra 등변성 붕괴의 해결 (soft-rank·거리
   가중·tie-invariant pooling). 고대칭 실물 부품에 그대로 해당.
3. **[학습]** dual 정규화(weight decay, 표본 8192↑), 다중 시드 3–5개로 gate
   개선폭(−0.4)의 유의성 확정, val 최저점(early-stop) 추적.
4. **[지표]** 블록 오차 분모 통일, 절대 오차 병기.
5. **[확장]** reciprocal-product 게이트 $\langle w, \ell\rangle$
   (adjoint·coadjoint 혼합 타워의 인코더 수준 invariant), Fréchet mean 기반
   다중 측정 융합.

---

### 참조 문서 (상세)

`bracket_blockage_analysis.md`(차단 이론+실측), `spd_geodesic_loss.md`(loss 상세),
`blockage_bench_report.md`(벤치 상세), `covector_ln_framework.md`(covector 프레임워크),
`wrench_pluecker_lift.md`(리프트 이론), `train_report.md`(baseline),
`Coadjoint_Equivariant_Network.pdf`(이론 draft).

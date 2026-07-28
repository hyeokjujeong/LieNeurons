# Covector Lie Neurons: Point Cloud → Stiffness $K = LL^\top$ Framework

**목표.** Compliance control을 위해 point cloud로부터 stiffness $K$를 예측한다.
$K$는 SE(3) congruence-equivariant 해야 하고, $K = LL^\top$로 분해했을 때 네트워크 출력 $L$이

$$f(T \cdot PC) = \mathrm{Ad}_T^{-\top} \, f(PC), \qquad \forall\, T = (R, p) \in SE(3)$$

를 만족해야 한다. Lie Neurons는 vector(twist, $\mathrm{Ad}$-equivariant) 입력용이므로,
이를 covector(wrench, $\mathrm{Ad}^{-\top}$-equivariant)에 **일대일 대응**시킨 framework를 정리한다.

> **표기.** wrench(covector)는 moment-first $F = (m, f) \in \mathfrak{se}(3)^*$,
> twist는 $\xi = (\omega, v) \in \mathfrak{se}(3)$.
> 목표 표현(coadjoint):
> $$\mathrm{Ad}_T^{-\top} = \begin{bmatrix} R & \hat{p}R \\ 0 & R \end{bmatrix}
> \quad\Longleftrightarrow\quad f \mapsto Rf, \quad m \mapsto Rm + p \times Rf.$$
> Killing form은 $\mathfrak{se}(3)$에서 degenerate이므로 사용하지 않고, nonlinearity는 Lie bracket만 사용한다.

---

## 1. 문제 1 — Lie bracket은 covector 연산이 아니다. 직접적 해결책은?

### 1.1 해결책: covector-native bracket (closed form, $Q$ 없음)

$$\boxed{\;[F_1, F_2]_* \;=\; \bigl(\, f_1 \times m_2 - f_2 \times m_1,\;\; f_1 \times f_2 \,\bigr)\;}$$

twist bracket과 나란히 놓고 보면 **슬롯 역할만 교환된 같은 공식**이다:

| | angular-슬롯 출력 | linear-슬롯 출력 |
|---|---|---|
| twist bracket $[\xi_1,\xi_2]$ | $\omega_1 \times \omega_2$ | $\omega_1 \times v_2 - \omega_2 \times v_1$ |
| covector bracket $[F_1,F_2]_*$ | $f_1 \times m_2 - f_2 \times m_1$ | $f_1 \times f_2$ |

대응 사전: $\omega \leftrightarrow f$, $v \leftrightarrow m$.

### 1.2 왜 성립하는가 — coadjoint는 adjoint의 거울상

$$\mathrm{Ad}_T = \begin{bmatrix} R & 0 \\ \hat{p}R & R \end{bmatrix} \;(\omega \text{는 } p\text{에 blind}),
\qquad
\mathrm{Ad}_T^{-\top} = \begin{bmatrix} R & \hat{p}R \\ 0 & R \end{bmatrix} \;(f \text{가 } p\text{에 blind}).$$

twist에서 $\omega$가 하던 역할(translation-blind, coupling의 소스)을 wrench에서는 $f$가 한다.
그래서 $[\cdot,\cdot]_*$는 $\mathrm{Ad}^{-\top}$에 대해 정확히 equivariant하다.
구현은 cross product 4번이 전부이며 $Q$ 행렬이 코드에 등장하지 않는다.

### 1.3 반드시 알아야 할 세 가지

1. **유일성.** $\mathfrak{se}(3)^*$ 위의 equivariant bilinear map 공간은 정확히 **2차원**이다
   ($[\cdot,\cdot]_*$ 와 $N_*:(m,f)\mapsto(f,0)$; repo 리포트 실험 C7에서 수치 확인).
   즉 "직접적인 해결책"은 이것뿐이고, 제3의 선택지는 존재하지 않는다.
2. **정직한 구조적 사실.** $[\cdot,\cdot]_*$가 존재하는 근본 이유는 표현으로서
   $\mathfrak{se}(3) \cong \mathfrak{se}(3)^*$ (Klein form이 그 동형사상)이기 때문이다.
   $Q$가 제거된 것이 아니라 공식 안으로 **컴파일**된 것. 다만 구현·물리 해석
   (screw theory: wrench도 screw, force part가 direction 역할) 관점에서는 완전히 covector-native이다.
   원했던 "일대일 대응"이 정확히 이 동형사상이다.
3. **twist bracket을 covector에 그대로 쓰면 안 된다.** $p = 0$에서는 통과하지만
   ($\mathrm{Ad}_{(R,0)}$은 직교라 두 표현이 일치) translation이 들어가는 순간 $O(1)$로 깨진다
   (리포트 실험 C4).

> **코드.** 이미 구현·검증 완료:
> `models.py:121 covector_bracket`, `models.py:140 LNCovectorBracket` (skip connection 포함),
> `models.py:164 CovectorBackbone` — float64에서 모든 translation scale에 대해 ~5e-16 (실험 C5).

---

## 2. 문제 2 — PC 입력은 covector가 아니다. 어떻게 넣는가?

### 2.1 해결책: pure-force wrench lifting

점 $r$, 방향 $n$에 대해 **처음부터 wrench로 lift**한다:

$$\boxed{\;W(r, n) \;=\; (\,m,\; f\,) \;=\; (\,r \times n,\;\; n\,)\;}$$

물리적 의미: 점 $r$을 지나고 방향 $n$인 직선을 따라 작용하는 **단위 힘의 origin 기준 wrench**.

### 2.2 증명 (한 줄)

$T = (R, p)$ 하에서 $r' = Rr + p$, $n' = Rn$이면:

$$f' = Rn = Rf, \qquad
m' = (Rr + p) \times Rn = R(r \times n) + p \times Rn = Rm + p \times Rf$$

$$\therefore\; W(T \cdot (r, n)) = \mathrm{Ad}_T^{-\top} W(r, n) \qquad \blacksquare$$

### 2.3 핵심 관찰

- 기존 `PlueckerEncoder`(`encoders.py:31`)와 **숫자가 완전히 동일, 슬롯 순서만 교환**.
  Plücker 직선 좌표는 twist(zero-pitch screw)로 읽으면 $\mathrm{Ad}$-equivariant,
  wrench(직선을 따르는 순수 힘)로 읽으면 $\mathrm{Ad}^{-\top}$-equivariant이다.
- direction field $n$의 조건은 기존과 동일: **translation-invariant + SO(3)-equivariant**.
  - 옵션 A: kNN pairwise difference $d_{ij} = r_j - r_i$ (closed form, 파라미터 없음)
  - 옵션 B: 학습형 VN-DGCNN (둘 다 `encoders.py`에 이미 있음)
- anchor-transport 보정(리포트 Lemma 2)도 같은 형태:
  wrench 좌표에서 $\mathrm{Ad}_{(I,c)}^{-\top}: (m,f) \mapsto (m + c \times f,\; f)$ —
  twist 때와 같은 연산이라 코드 수정은 슬롯 교환뿐이다.
- 결과: **파이프라인 전체가 처음부터 끝까지 covector 공간에서 진행**되며,
  중간에 twist ↔ wrench 변환이 한 번도 없다.

---

## 3. 전체 Framework — Lie Neurons ↔ Covector Lie Neurons 대응 사전

논문 (arXiv:2310.04521) 의 각 구성요소를 $\mathfrak{g}^*$ 버전으로 옮긴 표:

| Lie Neurons (논문, $\mathfrak{g}$, $\mathrm{Ad}$) | Covector 버전 ($\mathfrak{g}^*$, $\mathrm{Ad}^{-\top}$) | 비고 |
|---|---|---|
| 입력: twist $\xi=(\omega,v)$, $x \in \mathbb{R}^{6\times C}$ | 입력: wrench $F=(m,f)$, $x \in \mathbb{R}^{6\times C}$ | |
| Hat/Vee (§3.1) | 불필요 | bracket이 closed form이라 행렬 표현 자체가 필요 없음 |
| **LN-Linear** $xW$ (식 8) | **그대로 사용** | 오른쪽 곱은 어떤 왼쪽 표현과도 commute (실험 C4b 검증) |
| **LN-Bracket** $x + [(xU)^\wedge,(xV)^\wedge]^\vee$ (식 12) | $x + [xU,\, xV]_*$ | skip connection 구조 동일, bracket만 교체 (`LNCovectorBracket`) |
| **LN-ReLU** (Killing 기반, 식 10) | **사용 불가** | covector라서가 아니라 $\mathfrak{se}(3)$ 자체의 문제. dual의 invariant form은 $f_x \cdot f_d$인데 $m$에 완전히 blind (radical = moment ideal). bracket-only가 정답 |
| **LN-Mix** (§4.3) | 해당 없음 | 논문에서도 $\mathfrak{so}(n)$ 전용 ($\mathrm{Ad}$의 직교성 필요) |
| **Mean pooling** (§4.4) | 그대로 사용 | 선형 연산 |
| **Max pooling** (Killing score, 식 16) | score를 $f_n \cdot d_n^{(f)}$로 대체 가능하나 같은 degeneracy 경고 | 안전한 대안: mean pooling, 또는 invariant weight ($f_i \cdot f_j$) attention pooling |
| **LN-Invariant** $B(x,x)$ (식 17) | 채널별 invariant는 정확히 2개: $\|f\|^2$ (degenerate Killing형), $m \cdot f$ (screw pitch형, 완전 invariant) | invariant 이차형식 공간이 2차원이라 이 둘이 전부 |

### 3.1 아키텍처

```
P = {r_i}  ──(§2: W(r,n) = (r×n, n))──▶  R^{6×C₀}
           ──(§1: LN-Linear + [·,·]* blocks)──▶  Z ∈ R^{6×C}
           ──(head)──▶  L = Z/√C,   K = LLᵀ
```

### 3.2 Head — $L$이 곧 출력이다

마지막 wrench feature $Z \in \mathbb{R}^{6 \times C}$ ($C \ge 6$)에 대해:

$$L := Z / \sqrt{C}, \qquad K := LL^\top$$

- $Z \mapsto \mathrm{Ad}_T^{-\top} Z$이므로 $L$ 자체가 목표 법칙
  $f(T \cdot PC) = \mathrm{Ad}_T^{-\top} f(PC)$를 **정확히** 만족한다.
  별도의 분해 절차가 없다.
- $K \mapsto \mathrm{Ad}_T^{-\top} K \,\mathrm{Ad}_T^{-1}$ (congruence)이 자동으로 따라온다.
  $K$는 construction상 symmetric PSD, $C \ge 6$이면 generically SPD.
- ⚠️ **순서 주의: $K$를 먼저 만들고 Cholesky로 $L$을 뽑으면 안 된다.**
  Cholesky는 기저 순서에 의존해 equivariant하지 않다.
  "$L$을 출력 → $K = LL^\top$" 방향이 유일하게 올바른 순서.
- Compliance control: $C_{\text{comp}} = K^{-1}$은 자동으로 twist-type 법칙
  $C_{\text{comp}} \mapsto \mathrm{Ad}_T \, C_{\text{comp}} \mathrm{Ad}_T^\top$를 만족
  → $\delta\xi = C_{\text{comp}} F$의 타입이 맞는다.

---

## 4. 실전 주의사항

| # | 금지/주의 | 이유 | 근거 |
|---|---|---|---|
| 1 | $K + \varepsilon I$ 금지 | $\varepsilon I$는 equivariance를 $O(1)$로 깨뜨림. conditioning이 필요하면 equivariant한 두 번째 출력 $M(PC)$로 $K + \varepsilon M(PC)$, 또는 loss 레벨 처리 | negative control B5 |
| 2 | LN-Linear에 bias 금지 | 회전에 대해서도 깨짐 ($p=0$에서도 실패) | negative control A4 |
| 3 | 검증은 반드시 $p \neq 0$에서 | 모든 타입 오류($\mathrm{Ad}$ vs $\mathrm{Ad}^{-\top}$ 혼동)는 $p=0$에서 안 보임. `verify.py`의 sweep $\|p\| \in \{0, 1, 10^2, 10^4\}$ + negative control 프로토콜 재사용 | 리포트 §6 |

---

## 5. 구현 현황 및 남은 일

**이미 있는 것** (`experiment/pc_se3_congruence/`, float64 검증 완료):

- `covector_bracket`, `LNCovectorBracket`, `LNLinearAndCovectorBracket`, `CovectorBackbone`, `ModelCNative` — covector **입력**용 백본 전체

**새로 필요한 것:**

1. `WrenchPlueckerEncoder` — 기존 `PlueckerEncoder` 출력을 wrench 해석 $(m, f) = (r \times n,\ n)$으로 슬롯 배치 (몇 줄 수정)
2. 학습형 인코더의 wrench 버전 — direction field는 그대로, lift 부분만 같은 슬롯 교환
3. End-to-end 모델: `WrenchPlueckerEncoder` → `CovectorBackbone` → Gram head ($L$, $K$)
4. `verify.py`에 end-to-end 검증 케이스 추가 (translation sweep + negative controls)

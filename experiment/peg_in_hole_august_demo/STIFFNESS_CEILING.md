# 축 강성 상한 — 원인 분석과 해결

`MODEL_CARD.md` §7이 보고한 "축 병진 강성이 라벨 500 대비 ~310–355에서 포화"를
끝까지 추적하고 고친 기록. **상태: 해결됨.**

- 대상: 커밋 `0961b07`의 `khat_pointwise.pt` (channels 16-64-64-32, factors 32)
- 라벨: `K_body = diag(30,30,30, 30,30,500)`, `[m; f]` 순서
- 해법: `pitch='head'` (§9). 재현: [부록 A](#부록-a-재현-스크립트) · [부록 C](#부록-c-도구와-실행법)

---

## 0. 요약

**원인.** 이 아키텍처의 모든 latent covector는 자기 점 $p_i$ 를 지나는
zero-pitch line wrench로 갇혀 있고(정리 1), 그 결과 출력 $K$ 는
**"표면에 밀기만 하는 스프링을 꽂아 만들 수 있는 강성"** 의 집합
$\mathcal{K}(P) = \{\sum_i J_i^\top C_i J_i : C_i \succeq 0\}$ 을 벗어날 수 없다.
라벨 `diag(30,…,500)` 은 이 집합 **밖**이다 — Ø63 구멍이 뚫린 납작한 와셔에서
축 방향으로만 16.7배 뻣뻣하려면 **당기는(접착) 접촉**이 필요한데
$\beta = \mathrm{softplus} > 0$ 이 그것을 금지한다 (§8.5).

capacity도, 데이터량도, 학습 스케줄도 원인이 아니다. $C_i$ 는 3×3이라 스프링
3개로 포화하므로 채널·factor를 늘려도 $\mathcal{K}(P)$ 는 넓어지지 않는다 (§2.3).

**해법.** 등변 선형 사상의 대수 $\mathrm{End}_{SE(3)}(\mathfrak{se}(3)^*)$ 는
2차원 $\mathrm{span}\{I, N\}$ 인데 `LNLinear` 가 $I$ 성분만 쓰고 있었다 (정리 3).
빠진 $N(m,f) = (f,0)$ 을 head에 넣으면 — 가중치 행렬 하나, zero-init — 도달
집합이 **전체 SPD cone** 이 된다 (정리 4). 등변성은 $N$ 이 intertwiner라 정확히
보존된다.

**결과.**

| | 원래 (`pitch=none`) | 패치 후 (`pitch=head`) |
|---|---|---|
| 학습 표본 3개의 AIRM `d` | 0.4284 / 0.5501 / 0.5965 | **0.0012 / 0.0013 / 0.0013** |
| 축 강성 (라벨 500) | 355.3 / 327.5 / 312.4 | **500.3 / 500.3 / 500.3** |
| 실제 파이프라인 `train/d` | — | **0.00016** (§9.5) |
| 등변성 잔차 | ~1e-11 | **1.39e-11** |

**핵심 부등호: $0.0012 \ll 0.1934$.** 0.1934는 `pitch=none` 이 가중치를 어떻게
잡아도 넘을 수 없는 구조적 하한($C_i$ 를 자유 최적화해 얻은 값, §4.1)이다.
패치가 그보다 **160배 아래**로 갔다 — "학습이 더 잘 됐다"가 아니라 도달 가능
집합 자체가 바뀌었다는 증거다.

**부수적으로 밝혀진 것.** 원래 커밋은 **자기 학습 표본 3개조차 맞추지 못했고**
(train `d` ≈ 0.43–0.60), 기록된 `val_d 0.5250` 은 학습 구름에서 잰 값과 소수점
4자리까지 같다 — 원래 val 셋이 학습 구름의 **pose 사본**이었다는 뜻이다.
모델이 정확히 등변이고 AIRM이 congruence 불변이므로 그 val은 일반화를 전혀
재지 않았다 (§10).

---

## 1. 1단계 — 이 모델이 출력할 수 있는 `K`는 무엇인가

### 1.1 wrench와 pitch

이 코드베이스의 feature는 se(3)\* 원소, 즉 **wrench** $z = (m, f)$ 다
(`[m; f]` 순서: 앞 3개가 모멘트, 뒤 3개가 힘).

한 점 $p$ 를 지나는 직선을 따라 힘 $f$ 를 가하면 원점 기준 모멘트는
$m = p \times f$ 다. 이런 wrench를 **line wrench** 또는 **zero-pitch wrench**
라고 부른다. pitch는 $\dfrac{m \cdot f}{|f|^2}$ 로 정의되는데,
$m = p \times f$ 이면 $m \cdot f = (p\times f)\cdot f = 0$ 이므로 pitch가 0이다.

일반적인 wrench는 pitch가 0이 아니다 (힘 + 그 힘축 방향 토크 = 나사 운동).
**6차원 se(3)\* 중 zero-pitch wrench는 5차원 부분집합**(Plücker 조건 하나)에
불과하고, 그중에서도 "점 $p$ 를 지나는" 것만 보면 $f$ 로 매개되는 **3차원**이다.

### 1.2 모든 층이 zero-pitch를 보존한다

이 아키텍처에서는 **모든 latent covector가 자기 점 $p_i$ 를 지나는 line
wrench로 갇혀 있다.** 층별로 확인한다.

**(a) 입력 — `pointwise_models.py:228-236`**

```python
d = graph.edge_vec                                   # f = p_j - p_i
m = torch.cross(P.unsqueeze(2).expand_as(d), d, ...) # m = p_i x f
return torch.cat([m, d], dim=-1)
```

태생부터 $p_i$ 를 지나는 선이다.

**(b) set pooling** — 불변 스칼라 가중치 $a_j$ 로 이웃 합:

$$\sum_j a_j (p_i \times f_j,\; f_j) = \Big(p_i \times \sum_j a_j f_j,\; \sum_j a_j f_j\Big)$$

$p_i$ 가 합 밖으로 빠져나온다. 여전히 $p_i$ 를 지나는 선.

**(c) `LNLinear`** — 같은 점의 채널만 섞는다. (b)와 같은 이유로 보존.

**(d) `covector_bracket` — `models.py:155-170`**

$$[a, b]_* = \big(f_a \times m_b - f_b \times m_a,\;\; f_a \times f_b\big)$$

$m_a = p\times f_a$, $m_b = p\times f_b$ 를 넣고 BAC-CAB
($x\times(y\times z) = y(x\cdot z) - z(x\cdot y)$)을 적용하면

$$
\begin{aligned}
f_a \times (p \times f_b) &= p(f_a\cdot f_b) - f_b(f_a\cdot p) \\
f_b \times (p \times f_a) &= p(f_b\cdot f_a) - f_a(f_b\cdot p) \\[2pt]
\text{차} &= f_a(p\cdot f_b) - f_b(p\cdot f_a) \;=\; p \times (f_a \times f_b)
\end{aligned}
$$

즉 새 모멘트가 정확히 $p \times (\text{새 힘})$ 이다. **브래킷은 zero-pitch를
정확히 보존한다.** (이건 우연이 아니다 — 한 점을 지나는 line wrench들은
se(3)\*의 부분대수를 이룬다.)

**(e) `KleinGate`** — 6성분 전체에 스칼라를 곱한다. 보존.

**(f) `message_passing`** — **유일하게 보존하지 않는다.** 기본값이 `False`다.

### 1.3 수치 확인

체크포인트에서 head의 coframe $L$ 열 4096개를 직접 뽑아 확인:

```
hole62: N=128 H=32  max |m - p x f| / |m| = 2.98e-14
hole63: N=128 H=32  max |m - p x f| / |m| = 5.10e-14
hole64: N=128 H=32  max |m - p x f| / |m| = 5.07e-14
```

근사가 아니라 **닫힌 구조**다. 가중치를 어떻게 바꿔도 벗어날 수 없다.

### 1.4 정확히 무엇에 갇혀 있나 — 점마다 3차원 Lagrangian 부분대수

두 집합을 구분해야 한다. 흔한 오해는 "zero-pitch니까 5차원"인데, **실제 제약은
훨씬 강하다.**

**(a) 모든 zero-pitch wrench** $\{(m,f) : m\cdot f = 0\}$ — R⁶ 안의 5차원이지만
제약이 **이차식**이므로 선형 부분공간이 아니다 (Klein quadric cone).
**덧셈에 닫혀 있지 않다**: 서로 다른 점을 지나는 두 line wrench를 더하면
pitch가 생긴다.

**(b) 고정된 점 $p$ 를 지나는 line wrench**

$$\mathcal{L}_p = \{(p\times f,\; f) : f \in \mathbb{R}^3\} \;\subset\; \mathfrak{se}(3)^*$$

$f$ 로 매개되는 **3차원 선형 부분공간**이다. **실제로 갇혀 있는 곳은 여기다.**

(b)가 (a)보다 결정적인 이유: `LNLinear`는 같은 점의 채널을 **선형결합**한다.
(a)는 덧셈에 안 닫혀 있어 층 통과를 설명하지 못하지만, (b)는 선형 부분공간이라
**선형결합에 완전히 닫혀 있다.**

$\mathcal{L}_p$ 의 세 성질이 이 문서의 모든 현상을 설명한다:

| 성질 | 결과 |
|---|---|
| **선형 부분공간** (3-dim) | `LNLinear`로 벗어날 수 없음 |
| **Lie 부분대수** — §1.2(d)에서 $[\cdot,\cdot]_*$ 에 닫힘을 증명. $f_a \times f_b$ 가 곱이므로 $\mathfrak{so}(3)$ 와 동형 = "$p$ 주위의 회전들" | 브래킷 층으로도 벗어날 수 없음 |
| **Klein form에 대해 isotropic (Lagrangian)**: $\langle a,b\rangle = (p\times f_a)\cdot f_b + (p\times f_b)\cdot f_a \equiv 0$ | `klein_pair`가 항등적으로 0 — `peghole_training_report.md` §3.7의 zero-pitch degeneracy가 바로 이것 |

마지막 줄이 `use_force_invariant=True`가 왜 **필수**인지도 설명한다. Klein form은
$\mathcal{L}_p$ 위에서 완전히 퇴화하고, `force_pair` ($f_a\cdot f_b$)는
$\mathcal{L}_p$ 위에서 비퇴화다. gate/β MLP로 가는 스칼라 트랙이 살아남는
유일한 통로다. (이전 실험: 동일 조건 N=4 overfit에서 `use_force_invariant`
off/on이 `d = 0.9399` vs `0.0012`, **780배**.)

서로 다른 점을 섞을 때만 — 즉 `message_passing`만 — $\mathcal{L}_{p_i} +
\mathcal{L}_{p_j}$ 가 되어 (a)의 5차원 곡면조차 벗어나 6차원 전체로 나간다.

---

## 2. 2단계 — 그래서 `K`는 "점접촉 스프링 다발"이다

head는 (`pointwise_models.py:561-575`)

$$K = \sum_{i,h} \beta_{ih}\, z_{ih} z_{ih}^\top, \qquad \beta = \mathrm{softplus}(\cdot) > 0$$

이고, §1에서 $z_{ih} = (p_i \times f_{ih},\; f_{ih})$ 임을 확인했다.

### 2.1 virtual work 해석

twist $\xi = (\omega, v)$ 와 wrench의 pairing은 $\langle \xi, z\rangle = \omega\cdot m + v\cdot f$ 다.
$m = p\times f$ 를 넣으면

$$\omega\cdot(p\times f) + v\cdot f = f\cdot(\omega\times p) + f \cdot v = f \cdot \underbrace{(v + \omega\times p)}_{V(p_i)}$$

$V(p_i)$ 는 **강체가 twist $\xi$ 로 움직일 때 점 $p_i$ 의 속도**다. 따라서

$$\boxed{\;\xi^\top K \xi = \sum_{i,h} \beta_{ih}\,\big(f_{ih}\cdot V(p_i)\big)^2\;}$$

이건 정확히 **접촉점 $p_i$ 마다 방향 $f_{ih}$, 강성 $\beta_{ih}$ 인 선형 스프링을
붙인 다발**의 강성이다. 물리학 교과서의 grasp/contact stiffness 그 자체다.

### 2.2 행렬 형태

$V(p) = v + \omega\times p = [-[p]_\times \;\; I]\,\xi \equiv J_i \xi$ 로 두면
$z = J_i^\top f$ 이고

$$\boxed{\;K = \sum_i J_i^\top C_i J_i, \qquad C_i = \sum_h \beta_{ih} f_{ih}f_{ih}^\top \succeq 0\;}$$

$C_i$ 는 점 $p_i$ 의 **3×3 접촉 강성**이다.

### 2.3 왜 capacity가 무관한가

$C_i$ 는 3×3 PSD이므로 **rank 3이면 이미 최대**다. `factors=32 ≥ 3` 이면
채널·factor를 아무리 늘려도 $\{C_i\}$ 가 표현할 수 있는 집합은 넓어지지 않는다.
도달 가능한 $K$ 의 집합은 **오직 점 위치 $\{p_i\}$ 로만 결정된다.**

이것이 `MODEL_CARD.md`가 관측한 "capacity 무관"의 정확한 이유다. 채널·factor를
늘리는 것은 "같은 점에 스프링을 더 꽂는 것"인데, 한 점의 접촉 강성 $C_i$ 는
스프링 3개로 이미 포화한다. `factors=32`는 10배 과잉이고, 더 꽂아도 새로
도달하는 $K$ 가 없다.

### 2.4 직관 — "spring bundle"이 무슨 뜻인가

수식 대신 그림으로:

> 표면 샘플 점 $p_i$ 마다 **선형 스프링을 몇 개씩 꽂아 놓았다.** 스프링 하나는
> 방향 $f$ 와 강성 $\beta$ 를 갖는다. 물체가 twist $\xi$ 로 움직이면 점 $p_i$ 는
> $V(p_i) = v + \omega\times p_i$ 만큼 움직이고, 스프링은 **자기 방향 성분만**
> 느끼므로 늘어난 길이가 $f\cdot V(p_i)$, 에너지가 $\tfrac12\beta(f\cdot V(p_i))^2$.

전부 더한 것이 §2.1의 식이다. 즉 모델이 뱉는 $K$ 는 **"이 점들에 이 스프링들을
꽂았을 때 실제로 측정될 강성"**이고, 네트워크가 학습하는 것은 *각 점에 어느
방향으로 얼마나 센 스프링을 꽂을지*뿐이다. 그 외의 자유도가 없다.

**테이블 비유.** 탁자를 아래로 누르기 어렵게 만들려면(축 강성 500) 다리를
튼튼히 해야 한다. 그런데 다리를 **바깥에 넓게** 놓으면 기울이기도 같이
어려워진다(회전 강성). "누르기는 어렵고 기울이기는 쉬운" 탁자를 만들려면 다리를
**한가운데 기둥 하나로** 모아야 한다.

이 부품은 한가운데에 **Ø60 구멍**이 뚫려 있다 — 기둥을 세울 자리가 통째로
없다. 그리고 스프링은 강성을 더할 수만 있고 뺄 수 없으니(§3.1), 바깥 다리가
만든 회전 강성을 지울 방법이 없다.

---

## 3. 3단계 — 왜 상한이 생기는가 (핵심)

### 3.1 상쇄가 불가능하다

$K = \sum_i J_i^\top C_i J_i$ 에서 **모든 항이 PSD이고 부호가 양수뿐**이다.
$\beta = \mathrm{softplus} > 0$ 이라 음의 기여가 없다.

일반 신경망이라면 "어떤 점의 원치 않는 기여를 다른 점이 빼서 없앤다"가
가능하지만, **여기서는 원리적으로 불가능하다.** 어떤 접촉이 만들어낸 회전
강성은 절대 지울 수 없고, 오직 더할 수만 있다.

이것이 상한의 근원이다.

### 3.2 예산 계산

canonical 좌표에서 이 부품의 실측 기하 (부록 A):

```
scale = 66.7 mm,  bbox = 90 x 90 x 50 mm
r_min = 0.446  (= 29.8 mm)   <- 축에서 가장 가까운 표면점
r_max = 0.957
```

**구멍 반지름이 30 mm** (Ø60), 블록 반높이가 25 mm. 즉 이 부품은 기하학적으로
**납작하고 구멍이 큰 와셔**다. 축 근처에는 점이 하나도 없다.

목표: $K_{ff} = \mathrm{diag}(30,30,500)$, $K_{mm} = 30 I$ (trace 90), $K_{mf}=0$.

축 강성 500을 만드는 방법은 두 가지뿐이고 **둘 다 실패한다.**

**전략 A — 축 방향 힘 ($f = e_z$)**

점 $p=(r,0,h)$ 에서 $f = e_z$ 면 $p \times e_z = (0,-r,0)$, 즉 모멘트 팔이 $r$.

- $K_{ff,zz}$ 에 $+\beta$
- $K_{mm}$ 에 $+\beta r^2$

$\sum\beta = 500$ 이고 모든 점이 $r \ge 0.446$ 이므로

$$\mathrm{tr}(K_{mm}) \;\ge\; 500 \times 0.446^2 \;=\; \mathbf{99.5} \;>\; 90 = \text{예산}$$

축 강성만으로 이미 회전 예산을 초과한다. **그리고 §3.1에 의해 이걸 뺄 방법이 없다.**

**전략 B — 반경 방향 힘 ($f = \hat p$)**

$f \parallel p$ 면 $p\times f = 0$ 이라 **모멘트가 공짜**다. 하지만 힘 방향이
$\hat p$ 로 고정되므로 축 성분만 뽑아낼 수 없다.

상면 구멍 테두리 $p = (0.455, 0, 0.375)$, $|p| = 0.590$:
$\hat p_z = 0.636$, $\hat p_r = 0.772$.

- $K_{ff,zz}$ 에 $+0.405\beta$
- $K_{ff,rr}$ 에 $+0.595\beta$

축에 500을 채우려면 $\beta = 1235$, 그러면 **횡방향 병진에 735**가 딸려온다
(예산 30, **24배 초과**). 이 점이 축에 가장 유리한 점이고, 나머지는 더 나쁘다.

**결론:** 두 전략의 혼합이 최적이 되며, 그 최적값이 아래 §4의 숫자다.

### 3.3 닫힌 형태 상한

전략 A를 구멍 벽(반지름 $\hat r = r_{\rm hole}/s$)에 축대칭으로 깔면,
$p\times e_z$ 가 크기 $\hat r$ 의 수평 접선 벡터이므로 방위각 평균에서

$$K_{mm} = S\cdot\mathrm{diag}\big(\tfrac{\hat r^2}{2},\; \tfrac{\hat r^2}{2},\; 0\big),
\qquad S = K_{ff,zz}$$

$K_{mm}$ 의 대각 예산이 $K_{\rm rot}$ 이므로 $S\hat r^2/2 \le K_{\rm rot}$, 즉

$$\boxed{\;\lambda_{\max} \;=\; \frac{2\,K_{\rm rot}}{\hat r_{\rm hole}^{\,2}}
\;=\; 2\,K_{\rm rot}\left(\frac{s}{r_{\rm hole}}\right)^{2}\;}$$

이 부품: $2\times 30\times (66.7/30)^2 = \mathbf{297}$.

§4.2의 수치 스윕에서 **라벨 300일 때 `d = 0.0001`(정확히 실현)** 이 나온 값이
바로 이것이다. 우연이 아니라 닫힌 해다.

이 식은 설계 지침으로도 쓸 수 있다 — 상한을 올리려면 (i) 구멍을 작게,
(ii) 부품을 크게(= $s$ 증가), (iii) 회전 강성 라벨을 올린다. 세 가지뿐이다.

---

## 4. 4단계 — 수치 검증

§2.2의 cone에서 $C_i \succeq 0$ 를 **직접** 최적화했다. 네트워크를 거치지 않고
접촉 강성을 자유롭게 고르는 것이므로, 이 값은 **이 head를 쓰는 어떤
네트워크도 넘을 수 없는 상한**이다.

### 4.1 라벨 500 고정

| | 학습된 모델 | | cone 최적 (도달 가능 상한) | |
|---|---|---|---|---|
| | `d` | `K_ff` 고유값 | `d` | `K_ff` 고유값 |
| hole62 | 0.4284 | (31.7, 32.5, **355.3**) | **0.1998** | (30.9, 31.0, **424.2**) |
| hole63 | 0.5501 | (31.9, 32.7, **327.5**) | **0.1934** | (31.0, 31.0, **426.7**) |
| hole64 | 0.5965 | (31.7, 32.2, **312.4**) | **0.2717** | (31.1, 31.3, **399.9**) |

### 4.2 라벨을 낮춰가며 — 실현 가능 경계는 어디인가 (hole64)

| 라벨 축 강성 | cone 최적 `d` | 해석 |
|---|---|---|
| **300** | **0.0001** | **정확히 실현 가능** |
| 400 | 0.0888 | 다른 축이 ~10% 틀어짐 |
| 440 | 0.1668 | |
| 500 | 0.2717 | 라벨 값 |
| 600 | 0.4221 | |

**나머지 5축을 30에 유지한 채 실현 가능한 축 강성의 최대는 약 300이다.**
학습된 모델의 312–355는 (다른 축이 전부 ~30인 상태에서) **이 물리적 경계에
거의 정확히 앉아 있다.** 학습 실패가 아니다.

### 4.3 점을 더 뽑으면 되나 — 안 된다 (hole64)

| N | `r_min` | cone 최적 `d` | 최대 축 강성 |
|---|---|---|---|
| 128 | 0.446 | 0.2717 | 399.9 |
| 512 | 0.417 | 0.1647 | 437.2 |
| 2048 | 0.455 | 0.1563 | 439.7 |
| 8192 | 0.460 | 0.1491 | 442.3 |

**64배 늘려도 400 → 442.** `r_min`이 구멍 반지름에서 막혀 있어서, 아무리
촘촘히 뽑아도 축 근처에 접촉점이 생기지 않는다. `MODEL_CARD.md` §4가 걱정한
"샘플링 밀도 드리프트"는 이 문제의 원인이 아니다.

---

## 5. 결정적 대조군 — 합성 데이터에서는 왜 loss가 0으로 갔나

이전 세션에서 `data_gen/gen_peg_hole_pcd.py`로 만든 합성 peg-hole 데이터에
대해서는 N=4 overfit이 **`d = 0.0012`, 조건수 193짜리 타깃에서 고유값 전부
상대오차 0.1% 이내**로 수렴했다. 같은 아키텍처인데 왜?

라벨 생성기를 보면 답이 나온다 — `peg_hole_synth.py:549-554`:

```python
m = torch.cross(Pc.unsqueeze(2).expand_as(u), u, dim=-1)   # m = p x u
w = torch.cat([m, u], dim=-1)                              # zero-pitch!
...
out[sl] = torch.einsum('bnk,bnki,bnkj->bij', kw, w, w)     # kw > 0
```

**합성 라벨 자체가 zero-pitch wrench의 양수 가중 outer product 합**이다. 즉
**설계상 모델의 cone 안에 있다.** 모델과 라벨이 정확히 같은 물리 모델을 쓰므로
train loss가 0으로 가는 게 당연하다.

반면 8월 데모의 `diag(30,30,30,30,30,500)`은 **손으로 고른 숫자**이고, 이
형상의 표면 접촉 강성으로 실현 가능한 집합 밖에 있다.

> **이것이 이 분석의 핵심 교훈이다.** 문제는 모델이 아니라, 모델의 물리 모델과
> 라벨의 물리 모델이 불일치한다는 점이다.

---

## 6. `MODEL_CARD.md`에서 정정할 것

**(1) §7 "스칼라 게인 $\alpha$ 로 조정할 것" — 정확히 서술해야 한다.**

$K_{\rm cmd} = \alpha K$ 는 **모든 고유값을 같은 비율로** 올린다. $\alpha = 500/355$
로 축을 500에 맞추면 나머지 5축이 **30 → 42**가 된다. 그 부작용이 허용되면
맞는 처방이지만, 제어기가 16.7:1 이방성 자체를 필요로 하면 $\alpha$ 로는
고칠 수 없다. 현재 문장은 이 점을 감춘다.

**(2) §7 "pointwise 인코더의 이방성 표현 상한" — 원인이 다르다.**

인코더의 표현력이 아니라 **zero-pitch head + 이 형상의 접촉 강성 실현
가능 집합**이다. 인코더를 아무리 키워도(그리고 실제로 키워봐도) 안 되는 이유가
§2.3에 있다.

**(3) §7의 sanity 기준값은 학습 구름 전용임을 명시해야 한다.**

`(31.7, 32.5, 355.3)` 등은 **seed 7로 뽑은 그 3개 128점 구름에서만** 성립한다.
같은 mesh를 다른 seed로 샘플링하면 축 강성이 **23–1252**로 흩어진다 (§10).
로드 검증용 기준값으로는 유효하지만, 모델 성능의 기술로 읽히면 안 된다.

**(4) `train_khat.py` 라벨 규약 잔재 — 수정 완료.**

- `train_khat.py` docstring: `diag(30,30,500,30,30,30)` → `diag(30,30,30, 30,30,500)`
- fallback 기본값: `[30.,30.,500.,30.,30.,30.]` → `[30.,30.,30.,30.,30.,500.]`

`[m; f]` 순서에서 index 2는 **z축 회전**이다. `83bb536`의 `[f;m] → [m;f]`
전환 때 남은 잔재였다. 데이터 파일에 `K_body_diag`가 있어 fallback은 발동하지
않았지만, 문서로서는 오독을 부른다.

---

## 7. 선택지

### 요구사항이 "임의의 SPD $K$ 를 반드시 표현"이라면 — §9로 갈 것

라벨을 손보는 것(축을 283으로 낮추거나 회전을 53으로 올리는 것)은 **이 부품
하나의 증상만** 없앤다. 다른 부품·다른 라벨에서 같은 문제가 다시 난다.
근본 해법은 아키텍처에서 pitch 자유도를 복원하는 것이고, **§9에 최소이면서
완전한 해법**(등변 선형 사상의 commutant를 다 쓰는 것)이 있다.

아래 (a)–(c)는 라벨을 손볼 수 있는 경우의 대안이다.

### (a) 라벨을 실현 가능한 값으로 바꾼다

실현 가능 창은 $\hat r_{\min}^2/2 \le K_{\rm rot}/K_{\rm axial}$ 이고
현재 라벨은 $30/500 = 0.060 < 0.106$ 이다. 둘 중 하나면 된다:

| | 결과 |
|---|---|
| `diag(30,30,30, 30,30,283)` | 축을 낮춤, `d → 0` |
| `diag(53,53,53, 30,30,500)` | **축 500 유지**, 회전만 올림 (여유 두면 55–60) |

### (b) `message_passing=True`

§1.2(f)에서 유일하게 zero-pitch를 깨는 기존 경로다. 서로 다른 점의 line
wrench를 섞으면 $m\cdot f \ne 0$ 이 된다. 검증했다:

```
message_passing=False  max |m - p x f|/|m| = 1.2e-13   normalised pitch = 5.4e-14
message_passing=True   max |m - p x f|/|m| = 1.05      normalised pitch = 4.3e-01
```

효과는 있지만 §9의 해법보다 무겁고(그래프 위 $O(Nk)$ 연산 추가) 간접적이다.
pitch가 이웃 관계에 종속되므로 "얼마나 생기는지"를 통제하기 어렵다.

### (c) 학습 갭만 회수한다 (상한 아래에서)

cone 최적 `d = 0.19–0.27` vs 학습 `d = 0.43–0.60`. 축 강성 기준 355 → ~424는
학습으로 회수 가능한 몫이다. 유력한 원인:

- `train_khat.py:91` — `CosineAnnealingLR(opt, T_max=args.epochs)`의 `eta_min`
  기본값이 **0**이라 마지막 epoch에서 LR이 정확히 0이 된다. 곡선이 눕는 것이
  수렴인지 LR 소멸인지 구분되지 않는다. (이전 실험에서 동일 step 수 기준
  cosine 0.2957 vs constant-LR 0.0988로 3배 차이가 났다.)
- 학습 표본 3개, batch 3.

단, 이걸 다 회수해도 **424가 천장**이라 500엔 도달하지 못한다.

---

## 8. 수학적 상세

기호: covector $z = (m, f) \in \mathfrak{se}(3)^*$, twist $\xi = (\omega, v) \in \mathfrak{se}(3)$,
pairing $\langle \xi, z\rangle = \omega\cdot m + v\cdot f$. 군작용은
$z \mapsto \mathrm{Ad}_T^{-\top} z$, 즉

$$\mathrm{Ad}_T^{-\top} = \begin{bmatrix} R & [p]_\times R \\ 0 & R\end{bmatrix},
\qquad T = (R, p).$$

점집합 $P = \{p_1,\dots,p_N\} \subset \mathbb{R}^3$ 에 대해
$J_i := [\,-[p_i]_\times \;\; I\,] \in \mathbb{R}^{3\times 6}$ 로 두면
$J_i \xi = v + \omega\times p_i = V(p_i)$ (점 $p_i$ 의 속도)이고
$J_i^\top f = (p_i\times f,\; f)$ 이다.

### 8.1 정리 1 — 도달 가능 집합의 정확한 특성화

> **정리 1.** `message_passing=False` 인 현재 아키텍처가 출력할 수 있는 $K$ 의
> 집합은 정확히
> $$\mathcal{K}(P) \;=\; \Big\{\; \sum_{i=1}^{N} J_i^\top C_i J_i \;:\; C_i \in \mathbb{S}^3_+ \;\Big\}$$
> 이며, `factors` $\ge 3$ 이면 채널 수·factor 수에 **의존하지 않는다.**

**증명.** $\mathcal{L}_p := \{(p\times f, f) : f\in\mathbb{R}^3\} = \mathrm{range}(J_p^\top)$
로 둔다. 각 층이 $\mathcal{L}_{p_i}$ 를 보존함을 보인다.

*(i) 입력.* `pointwise_models.py:228-236`에서 edge wrench가
$w_{ij} = (p_i\times d_{ij},\, d_{ij}) \in \mathcal{L}_{p_i}$.

*(ii) set pooling / `LNLinear`.* $\mathcal{L}_{p}$ 는 **선형 부분공간**이므로
같은 점의 채널에 대한 임의의 선형결합에 닫혀 있다.

*(iii) `covector_bracket`.* $a = (p\times f_a, f_a),\ b = (p\times f_b, f_b)$ 에 대해
BAC-CAB ($x\times(y\times z) = y(x\cdot z) - z(x\cdot y)$)으로

$$
f_a\times(p\times f_b) - f_b\times(p\times f_a)
= f_a(p\cdot f_b) - f_b(p\cdot f_a)
= p\times(f_a\times f_b),
$$

즉 $[a,b]_* = \big(p\times(f_a\times f_b),\; f_a\times f_b\big) \in \mathcal{L}_p$.
($\mathcal{L}_p$ 는 $f\mapsto f_a\times f_b$ 를 곱으로 갖는 $\mathfrak{so}(3)$ 와
동형인 **Lie 부분대수**다.)

*(iv) `KleinGate`.* 6성분 전체에 불변 스칼라를 곱하므로 보존.

따라서 head 입력 $z_{ih} \in \mathcal{L}_{p_i}$, 즉 $z_{ih} = J_i^\top f_{ih}$.
head는 $K = \sum_{i,h}\beta_{ih} z_{ih}z_{ih}^\top$ ($\beta = \mathrm{softplus} > 0$)
이므로

$$K = \sum_i J_i^\top \Big(\underbrace{\textstyle\sum_h \beta_{ih} f_{ih}f_{ih}^\top}_{=: C_i \succeq 0}\Big) J_i .$$

역으로 임의의 $C_i \succeq 0$ 는 고유분해로 3개의 $\beta f f^\top$ 합이므로
`factors` $\ge 3$ 이면 전부 실현된다. 그러므로 등호. $\;\blacksquare$

**따름정리 (capacity 무관).** $C_i$ 는 $3\times3$ 이라 스프링 3개로 포화한다.
`factors=32`는 10배 과잉이고, 채널을 늘려도 $\mathcal{K}(P)$ 는 넓어지지 않는다.

### 8.2 정리 2 — 상한이 존재한다 (엄밀)

축을 $z$, 원통좌표를 $p_i = (r_i\cos\varphi_i,\, r_i\sin\varphi_i,\, h_i)$ 로 쓴다.

> **보조정리.** 임의의 $f \in \mathbb{R}^3$ 와 $p = (r, 0, h)$ 에 대해
> $$|p\times f|^2 + |f_\perp|^2 \;\ge\; \frac{r^2}{1+h^2}\, f_z^2,
> \qquad f_\perp := (f_x, f_y).$$

**증명.** $f_z = 1$ 로 정규화(동차성). $p\times f = (-hf_y,\; hf_x - r,\; rf_y)$ 이므로

$$|p\times f|^2 + |f_\perp|^2 = \underbrace{(h^2 + r^2 + 1)f_y^2}_{\ge 0} + (hf_x - r)^2 + f_x^2 .$$

$f_y = 0$ 이 최적. $f_x$ 에 대해 미분하면 $h(hf_x - r) + f_x = 0$, 즉
$f_x = \dfrac{hr}{1+h^2}$. 대입하면

$$\Big(\tfrac{h^2 r}{1+h^2} - r\Big)^2 + \Big(\tfrac{hr}{1+h^2}\Big)^2
= \frac{r^2}{(1+h^2)^2} + \frac{h^2r^2}{(1+h^2)^2} = \frac{r^2}{1+h^2}. \;\blacksquare$$

> **정리 2.** $K \in \mathcal{K}(P)$ 이면
> $$K_{ff,zz} \;\le\; \gamma(P)\,\big(\operatorname{tr}K_{mm} + K_{ff,xx} + K_{ff,yy}\big),
> \qquad \gamma(P) := \max_i \frac{1 + h_i^2}{r_i^2}.$$

**증명.** $K = \sum_{i,h}\beta_{ih} z z^\top$, $z = (p_i\times f, f)$ 에서
$K_{ff,zz} = \sum \beta f_z^2$, $\operatorname{tr}K_{mm} = \sum\beta|p_i\times f|^2$,
$K_{ff,xx}+K_{ff,yy} = \sum\beta|f_\perp|^2$. 보조정리를 항별로 적용하고 합한다. $\;\blacksquare$

**핵심은 $\gamma$ 의 형태다.** $\gamma \to \infty$ 는 $r_{\min}\to 0$ 일 때만 일어난다.
즉 **축 위에 접촉점이 있을 때만 상한이 사라진다.** 구멍이 그것을 막는다.

실측값 (canonical):

| | $\gamma(P)$ | 정리 2의 상한 | 학습 모델 실제 |
|---|---|---|---|
| hole62 | 5.430 | 876 | 355.3 |
| hole63 | 5.480 | 902 | 327.5 |
| hole64 | 5.446 | 906 | 312.4 |

정리 2는 **상한의 존재와 그 메커니즘을 증명**하지만 값은 느슨하다
(가장 불리한 점 하나로 $\gamma$ 를 잡고 두 예산을 뭉뚱그리기 때문).
정확한 값은 §8.3의 구성과 §4의 수치 최적화가 준다.

### 8.3 정확한 값 — 축대칭 구성

구멍 벽 $\hat r$ 에 $z$ 방향 스프링을 방위각 균일하게 깔면
$p\times e_z = (r\sin\varphi,\, -r\cos\varphi,\, 0)$ 이고
$\mathbb{E}_\varphi[\sin^2] = \mathbb{E}_\varphi[\cos^2] = \tfrac12$ 이므로

$$K_{ff,zz} = S, \qquad
K_{mm} = S\cdot\mathrm{diag}\Big(\tfrac{\hat r^2}{2},\, \tfrac{\hat r^2}{2},\, 0\Big).$$

$K_{mm}$ 의 대각 예산이 $K_{\rm rot}$ 이므로 $S\hat r^2/2 \le K_{\rm rot}$:

$$\boxed{\;\lambda_{\max} = \frac{2K_{\rm rot}}{\hat r_{\rm hole}^2}
= 2K_{\rm rot}\Big(\frac{s}{r_{\rm hole}}\Big)^2\;}$$

학습에 쓴 N=128 구름 ($\hat r = 0.446$): $60/0.446^2 = 302$.
조밀 샘플링 ($\hat r = 0.460$): $60/0.460^2 = 284$.
§4.2의 스윕에서 라벨 300이 $d = 0.0001$ 로 **정확히 실현**된 것과 일치한다.

동치 형태 — **비율 조건**:

$$\frac{K_{\rm rot}}{K_{\rm axial}} \;\ge\; \frac{\hat r_{\rm hole}^2}{2} = 0.106,
\qquad\text{라벨}: \frac{30}{500} = 0.060 .$$

라벨이 기하가 허용하는 것보다 1.77배 극단적이다.

### 8.4 정리 5 — pitch 항등식 $\operatorname{tr}(K_{mf}) = 0$

$\mathcal{K}(P)$ 는 $\mathbb{S}^6$ 의 **초평면 안에** 들어 있다. 즉 기하와 무관한
선형 제약이 하나 더 있다.

> **정리 5.** 모든 $K \in \mathcal{K}(P)$ 에 대해
> $$\operatorname{tr}(K_{mf}) \;=\; K_{14} + K_{25} + K_{36} \;=\; 0
> \qquad (\text{[m; f] 순서}).$$

**증명.** 정리 1에서 $K_{mf} = \sum_i [p_i]_\times C_i$. $[p]_\times$ 는 반대칭,
$C_i$ 는 대칭이고, 반대칭 $A$ · 대칭 $B$ 에 대해
$\operatorname{tr}(AB) = \operatorname{tr}\big((AB)^\top\big)
= \operatorname{tr}(B^\top A^\top) = -\operatorname{tr}(BA) = -\operatorname{tr}(AB)$
이므로 $\operatorname{tr}(AB) = 0$. $\;\blacksquare$

**의미 — 이것은 pitch의 지문이다.** rank-1 항 $zz^\top$ 의 $mf$ 블록은 $mf^\top$
이고 그 대각합은 $m\cdot f = \text{pitch}\cdot|f|^2$. 따라서

$$\operatorname{tr}(K_{mf}) = \sum_{i,h}\beta_{ih}\,(m_{ih}\cdot f_{ih})
= \text{총 pitch (}\beta\text{-가중)}$$

이고, §1의 zero-pitch 구속이 출력 $K$ 에 남긴 **불변량**이다. 수치 확인:
생성자 768개의 선형 span $= 20/21$, 여공간이 정확히 $\operatorname{tr}(K_{mf})$
방향(비-$mf$ 블록 성분 $7\times10^{-16}$).

> **한 줄 진단법.** 목표 $K$ 에 대해 $K_{14}+K_{25}+K_{36}$ 을 계산해서 0이
> 아니면, 그 $K$ 는 현재 아키텍처로 **절대** 표현되지 않는다. 라벨을 어떻게
> 조정해도, 데이터를 아무리 늘려도 안 된다.

이번 라벨은 $K_{mf} = 0$ 이므로 이 검사는 통과한다. 그래서 실제로 걸린 것은
아래 §8.5의 positivity다.

### 8.5 상한의 정체 — 선형대수가 아니라 **부호**

정리 5의 초평면 안에서, 라벨은 **선형적으로는 정확히 도달 가능**하다.
$C_i$ 를 (부호 제약 없이) 대칭행렬로 두고 $\sum_i J_i^\top C_i J_i = K_{\rm gt}$
를 최소제곱으로 풀면

```
상대잔차 = 1.28e-15           <- 선형 span 안에 있다
해의 C_i 최소 고유값 = -4.971  <- 384개 고유값 중 123개가 음수
```

즉 라벨을 실현하는 해는 존재하지만 **$C_i \succeq 0$ 을 위반한다.**
$\beta = \mathrm{softplus} > 0$ 이 그것을 금지한다.

> **상한의 정체: "접촉은 밀 수만 있고 당길 수 없다."**
> $C_i$ 의 음의 고유값은 물리적으로 **당기는(접착) 스프링**이다. 접착 없는
> 표면 접촉만으로는 이 와셔를 축 방향으로 16.7배 뻣뻣하게 만들 수 없다.

정리 1의 집합을 다시 쓰면 두 겹의 제약이다:

$$\mathcal{K}(P) \;=\; \underbrace{\{\operatorname{tr}K_{mf} = 0\}}_{\text{선형, 여차원 1 (정리 5)}}
\;\cap\; \underbrace{\{\text{밀기만 하는 접촉으로 실현 가능}\}}_{\text{positivity, } \mathbb{S}^6_+ \text{의 진부분원뿔}}$$

§4의 수치 상한(400–427)과 §8.3의 닫힌 형태(284–302)는 **두 번째 겹**의 값이다.

(§8.4–8.5의 수치는 repo import 없이 동일 기하 — 90×90×50 블록 + Ø63 관통구멍 —
를 재생성한 자립 스크립트에서 얻었다(부록 B, $r_{\min} = 0.424$ vs 실제 0.446).
정리 5는 해석적으로 정확하므로 기하와 무관하고, positivity 결론은 실제 구름에
대한 §4.1의 cone 최적화($d = 0.27 \gg 0$)가 독립적으로 뒷받침한다.)

---

## 9. 임의의 SPD $K$ 를 표현하려면 — 빠져 있는 등변 연산자 $N$

### 9.1 정리 3 — 등변 선형 사상은 2차원이고, 코드는 절반만 쓴다

> **정리 3.** $\mathrm{End}_{SE(3)}\big(\mathfrak{se}(3)^*\big) = \mathrm{span}\{I,\, N\}$,
> $$N := \begin{bmatrix} 0 & I \\ 0 & 0\end{bmatrix}, \qquad N(m, f) = (f,\, 0).$$
> ($N^2 = 0$ 이므로 이 대수는 이중수 $\mathbb{R}[\varepsilon]/(\varepsilon^2)$ 다.)

**증명.** $\phi = \begin{bmatrix}A&B\\C&D\end{bmatrix}$ 가 모든
$\mathrm{Ad}_T^{-\top}$ 와 교환한다고 하자.

*(i) $T = (R,0)$*: $\mathrm{Ad}^{-\top} = \mathrm{diag}(R,R)$ 이므로 각 블록이
모든 $R\in SO(3)$ 와 교환한다. $\mathbb{R}^3$ 는 $SO(3)$ 의 **실 기약표현**이고
$\mathrm{End}_{SO(3)}(\mathbb{R}^3) = \mathbb{R}I$ (Schur, 실형) 이므로
$A = aI,\ B = bI,\ C = cI,\ D = dI$.

*(ii) $T = (I,p)$*: $\mathrm{Ad}^{-\top} = \begin{bmatrix}I&[p]_\times\\0&I\end{bmatrix}$.

$$\phi\,\mathrm{Ad}^{-\top} = \begin{bmatrix} aI & a[p]_\times + bI \\ cI & c[p]_\times + dI\end{bmatrix},
\qquad
\mathrm{Ad}^{-\top}\phi = \begin{bmatrix} aI + c[p]_\times & bI + d[p]_\times \\ cI & dI \end{bmatrix}.$$

(1,1) 성분에서 $c[p]_\times = 0\ \forall p \Rightarrow c = 0$;
(1,2) 성분에서 $a[p]_\times = d[p]_\times \Rightarrow a = d$.
따라서 $\phi = aI + bN$. $\;\blacksquare$

**정합성 확인.** Klein form $B(z,z') = m\cdot f' + m'\cdot f$ 에 대해
$B(Nz, z') = f\cdot f' = $ `force_pair`. 즉 **2차원 commutant $\leftrightarrow$
2차원 불변 쌍선형형식**(`klein_pair`, `force_pair`)이 정확히 대응한다.
리포트가 말한 "se(3)\* 위 등변 쌍선형사상의 2차원 공간"의 선형 버전이 정리 3이다.

**그런데** `core/lie_neurons_layers.py:21-31`의 `LNLinear`는 채널 축에만
`nn.Linear`를 거는, 즉 $a\!\cdot\!I$ 성분**만** 쓰는 층이다.
**$b\!\cdot\!N$ 성분이 아키텍처 전체에 없다.**

$$(aI + bN)(m, f) = (a\,m + b\,f,\; a\,f)$$

이고 $z$ 가 zero-pitch면 $z + \lambda Nz = (m + \lambda f,\, f)$ 의 pitch는
$\dfrac{(m+\lambda f)\cdot f}{|f|^2} = \lambda$. **$b$ 가 곧 pitch다.**
$N$ 이 없으니 pitch가 영원히 0인 것이고, 그것이 §8.1 정리 1의 원인이다.

수치 확인: 무작위 $T$ 200개에 대해 $\max\|NG - GN\| = 0.00\mathrm{e}{+}00$ (정확히 0).

### 9.2 정리 4 — $N$ 을 넣으면 전체 SPD cone에 도달한다

> **정리 4.** 층을 $\;x'_h = \sum_c (a_{hc}I + b_{hc}N)\,x_c\;$ 로 일반화하고,
> 각 점에서 채널의 force 성분 $\{f_c\}$ 가 $\mathbb{R}^3$ 를 span하면,
> 도달 가능 집합은 **전체 SPD cone** $\mathbb{S}^6_+$ 이다.

**증명.** $x_c = (p\times f_c,\, f_c)$ 에 대해

$$x'_h = \Big(p\times \underbrace{\textstyle\sum_c a_{hc}f_c}_{F_h}
+ \underbrace{\textstyle\sum_c b_{hc}f_c}_{G_h},\;\; F_h\Big)
= (p\times F_h + G_h,\; F_h).$$

$\{f_c\}$ 가 $\mathbb{R}^3$ 를 span하므로 $F_h, G_h$ 를 **독립적으로 임의로**
고를 수 있다. 목표 $z^\star = (m^\star, f^\star)$ 에 대해 $F = f^\star$,
$G = m^\star - p\times f^\star$ 로 두면 $x'_h = z^\star$. 즉 $x'_h$ 는
$\mathbb{R}^6$ 전체를 훑는다.

따라서 $K = \sum \beta\, z z^\top$ 는 **모든** rank-1 $zz^\top$ 의 원뿔결합이 되고,
$\mathbb{S}^6_+$ 의 극단선이 정확히 rank-1들이므로 도달 집합 $= \mathbb{S}^6_+$.
임의의 SPD $K = \sum_{k=1}^{6}\lambda_k v_kv_k^\top$ 는 6개 열이면 되고
현재 $NH = 128\times32 = 4096$ 열이 있다. $\;\blacksquare$

**전제 확인 (수치).** 학습된 모델의 마지막 층에서 점별 force 부분공간의
$\sigma_3/\sigma_1$ 최소값 $= 3.25\times10^{-2} > 0$ → 모든 점에서 rank 3. ✓

**결론 확인 (수치).** $z$ 를 $\mathbb{R}^6$ 에서 자유롭게 최적화하면
라벨 `diag(30,30,30,30,30,500)`에 대해 $d = 1.8\times10^{-4}$
(zero-pitch 구속 시 0.2717). ✓

### 9.3 패치 — 적용 완료

`LNLinearPitch` 는 `experiment/pc_se3_congruence/models.py` 의
`covector_bracket` 옆에 있다. (`core/lie_neurons_layers.py` 가 아니다 —
`core` 의 `LNLinear` 는 대수 무관이라 sl3의 8차원도 처리하는 반면, 이 연산은
`[m; f]` 6차원 저장을 가정하는 se(3)\* 전용이다.)

```python
class LNLinearPitch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.a = nn.Linear(in_channels, out_channels, bias=False)
        self.b = nn.Linear(in_channels, out_channels, bias=False)
        nn.init.zeros_(self.b.weight)          # 초기값에서 LNLinear 와 동일

    def forward(self, x):                      # [B, C, 6, N], [m; f]
        def mix(w, y):
            return w(y.transpose(1, -1)).transpose(1, -1)
        m, f = x[:, :, 0:3], x[:, :, 3:6]
        return torch.cat([mix(self.a, m) + mix(self.b, f), mix(self.a, f)], dim=2)
```

**스위치.** `PointwiseStiffnessModel(pitch=...)`, `PITCH_MODES = ('none',
'head', 'all')`. 기본값 `'none'` 이라 **기존 체크포인트가 그대로 로드된다**
(`state_dict` 키가 안 바뀜 — `khat_pointwise.pt` 로 검증). CLI는
`blockage_bench.py --pw-pitch`, `train_khat.py --pitch`.

| 값 | 교체 대상 | 효과 |
|---|---|---|
| `none` | 없음 | 기존 모델과 **완전히 동일** |
| `head` | `LateSecondMomentHead.linear` (`pointwise_models.py:524`) | 정리 4 — $K$ 가 전체 SPD cone 도달 |
| `all` | 위 + `PointwiseCovectorBlock.linear` (`:467`) | 추가로 스칼라 트랙 복원 (아래) |

**왜 head 하나로 충분한가.** 정리 4대로, 그 앞까지는 zero-pitch여도 되고
마지막 채널 혼합에서 $G_h$ 가 만들어진다. `proj_u`/`proj_v` 계열
(`:403-404`, `:532-533`)은 covector를 만들지만 곧바로 스칼라로 축약되므로
$K$ 의 도달 집합과 무관하다 — 바꾸지 않는다.

**`all` 의 추가 이득.** pitch가 있는 feature에 대해

$$B(z, z') = (\lambda + \lambda')\,(f\cdot f') \;\ne\; 0$$

이므로 **`klein_pair`가 더 이상 항등적으로 0이 아니다.** 즉 리포트 §3.7의
zero-pitch degeneracy(스칼라 트랙 붕괴)도 같이 해소된다.

**보존되는 것.**

- **등변성**: $N$ 이 intertwiner이므로 정확히 보존 ($\max\|NG-GN\| = 0$).
  `train_khat.py`의 인증값 ~1e-11이 그대로 나와야 한다.
- **SPD**: $K = LL^\top$ 구조는 $z$ 가 무엇이든 유지된다.
- **하위 호환**: $b$ zero-init이라 초기값에서 기존 `LNLinear`와 수치적으로 동일.

**버리는 것.**

"출력이 항상 물리적으로 실현 가능한 표면 접촉 강성"이라는 inductive bias.
임의의 SPD를 표현해야 한다는 요구사항이 이 bias와 **정확히 양립 불가**이므로
(정리 1), 이건 피할 수 있는 대가가 아니라 요구사항의 논리적 귀결이다.

### 9.4 검증 결과

**표현력 (결정적).** hole62/63/64 canonical cloud 3개를
`diag(30,30,30,30,30,500)` 으로 overfit. 4000 epoch, batch 3, lr 2e-3,
cosine($\eta_{\min} = $ lr/100), seed 0, channels 16-64-64-32, factors 32 —
`pitch` 외에는 **전부 동일**.

| `pitch` | 최종 `d` | 최저 `d` | 축 강성 | $\operatorname{tr}(K_{mf})$ | params |
|---|---|---|---|---|---|
| `none` | 0.5453 | 0.5453 | 356 / 317 / 300 | $\pm10^{-13}$ | 48,769 |
| `head` | 0.0016 | 0.0007 | **500 / 500 / 500** | $5\times10^{-2}$ | 49,793 |
| **`all`** | **0.0007** | **0.0006** | **500 / 500 / 500** | $8\times10^{-3}$ | 56,961 |

**핵심은 부등호다: $0.0007 \ll 0.1934$.** 0.1934는 §4.1에서 $C_i$ 를 자유
최적화해 얻은, `pitch=none` 이 가중치를 어떻게 잡아도 넘을 수 없는 **하한**이다.
`head`/`all` 이 그보다 **270배 아래**로 갔다 — "학습이 더 잘 됐다"가 아니라
도달 가능 집합 자체가 바뀌었다는 증거다. 축 강성도 정확히 500이다.

예측한 세 가지가 동시에 일어났다:

1. **cone 탈출** — 축 강성 300–356 → 500, `d` 가 구조적 하한 아래로.
2. **초평면 탈출 (정리 5)** — $\operatorname{tr}(K_{mf})$ 가 $10^{-13}$ →
   $10^{-2}$. `pitch=none` 은 4000 epoch 내내 $\pm10^{-13}$ 을 벗어난 적이 없다.
3. **등변성 무손상** — 아래 표.

`all` 이 `head` 보다 2배 낮지만, 둘 다 이미 라벨을 사실상 정확히 맞춘
($d < 10^{-3}$) 상태라 이 실험은 둘을 가르지 못한다. `all` 의 진짜 이득은
스칼라 트랙(`klein_pair` 부활)이 필요한 문제에서 나오므로, 그건 §5의 합성
데이터처럼 β 변조가 실제로 필요한 과제에서 재봐야 한다.

**등변성·하위 호환.**

| 검사 | 결과 |
|---|---|
| 기존 `khat_pointwise.pt` 로드 | OK, $K_{ff} = (31.7, 32.2, 312.4)$ — MODEL_CARD와 일치 |
| `pitch=head` vs `none` 초기값 | $\|\Delta K\|_\infty = 0.00\mathrm{e}{+}00$ ($b$ zero-init) |
| 등변성 잔차 ($b$ 를 무작위로 켠 상태) | `none` 3.8e-19 · `head` 1.8e-17 · `all` 8.7e-17 |
| pitch 실측 $\|m - p\times f\|/\|m\|$ | `none` 1.2e-13 · `head` **4.90** · `all` **1.53** |

### 9.5 실제 파이프라인 검증 — 8월 데모 재학습

위 §9.4는 3개 구름을 직접 최적화한 통제 실험이다. 실제 진입점
(`train_khat.py`, wandb, MPS)으로도 돌렸다.

```
train_khat.py --data data/real_objects/holes_canonical_axis.pt --pitch head
              --epochs 4000 --batch 3 --lr 2e-3 --channels 16 64 64 32
              --factors 32 --device mps
```

wandb run `o6dl44x6` (`adjoint_equivariant_network/pc-se3-congruence`), 290초:

```
train/d              0.00016     <- pitch=none 의 구조적 하한은 0.19
equivariance/final   1.39e-11    <- MODEL_CARD 의 ~1e-11 유지
val/tr_K_mf          27.0        <- 초평면 탈출 (none 이면 1e-13)
```

**출력 $K$ 를 라벨과 직접 비교** (hole64, 6×6 전체). 라벨은
`diag(30,30,30, 30,30,500)`:

*원래 커밋 (`pitch=none`, ep3999) — `d = 0.5965`*

```
    37.91   -3.94    0.78  |   1.34    1.72    0.91
    -3.94   36.42   -0.27  |  -0.58   -1.21    1.27
     0.78   -0.27   28.17  |  -0.03    0.22   -0.13
   ─────────────────────────────────────────────────
     1.34   -0.58   -0.03  |  32.00    0.27    2.44
     1.72   -1.21    0.22  |   0.27   31.90    1.88
     0.91    1.27   -0.13  |   2.44    1.88  312.36     <- 라벨은 500
```

*`pitch=head` 수렴 후 — `d = 0.0013`*

```
    30.01   -0.00    0.00  |   0.00    0.00   -0.00
    -0.00   30.01    0.00  |   0.00    0.00   -0.01
     0.00    0.00   30.02  |   0.00    0.00    0.05
   ─────────────────────────────────────────────────
     0.00    0.00    0.00  |  30.01   -0.00    0.01
     0.00    0.00    0.00  |  -0.00   30.01    0.02
    -0.00   -0.01    0.05  |   0.01    0.02  500.26     <- 라벨 정확히 재현
```

원래 커밋의 행렬이 **어떻게 실패했는지**까지 보인다: 회전 블록이
`37.91 / 36.42 / 28.17` 에 비대각 `-3.94` 로 일그러져 있다. 라벨은 $30I$ 인데,
축 강성을 조금이라도 더 짜내려고 shape 오차를 지불한 흔적이다 — §3.2의 예산
계산이 예측한 그대로다. 그러고도 312에서 멈춘다.

**주의 — 저장된 체크포인트.** `train_khat.py` 는 best val_d 에서 저장하는데
이 run은 val_d 가 ep49에서 최저(2.26)였다가 계속 나빠졌다. 따라서
`khat_pitch_head.pt` 는 **epoch 49의 미학습 모델**이고 (병진 블록 73–115),
`train/d = 0.00016` 을 낸 최종 모델은 저장되지 않았다. 위 "수렴 후" 행렬은
동일 설정으로 재학습해 뽑은 것이다. val이 의미를 갖기 전까지(§10) best-val
게이팅은 학습된 모델을 버린다.

**남은 확인 (합성 데이터 회귀).** §5의 합성 peg-hole 라벨은 zero-pitch cone
**안에** 있으므로 $b \to 0$ 이 최적해다. 학습이 그걸 찾는지 —
`--pw-pitch head` 가 `none` 대비 성능을 떨어뜨리지 않는지 — 확인해야 한다.
`run_experiments.sh` 의 `PW_PITCH` 환경변수로 돌린다.

---

## 10. 상한을 걷어내자 드러난 다음 문제 — 3표본 암기

상한이 사라지면서 그것에 가려져 있던 문제가 보이게 됐다. **이 문서의 주제는
아니지만, 다음 작업의 출발점이므로 기록한다.**

원래 커밋은 **애초에 맞출 수가 없어서** 과적합이 일어날 수도 없었다. 이제
맞출 수 있게 되니 `train/d = 0.00016` 인데 `val/d = 3.42` 다 — 표본 3개를
외운 것이다.

**원래 실험의 val은 무의미했다.** `khat_pointwise.pt` 를 재샘플 구름에 돌려본
결과:

| | `khat_pointwise.pt` (pitch=none) | `khat_pitch_head.pt` (ep49) |
|---|---|---|
| 학습 구름 (seed 7) | `d 0.5250`  축강성 331.7 | `d 2.5454`  축강성 106.8 |
| 재샘플 (seed 1000–1004) | **`d 7.02`**  축강성 255 (**52–724**) | `d 2.19`  축강성 88 |
| 재샘플 (seed 2000–2004) | **`d 7.51`**  축강성 300 (**23–1252**) | `d 2.27`  축강성 83 |

학습 구름의 `d = 0.5250` 이 `train_meta` 에 기록된 `val_d 0.5250` 과 소수점
4자리까지 같다. 모델이 정확히 등변이고 AIRM이 congruence 불변이므로, 이 일치는
**원래 val 셋이 학습 구름의 congruence 수송본**이었음을 뜻한다. 그 val은
등변성만 재고 있었다.

같은 mesh를 **seed만 바꿔 다시 샘플링해도** 축 강성이 23–1252로 흩어진다.
`MODEL_CARD.md` §7의 312/327/355는 그 3개 특정 샘플링에서만 성립하는 값이다.

**빠진 증강 축은 샘플링이다.** `MODEL_CARD` §8이 pose 증강을 뺀 것은 옳다 —
AIRM의 congruence 불변성과 모델의 등변성 때문에 gradient 기여가 정확히 0이다.
그러나 **그 논리는 재샘플링에는 적용되지 않는다**: 다른 샘플링은 실제로 다른
입력이고 gradient가 0이 아니다. `data_gen/gen_holes_canonical.py` 의
`--n-val-per-mesh` 와 seed 대역을 늘려 train을 mesh당 수십~수백 샘플링으로
만들면, 그때 비로소 `train/d` 대 `val/d` 격차가 의미를 갖는다.

그 데이터로 `none` / `head` 를 다시 비교하면 **상한 제거가 일반화에도 도움이
되는지**, 아니면 접촉 원뿔 제약이 정규화로 일하고 있었는지가 갈린다. 후자일
가능성도 실재한다.

---

## 부록 A: 재현 스크립트

```python
"""cone_ceiling.py — repo 루트에서 실행"""
import sys; sys.path.append('.')
import torch
torch.set_default_dtype(torch.float64)
from experiment.peg_in_hole_august_demo.khat_infer import KhatEstimator, _NAMES
from experiment.pc_se3_congruence.spd_loss import affine_invariant_d

K_GT = torch.diag(torch.tensor([30., 30., 30., 30., 30., 500.]))
CHOL = torch.linalg.cholesky(K_GT)[None]

def hat(p):
    z = torch.zeros_like(p[..., 0])
    return torch.stack([
        torch.stack([z, -p[..., 2], p[..., 1]], -1),
        torch.stack([p[..., 2], z, -p[..., 0]], -1),
        torch.stack([-p[..., 1], p[..., 0], z], -1)], -2)

def best_cone_K(P, steps=3000, lr=0.05):
    """min_{C_i >= 0} AIRM(K_gt, sum_i J_i^T C_i J_i) — 구조적 하한."""
    J = torch.cat([-hat(P), torch.eye(3).expand(P.shape[0], 3, 3)], -1)
    A = (torch.eye(3).expand(P.shape[0], 3, 3).clone() * 0.5).requires_grad_()
    opt = torch.optim.Adam([A], lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 1e-2)
    best = (1e9, None)
    for _ in range(steps):
        C = A @ A.transpose(-1, -2)
        K = torch.einsum('nai,nab,nbj->ij', J, C, J)
        K = 0.5 * (K + K.T) + 1e-10 * torch.eye(6)
        d, _ = affine_invariant_d(CHOL, K[None])
        opt.zero_grad(set_to_none=True); d.mean().backward()
        opt.step(); sch.step()
        if d.item() < best[0]:
            best = (d.item(), K.detach().clone())
    return best

est = KhatEstimator()
for n in _NAMES:
    P = est.P_canon[n]
    d, K = best_cone_K(P)
    print(n, f'cone d={d:.4f}', torch.linalg.eigvalsh(K[3:6, 3:6]))
```

zero-pitch 검증:

```python
X = est.model.features(P[None]); L, K = est.model.head(X)
N = P.shape[0]; H = L.shape[-1] // N
Lc = L[0].reshape(6, H, N); m, f = Lc[0:3], Lc[3:6]
pxf = torch.cross(P.T[:, None, :].expand(3, H, N), f, dim=0)
print((m - pxf).norm(dim=0).max() / m.norm(dim=0).max())   # ~1e-14
```

## 부록 B: span / positivity 검증 (§8.4–8.5)

repo import 없이 동일 기하(90×90×50 블록 + Ø63 관통구멍)를 재생성해 정리 5와
§8.5를 확인한다. 전체 스크립트는 `scratchpad/span_check3.py`.

```python
J = torch.cat([-hat(P), torch.eye(3).expand(N, 3, 3)], -1)      # [N, 3, 6]

# 대칭 6x6 -> 21차원 좌표.  비대각에 sqrt(2) 를 곱해야 Frobenius 내적이 보존되고,
# 그래야 아래 SVD 의 여공간이 곧 '항등식'이 된다.
iu = torch.triu_indices(6, 6)
wgt = torch.where(iu[0] == iu[1], 1.0, 2.0 ** 0.5)
vec = lambda S: S[iu[0], iu[1]] * wgt

E = []                                        # 대칭 3x3 기저 6개
for a, b in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]:
    M = torch.zeros(3, 3); M[a, b] = M[b, a] = 1.0
    E.append(M)

G = torch.stack([vec(J[i].T @ M @ J[i]) for M in E for i in range(N)])   # [6N, 21]
print(torch.linalg.matrix_rank(G))            # -> 20, NOT 21  (정리 5)

_, _, Vh = torch.linalg.svd(G)                # 여공간 Vh[20:] = tr(K_mf) 방향
K_GT = torch.diag(torch.tensor([30., 30., 30., 30., 30., 500.]))
x = torch.linalg.pinv(G.T) @ vec(K_GT)        # 부호 자유 해
# -> 잔차 1.3e-15 인데 대응하는 C_i 의 최소 고유값이 -4.97 (= 당기는 스프링)
```

**함정.** `torch.linalg.lstsq` 는 이 과소결정계(36×768)에서 신뢰할 수 없다 —
기본 driver가 rank 결손을 처리하지 못해 잔차 0.9를 내놓는다. `pinv` 를 쓸 것.

## 부록 C: 도구와 실행법

이 분석 과정에서 추가·수정된 것들.

**새 코드**

| 위치 | 내용 |
|---|---|
| `experiment/pc_se3_congruence/models.py` | `LNLinearPitch` — 빠져 있던 등변 생성자 $N$ (§9.3) |
| `experiment/pc_se3_congruence/pointwise_models.py` | `PointwiseStiffnessModel(pitch=...)`, `PITCH_MODES` |
| `data_gen/gen_holes_canonical.py` | STL → `holes_canonical_axis.pt`. `mesh_pcd.py` 의 `sample_mesh_surface` / `body_frame_stiffness_labels` 를 `MODEL_CARD` §8 레시피로 조립한다 (원래 커밋에 이 진입점이 없었다) |

**새 플래그**

| | |
|---|---|
| `blockage_bench.py --pw-pitch {none,head,all}` | wandb run 이름에도 반영 |
| `run_experiments.sh` `PW_PITCH=` | teacher 타깃 모델에도 적용되므로 realizability 검사가 '지금 학습하는 클래스' 기준으로 유지된다 |
| `train_khat.py --pitch {none,head,all}` | |
| `train_khat.py --const-lr` | 기존 `CosineAnnealingLR(T_max=epochs)` 는 `eta_min` 기본값이 **0** 이라 마지막 epoch에서 LR이 정확히 0이 된다 — 곡선이 눕는 것을 수렴으로 오독하게 만든다 |
| `train_khat.py --device mps --dtype ...` | 아래 |
| `train_khat.py --wandb-mode {online,offline,disabled}` | repo 공용 `init_wandb`. `val/axial_stiffness` 와 `val/tr_K_mf` 가 이 실험의 핵심 지표다 |

**MPS (Mac) 지원.** MPS에는 float64가 아예 없고 `aten::_linalg_eigh` 도 없다.
백본만 올리고 6×6 손실은 CPU/float64에 남긴다 — CPU/CUDA에서는 손실도 학습
장치에 그대로 두므로 스텝마다 동기화가 생기지 않는다. 등변성 인증은 항상
CPU/float64 사본으로 잰다 (float32로 재면 ~1e-6이 나와 MODEL_CARD의 ~1e-11과
비교할 수 없다). 체크포인트는 학습 장치와 무관하게 cpu/float64로 저장한다.

```
--device mps  →  4000 epoch 약 6분,  val_d 가 float64/CPU 와 소수점 4자리까지 일치
```

**함정.** `.to(device='cpu', dtype=torch.float64)` 를 한 번에 쓰면 MPS가 dtype
캐스팅을 장치 이동보다 먼저 시도해서 터진다. `.cpu().double()` 로 분리할 것.

**실행**

```bash
python data_gen/gen_holes_canonical.py

python experiment/peg_in_hole_august_demo/train_khat.py \
    --data data/real_objects/holes_canonical_axis.pt \
    --pitch head --epochs 4000 --batch 3 --lr 2e-3 \
    --channels 16 64 64 32 --factors 32 \
    --device mps --wandb-mode online \
    --out experiment/peg_in_hole_august_demo/khat_pitch_head.pt
```

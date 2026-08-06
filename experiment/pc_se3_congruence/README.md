# `pc_se3_congruence` 실험 기준 문서

이 파일은 이 폴더의 **현재 결론과 다음 의사결정만 모은 기준 문서**다. 이후 작업은 먼저 이 파일을 확인하고, 새 실험을 추가하면 아래 표와 결론을 함께 갱신한다. 긴 증명과 세부 설정은 하단의 원문 문서를 따른다.

## 1. 결론

1. Point cloud에서 stiffness $K$를 예측하는 두 경로는 모두 정확히 SE(3) congruence-equivariant하다.
   - vector 경로: twist로 계속 계산한 뒤 마지막 head에서 $Q$로 wrench로 내린다.
   - covector 경로: 처음부터 wrench로 lift하고 covector bracket으로 계속 계산한다.
2. 두 경로는 Klein map $Q$로 서로 켤레인 **같은 표현**이다. 가중치를 대응시키면 encoder 출력은 오차 0, 최종 $K$는 $2.2\times10^{-16}$ 차이로 같다. covector 경로 자체가 더 큰 함수 클래스를 주지는 않는다.
3. bracket-only vector backbone의 $v\to\omega$ 막힘은 covector에서 $m\to f$ 막힘으로 그대로 옮겨진다. 따라서 “covector로 계속 가면 translation 정보가 orientation에 영향을 줄 수 있다”는 가설은 **거짓**이다.
4. Killing form을 Klein form으로 바꾸면 rank 결손은 사라지지만, Klein form은 signature $(3,3)$의 indefinite form이다. 따라서 norm이나 정사영 분모로 쓰는 **Klein-ReLU는 안전하지 않다**.
5. Klein form은 서로 다른 학습 채널 사이의 **bounded invariant scalar gate**로 쓰면 유효하다. 이 방식은 정확한 등변성을 유지하면서 translation/moment slot이 translation-blind slot을 조절하는 경로를 연다.

현재 권장안은 다음과 같다.

> 타입은 vector/covector 중 구현하기 편한 하나를 일관되게 유지하고, 비선형성은 bracket에 bounded cross-channel Klein gate를 더한다. Klein form으로 norm을 만들거나 $B(d,d)$로 나누지 않는다.

## 2. 표기와 두 경로

저장소는 다음 순서를 쓴다.

- twist: $\xi=[v;\omega]$, $v$=linear/moment slot, $\omega$=angular/direction slot
- wrench: $F=[f;m]$, $f$=force slot, $m$=moment slot

강체변환 $T=(R,p)$의 작용은

$$
\begin{aligned}
\mathrm{Ad}_T[v;\omega] &= [Rv+p\times R\omega;\ R\omega],\\
\mathrm{Ad}_T^{-\top}[f;m] &= [Rf;\ Rm+p\times Rf].
\end{aligned}
$$

즉 vector에서는 $\omega$, covector에서는 $f$가 global translation에 blind하다. 이는 등변성의 요구사항이지 결함이 아니다. 여기서 문제 삼는 “translation 정보의 영향”은 global origin 이동이 아니라, 입력의 상대 위치·moment slot이 네트워크 내부에서 $\omega$ 또는 $f$ 출력을 조절할 수 있는지다.

### 경로 A — vector backbone, 마지막에 covector

$$
P\to [r\times n;n]\xrightarrow{\mathrm{Ad}}
\text{LNLinear+twist bracket}\to Z
\xrightarrow{Q}Y\to K=YY^\top/C.
$$

### 경로 B — 처음부터 covector

$$
P\to[n;r\times n]\xrightarrow{\mathrm{Ad}^{-\top}}
\text{LNLinear+covector bracket}\to Y\to K=YY^\top/C.
$$

$Q=\left[\begin{smallmatrix}0&I\\I&0\end{smallmatrix}\right]$이므로 wrench lift는 twist lift의 정확한 block swap이다. Covector bracket도

$$
[F_1,F_2]_*=[f_1\times f_2;\ f_1\times m_2-f_2\times m_1]
=Q[Q^{-1}F_1,Q^{-1}F_2]
$$

로 twist bracket을 $Q$로 옮긴 것이다. 따라서 두 전체 경로는 수학적으로 같은 모델이다.

## 3. 왜 covector 경로도 막힘을 해결하지 못하는가

Twist bracket의 angular 출력과 covector bracket의 force 출력은 각각

$$
[\xi_1,\xi_2]_\omega=\omega_1\times\omega_2,
\qquad
[F_1,F_2]_{*,f}=f_1\times f_2
$$

이다. 첫 식은 $v$를, 둘째 식은 $m$을 보지 않는다. Bias 없는 channel mixing과 이 bracket만 반복하면 이 의존성 차단은 깊이와 무관하게 유지된다.

새 수치 검사 결과:

| 검사 | scaled change |
|---|---:|
| vector bracket backbone: $v$만 변경 후 $\omega_{out}$ 변화 | **0.0** |
| covector bracket backbone: $m$만 변경 후 $f_{out}$ 변화 | **0.0** |
| Klein gate 추가: $v$ 변경 후 $\omega_{out}$ 변화 | 0.855 |
| Klein gate 추가: $m$ 변경 후 $f_{out}$ 변화 | 0.725 |

따라서 해결책은 representation을 뒤집는 것이 아니라 **두 slot을 포함하는 invariant scalar**를 만드는 것이다.

## 4. Killing form과 Klein form

Repo 순서에서 두 Gram matrix의 스펙트럼은 다음과 같다.

| form | 식 | rank | spectrum | 판단 |
|---|---|---:|---|---|
| Killing | $B_K(x,d)=\omega_x\cdot\omega_d$ | 3 | $0\times3,+1\times3$ | $v$를 전혀 못 봄 |
| Klein | $B_Q(x,d)=v_x\cdot\omega_d+\omega_x\cdot v_d$ | 6 | $-1\times3,+1\times3$ | non-degenerate지만 indefinite |

### 4.1 Klein으로 바꾸면 좋아지는 점

- $B_Q(\mathrm{Ad}x,\mathrm{Ad}d)=B_Q(x,d)$이므로 Klein score와 그 score로 가중한 equivariant feature는 정확히 equivariant하다.
- $v$만 바꿨을 때 Killing score 변화는 0, Klein score 변화는 0.931이었다. 즉 Klein score는 bracket-only가 놓치는 cross-slot 정보를 본다.
- 같은 Klein matrix는 covector 표현에서도 invariant하므로 vector/covector 양쪽에 동일한 gate 원리를 쓸 수 있다.

### 4.2 Klein으로도 해결되지 않는 점

- Klein form은 양의 정부호 norm이 아니다. $B_Q(d,d)$는 양수·음수·0이 모두 가능하다.
- 모든 raw Plücker line $[r\times n;n]$은 null이다: $B_Q(x,x)=2(r\times n)\cdot n=0$. 실측 최대값은 $1.8\times10^{-15}$였다.
- pure-translation feature끼리의 Klein pairing도 항상 0이다. Klein의 non-degeneracy는 translation feature가 적절한 angular feature와는 pair될 수 있다는 뜻이지, translation 부분공간이 스스로 metric을 갖는다는 뜻이 아니다.
- 서로 다른 두 null line은 일반적으로 pairing이 0이 아니다. 실측 평균 $|B_Q(x_i,x_j)|=2.30$이므로 $B_Q(x,d)/B_Q(d,d)$ 형태의 정사영은 0으로 나눈다. $\epsilon=10^{-12}$를 넣은 실험도 출력 norm을 $6.9\times10^{11}$배 증폭했다.

### 4.3 직접 Klein-ReLU와 권장 gate

Klein score의 부호로 branch를 나누고 $x-sd$를 반환하는 직접 치환은 등변이다. 그러나 metric projection이 아니라 indefinite sign이 선택하는 shear이며, 입력을 1, 10, 100배 했을 때 출력/입력 norm 비가 각각 2.62, 241, 24,089로 커졌다. 비선형 항이 cubic scale을 갖기 때문이다.

권장하는 형태는 form을 분모로 쓰지 않는 bounded gate다.

$$
q=VW_q,\quad k=VW_k,\quad s_c=q_c^\top Qk_c,\quad
V_c' = V_c\bigl(1+\tanh(s_c)\bigr).
$$

- $q,k$는 서로 다른 learned channel mix를 사용한다. 한 raw Plücker feature의 self-pairing은 0이므로 self-form만 쓰면 안 된다.
- score는 invariant scalar이고 $1+\tanh(s)\in(0,2)$이므로 출력은 equivariant하며 amplification이 bounded다.
- 실측 equivariance 오차는 $\|p\|\in\{0,1,10^2,10^4\}$에서 vector/covector 모두 최대 $6.3\times10^{-12}$였다. Euclidean score 대조군은 $p\ne0$에서 0.64–0.87로 실패했다.

Cross-slot 의존성만 분리한 synthetic target에서 test NMSE는 Klein gate 0.00138, Killing gate 0.410, bracket-only 0.411이었다. Target 자체가 Klein gate로 만들어진 통제 실험이므로 이는 **구조적 가능성·학습 가능성**의 증거이며, 실제 contact-spring 성능 향상의 증거는 아니다.

## 5. 전체 실험 결과 요약

모든 구조 검사는 float64, 랜덤 가중치, $\|p\|\in\{0,1,10^2,10^4\}$에서 수행했다. Positive test의 $10^{-16}$–$10^{-12}$ 증가는 큰 translation 행렬곱의 round-off이며, negative control의 $O(1)$ 실패와 명확히 분리된다.

| 묶음 | 핵심 결과 | 판정 |
|---|---|---|
| O — 대수 항등식 | Ad homomorphism, $\mathrm{Ad}^\top Q\mathrm{Ad}=Q$, flat/sharp, $Q^2=I$; 최대 2.9e-12 | PASS |
| L — point-cloud lift | closed-form/learnable encoder 최대 3.5e-12; anchor transport 누락은 $p\ne0$에서 0.67–1.0 | PASS / 대조군 검출 |
| A — compliance형 | $C'=\mathrm{Ad}C\mathrm{Ad}^\top$ 최대 3.8e-12; linear bias는 0.99–1.31 실패 | PASS / 대조군 검출 |
| B — stiffness형 | $K'=\mathrm{Ad}^{-\top}K\mathrm{Ad}^{-1}$ 최대 6.9e-12, $K$ full-rank SPD; head의 $Q$ 누락은 약 1.02 실패 | PASS / 대조군 검출 |
| B 정규화 대조군 | $K+\epsilon I$는 $p\ne0$에서 0.43–0.66 실패 | 금지 확인 |
| C — covector | raise/backbone/lower, cascade, native covector bracket 모두 약 1e-15; twist bracket을 wrench에 적용하면 0.42–1.0 실패 | PASS / 타입 오류 검출 |
| C — 연산 분류 | covector equivariant bilinear map 공간 차원 2; transported bracket과 $N_*$가 전부 | 확인 |
| Killing 진단 | rank 3, radical 차원 3; pure translation에서 `LNKillingRelu` 정확히 identity; $v$ perturbation에 gate 변화 0 | 사용 한계 확인 |
| 학습 — teacher | train $d$: 0.384→0.00207, val 0.00218, 최종 equivariance 7.8e-16 | 최적화 PASS |
| 학습 — analytic | train $d$: 6.63→1.55, val 2.55, 최종 equivariance 2.7e-15 | 학습 PASS, 표현력 gap |
| 학습 안정성 | 두 target 모두 9,600 step 동안 clamp 0, NaN/Inf 0 | PASS |
| 시각화 | spatial frame에서 translation에 따라 moment ellipsoid 변형, body pullback은 고정; naive head는 drift | 정성 확인 |
| 새 경로 비교 | wrench lift = $Q$ twist lift 오차 0; shared-weight 최종 $K$ 차이 2.2e-16; 두 경로 최대 equivariance 약 1.2e-12 | 동일성 확인 |
| 새 Klein gate | cross-slot 경로 복구, bounded gate equivariant; normalized/direct ReLU는 null cone·scale 폭증 | gate만 유망 |

## 6. 구현 결정

### 유지

- Bias 없는 `LNLinear`
- 타입에 맞는 twist bracket 또는 covector bracket
- Stiffness는 wrench feature $L$을 먼저 출력하고 $K=LL^\top$로 구성
- `float64`, $p\ne0$ translation sweep, 의도적 negative control
- Affine-invariant SPD loss

### 금지

- Twist bracket을 wrench에 직접 적용
- $K+\epsilon I$
- $K$에서 Cholesky factor를 뽑아 equivariant $L$이라고 해석
- Klein self-norm, $\sqrt{|B_Q(x,x)|}$ 정규화, $B_Q(d,d)$ 나눗셈
- Euclidean dot-product gate
- “covector로 바꾸면 bracket의 cross-slot blockage가 사라진다”는 가정

### 다음 실험

1. `bounded Klein gate + bracket` 모델을 실제 analytic contact-spring target에 넣는다.
2. bracket-only와 파라미터 수·데이터·seed를 맞춰 최소 3 seed로 val $d$를 비교한다.
3. gate score 분포, saturation 비율, gradient norm을 기록한다.
4. vector와 covector 구현 중 하나만 주 경로로 선택한다. 수학적 표현력은 같으므로 물리적 입력/출력 타입과 코드 단순성이 선택 기준이다.

## 7. 재현과 파일 지도

```bash
conda run -n lieneurons python experiment/pc_se3_congruence/verify.py
conda run -n lieneurons python experiment/pc_se3_congruence/check_killing_degeneracy.py
conda run -n lieneurons python experiment/pc_se3_congruence/analyze_klein_gate.py
conda run -n lieneurons python experiment/pc_se3_congruence/train.py --quick
```

| 파일 | 역할 |
|---|---|
| `README.md` | 현재 기준 결론; 이후 작업에서 가장 먼저 갱신 |
| `pc_se3_congruence_report.md` | O/L/A/B/C 구조 검증의 상세 보고서 |
| `train_report.md` | 학습·loss·dtype 버그 상세 보고서 |
| `covector_ln_framework.md` | covector-native 설계와 물리적 해석 |
| `results.json` | `verify.py`의 재현 수치 |
| `train_results/*_results.json` | 150-epoch 학습 history와 구조 검사 |
| `analyze_klein_gate.py` | 두 경로 동치, cross-slot blockage, Klein gate/Relu 실험 |
| `klein_gate_results.json` | 새 실험의 전체 수치 |
| `figs/` | experiment B의 spatial/body/negative-control 시각화 |

마지막 갱신: 2026-08-03.

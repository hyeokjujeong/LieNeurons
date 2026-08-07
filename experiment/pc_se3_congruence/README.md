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

## 6.5 Pointwise pipeline (구조 검증 + full GPU 학습 완료)

정식 실험 문서는 `pointwise_pipeline.md`이고 아래는 요약이다. 요지는 **neighbor 축
$k$를 backbone 이전에 learned set aggregation으로 없애고, point 축 $N$은 second
moment 직전까지 유지**하는 것이다.

$$
P\to W^{\mathrm{edge}}[N,k,6]\to X^{(0)}[N,C_0,6]
\to\text{LN-Linear/covector bracket/Klein gate}\to Z[N,H,6]\to\mathbf K .
$$

$N$을 유지하면 대칭이 factor를 *고정*하는 대신 *permutation*할 수 있다
($L(T\!\cdot\!P)=A_TL(P)(\Pi_T\otimes I_H)$, orthogonal gauge는 Gram에서 소거).
Global vector pooling이 centro/$C_2$에서 rank를 무너뜨린 이유가 이 gauge의 상실이다.

확인된 것 (`run_pointwise_gpu_experiments.sh`: Phase 1 구조 검증 + Phase 2–3 full 학습
18 run, CUDA·float64·150 epoch·4096 샘플, 2026-08-07):

- iid/centro/c2/tetra 전부 equivariance $\le1.7\times10^{-15}$,
  permutation $\le6.0\times10^{-16}$, rank 6, $\lambda_{\min}\sim10^{-3}$,
  `graph_truncation_frac` 0. 15개 아키텍처 변형 전부 동일.
- **tie 실패의 원인 규명.** 정육면체 격자에서 pointwise 모델의 permutation 오차는
  3.0e-16, rank-channel 모델(`WrenchSecondMomentModel` + LN backbone)은 1.9e-01이다.
  jitter $10^{-8}$로 tie를 깨면 후자도 즉시 회복된다. 즉 원인은 compact kernel이나
  adaptive radius가 아니라 **neighbor rank를 channel index로 쓴 것**이며,
  `knn_adaptive` radius를 쓴 pointwise 모델도 격자에서 machine precision을 유지한다.
- **Realizability.** 같은 클래스의 고정 난수 teacher 타깃에서 표준 suite 9 케이스
  (centro $\eta\in\{0,0.02,0.1,0.5\}$, $C_2$, tetra, iid $N\in\{32,128,512\}$) 전부
  val $d\le1.5\times10^{-3}$, ff/mm 블록 상대오차 $\le5.5\times10^{-4}$, train≈val.
  대칭 데이터셋과 iid의 수렴값이 구별되지 않는다.
- **학습 18 run 전부에서 rank 6, `clamped`$=0$, `equiv_err_final`$\le8.9\times10^{-15}$.**
  gradient step이 구조를 훼손하지 않는다.
- **Analytic(`kernel`) 타깃은 val $d$ 0.297–1.718에서 평탄화한다.** ep30 이후 거의
  움직이지 않고 train/val이 붙어 있으므로 과적합도 학습 부족도 아닌 **모델 클래스
  불일치의 바닥**이다 (아래 참조). iid에서 $N=32/128/512$에 대해 1.718/0.708/0.297로
  단조 감소한다.
- **Encoder는 학습 파라미터가 0개다.** $k\to C_0$ 축약을 학습된 MLP 대신 고정 거리
  shell $\rho_c(q_{ij})$로 하고 shell별 정규화만 남기면(`--pw-pool basis_mean`, 기본값)
  가중치 MLP가 불필요하다 — 뒤따르는 LN-Linear가 $\mathrm{span}\{\rho_c\}$ 안의 임의
  radial kernel을 복원하기 때문이다. 같은 논리로 encoder bracket도 block 0의 bracket과
  중복이라 제거했다(`--pw-bracket none`, 기본값). 1,896 → **0개**. CPU smoke 기준으로
  analytic 타깃 4개 데이터셋 평균 val $d$는 1.338로 attention(1.398)·
  encoder-bracket(1.350)보다 낫고, iid에서만 attention이 앞선다(1.200 vs 1.442).
  full recipe 재확인은 아직이다. 상세는 `pointwise_pipeline.md` §5.6.
- **실무 주의.** 기본 반경 설정(`global_scale` $\alpha=0.75$, `candidate_k=32`)은
  $N=128$에서 `graph_truncation_frac`이 0.36까지 오른다. GPU 실험에서는
  `density_scaled` $\alpha=1.15$, $k_{\rm target}=16$, `candidate_k=64`를 쓴다
  (`run_pointwise_suite.py`의 기본값). 이 설정에서도 $N=512$는 trunc 3.7e-04로 0이
  아니므로 그보다 큰 $N$에서는 `--pw-candidates`를 올려야 한다.

analytic contact-spring 타깃은 raw wrench의 second moment이고 이 모델은 latent
covector의 second moment이므로 **matched target이 아니다** — 잔차를 표현력 부족으로
읽기 전에 `--phase teacher`를 먼저 확인해야 한다. 위 결과는 그 순서로 읽은 것이다:
teacher가 $10^{-3}$까지 내려가므로 analytic의 $10^{-1}$대 바닥은 최적화 문제가 아니다.

아직 하지 않은 것: ablation 정량 비교(`--phase ablation`), seed 분산, analytic 잔차를
이산화 오차와 함수족 차이로 분해하는 것. 또한 `err_rel_fm`은 centro $\eta=0$과
tetra에서 타깃의 $K_{fm}$ 블록이 정확히 0이라 분모가 0이므로 읽지 않는다.

## 7. 재현과 파일 지도

```bash
conda run -n lieneurons python experiment/pc_se3_congruence/verify.py
conda run -n lieneurons python experiment/pc_se3_congruence/check_killing_degeneracy.py
conda run -n lieneurons python experiment/pc_se3_congruence/analyze_klein_gate.py
conda run -n lieneurons python experiment/pc_se3_congruence/train.py --quick
conda run -n lieneurons python experiment/pc_se3_congruence/verify_pointwise.py --full
conda run -n lieneurons python -m pytest -q test/test_pc_pointwise_pipeline.py
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
| `tensor_kernel_experiment_report.md` | compact kernel + second moment 보고서 |
| `pointwise_pipeline.md` | pointwise 파이프라인 실험 보고서 (설계·구조 검증·full GPU 결과) |
| `pointwise_graph.py` | tie-safe local graph, Wendland window, edge invariant |
| `pointwise_models.py` | set encoder / message passing / Klein gate / late Gram head |
| `verify_pointwise.py` | pointwise 구조 검증 A–F + ablation 표 |
| `run_pointwise_suite.py` | verify / teacher / analytic / ablation phase 실행 |
| `run_pointwise_gpu_experiments.sh` | Phase 1–3 일괄 실행 (§6.5 수치의 출처) |
| `pointwise_verify_results.json` | Phase 1 구조 검증 수치 |

마지막 갱신: 2026-08-07 (pointwise pipeline full GPU 결과 반영).

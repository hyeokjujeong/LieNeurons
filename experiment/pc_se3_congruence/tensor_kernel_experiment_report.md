# Local Compact-Kernel + Second-Moment Stiffness 실험 보고서

**실험일:** 2026-08-06  
**W&B project:** `adjoint_equivariant_network/pc-se3-congruence`  
**정밀도/장치:** `torch.float64`, CUDA  
**목적:** all-pairs 없이 local geometry만 사용하면서, 대칭 point cloud에서 발생한
global vector pooling의 rank collapse와 hard top-k 경계 불안정성을 함께 해결할 수
있는지 검증한다.

---

## 0. 결론

> **32개의 local 후보, 경계에서 0이 되는 compact kernel, 그리고 vector 평균보다
> 먼저 수행하는 second-moment pooling을 결합하면, 본 synthetic benchmark의 모든
> 대칭 조건에서 full-rank이고 수치적으로 정확한 SE(3)-equivariant stiffness를
> 학습할 수 있었다.**

전체 9개 full run에서 공통으로 다음 결과를 얻었다.

- `rank_pred = 6`
- 모든 epoch에서 `clamped = 0`
- `equiv_err_final = 2.03e-15 ~ 8.19e-15`
- `val_d = 3.76e-4 ~ 6.09e-4`
- median `lam_min = 5.94e-4 ~ 3.35e-3`: 수치 잡음으로 만들어진 가짜 rank 6이 아님
- 정확한 대칭을 가진 `centro eta=0`, `C2`, `tetra`에서도 collapse 없음

따라서 기존 실패의 직접 원인은 “대칭 물체에는 stiffness를 정의할 수 없음”이
아니라 다음 두 설계 선택이었다는 해석과 일치한다.

1. 대칭 관련 edge vector를 먼저 global mean하여 1차 모멘트를 없앤 것
2. 동일/근접 거리 shell을 hard top-k로 절단하여 이웃 선택을 불연속적으로 만든 것

단, 본 실험의 target도 동일한 compact-kernel second-moment 법칙으로 만들었으므로
이 결과는 **구조적 정확성, rank 보존, equivariance 및 최적화 가능성에 대한
matched-target 검증**이다. 임의의 실제 stiffness를 모두 표현한다는 증명은 아니다.

---

## 1. 기존 문제

### 1.1 Global vector mean에 의한 정보 소실

기존 Plücker encoder는 점 $i$와 이웃 $j$로부터 wrench

$$
w_{ij}=\begin{bmatrix}f_{ij}\\m_{ij}\end{bmatrix},\qquad
f_{ij}=p_j-p_i,\qquad m_{ij}=p_i\times f_{ij}
$$

를 만든 뒤, point 축에 대해 먼저 평균했다.

$$
\mu_c=\frac1N\sum_i w_{i,c}.
$$

대칭 관련 edge가 $w$와 $-w$처럼 나타나면 1차 모멘트 $\mu_c$에서 상쇄된다.
정확한 proper symmetry $H$를 갖는 cloud에서는 global equivariant vector가
$\operatorname{Fix}_H(\rho)$ 안에 놓여야 하므로, 채널 수를 증가시키거나 lift를
learnable하게 바꾸어도 물리 방향 span을 복구할 수 없다.

### 1.2 Hard top-k의 tie 및 near-tie

거리로 정렬한 $k$번째와 $k+1$번째 이웃의 거리가 같거나 매우 가까우면, 작은
부동소수점 오차나 point permutation으로 선택된 이웃이 바뀔 수 있다. 선택된 edge의
가중치가 유한하면 이 교체는 출력의 $O(1)$ 변화가 될 수 있다.

정확한 tie는 continuous random scan에서는 드물지만 다음 데이터에서는 구조적으로
발생한다.

- CAD/mesh의 규칙적 샘플
- voxel 또는 quantized point cloud
- 원, 격자, 정다면체와 같은 대칭 형상
- 균일 각도 간격을 갖는 센서 데이터

all-pairs는 이 문제를 제거하는 좋은 대조군이지만 edge 저장과 tensor 구성이
$O(N^2)$이므로 최종 local architecture로는 부적합하다.

---

## 2. 제안한 local tensor 방법

### 2.1 Adaptive local 후보

각 anchor point $i$에서 최근접 후보를 최대 $K_c=32$개만 유지한다. 점 개수가
32여서 $N-1<K_c$인 경우에는 자동으로 $K_c=N-1$을 사용한다.

가장 먼 후보의 거리를 local support radius로 둔다.

$$
r_i=d_{i,(K_c)},\qquad q_{ij}=\frac{\lVert p_j-p_i\rVert}{r_i}.
$$

이는 하나의 고정된 물리 radius가 아니라, point density에 따라 변하는 adaptive
local radius이다. 서로 다른 $N$과 밀도를 갖는 전체 suite를 bounded degree로
처리하기 위해 선택했다.

### 2.2 Compact Wendland kernel

후보 edge에는 C2 Wendland window를 곱한다.

$$
\phi(q)=
\begin{cases}
(1-q)^4(1+4q), & 0\le q<1,\\
0, & q\ge1.
\end{cases}
$$

따라서 가장 바깥 후보는 항상 $\phi(1)=0$이다.

- exact tie shell이 후보 경계를 가르면 그 shell의 선택/비선택 edge가 모두 0 기여
- near-tie에서 이웃이 교체되어도 경계 edge의 기여가 매우 작음
- edge tensor 저장량은 $O(NK_c)$
- 후보 순서는 최종 합에서 제거되므로 채널 순열에도 불변

### 2.3 Learned radial weight와 second moment

compact window 안에서 positive radial rate를 학습한다.

$$
\alpha_\theta(d^2)
=\exp\!\left[-\operatorname{softplus}
\left(g_\theta(\log(1+d^2))\right)d^2\right].
$$

최종 stiffness는 vector 평균 후 outer product를 하는 대신, 각 edge에서 outer
product를 먼저 만든 후 합한다.

$$
K(P)=\frac{1}{NK_c}\sum_i\sum_{j\in\mathcal C(i)}
\phi(q_{ij})\,\alpha_\theta(\lVert f_{ij}\rVert^2)\,
w_{ij}w_{ij}^{\top}.
$$

대칭 edge $w,-w$에 대해

$$
w+(-w)=0,\qquad
ww^\top+(-w)(-w)^\top=2ww^\top
$$

이므로 1차 모멘트는 사라져도 stiffness 기여는 유지된다.

### 2.4 Equivariance와 PSD

rigid transform $T$에서 wrench가

$$
w_{ij}\mapsto A_Tw_{ij},\qquad A_T=\operatorname{Ad}_T^{-\top}
$$

로 변하고, 거리·radial weight·compact window는 invariant이므로

$$
K(T\cdot P)=A_TK(P)A_T^\top
=\operatorname{Ad}_T^{-\top}K(P)\operatorname{Ad}_T^{-1}.
$$

positive weight의 outer-product 합이므로 $K\succeq0$이며,

$$
\operatorname{rank}(K)
=\dim\operatorname{span}\{w_{ij}:\phi(q_{ij})\alpha_{ij}>0\}.
$$

즉 second moment는 mean cancellation에 의한 **가짜 rank deficiency**는 막지만,
실제 local wrench들이 6차원을 span하지 않으면 자동으로 full rank를 만들지는 않는다.

### 2.5 계산 복잡도

| 단계 | 현재 구현 | 비고 |
|---|---:|---|
| 이웃 거리 검색 | $O(N^2)$ | 현재 `torch.cdist` 사용 |
| 선택 edge/wrench 저장 | $O(NK_c)$ | all-pairs의 $O(N^2)$ edge tensor를 피함 |
| second-moment 합 | $O(NK_c\cdot6^2)$ | 모델 파라미터 수는 $N,K_c$와 무관 |

따라서 중간 feature와 tensor head는 local이지만, 대규모 cloud에서는 이웃 검색을
spatial hash, voxel grid 또는 radius-query backend로 교체할 여지가 남아 있다.

---

## 3. 구현

| 파일 | 변경 내용 |
|---|---|
| `encoders.py` | `compact_wendland_weights`, `WrenchEdgeEncoder(graph='kernel')` |
| `models.py` | learned/analytic radial weight와 compact window를 결합한 `WrenchSecondMomentModel` |
| `data_synth.py` | 모델과 동일한 local kernel target `contact_spring_kernel_K` |
| `blockage_bench.py` | `--tensor-graph kernel`, `--target-graph kernel`, `--kernel-candidates` CLI |
| `run_tensor_kernel_suite.py` | 전체 9개 조건을 순차 실행하는 runner |
| `test/test_pc_symmetry_tensor.py` | analytic exact match, permutation invariance, equivariance, rank 테스트 |

Tensor model에는 별도의 Lie-Neuron backbone을 두지 않았다. 이번 실험의 목적은
local wrench의 second moment와 radial kernel만으로 rank/equivariance 문제가
해결되는지를 최소 구성에서 분리해 확인하는 것이기 때문이다.

---

## 4. 실험 설정

### 4.1 전체 object suite

| Dataset | 조건 | Point 수 |
|---|---|---:|
| centro | $\eta=0,0.02,0.1,0.5$ | 128 |
| C2 | $\eta=0$ | 128 |
| tetra | $\eta=0$ | 120 (A4 orbit 때문에 12의 배수) |
| IID | $N=32,128,512$ | 32/128/512 |

`fiber`는 학습 object distribution이 아니라 기존 vector summary의 collision을
보이는 eval-only 진단이므로 suite에서 제외했다.

### 4.2 공통 hyperparameter

| 항목 | 값 |
|---|---:|
| recipe | `full` |
| epochs | 150 |
| train / validation | 4096 / 512 |
| batch | 64 |
| optimizer | Adam |
| initial learning rate | `1e-3` |
| scheduler | cosine, final LR `1e-5` |
| kernel candidates | 32 |
| target/model graph | `kernel` / `kernel` |
| target radial sigma | 0.5 |
| tensor radial weight | `learned` |
| dtype | float64 |

Target은 동일한 compact window와 analytic radial spring
$\exp(-d^2/(2\sigma^2))$로 생성했다. 모델은 target의 $\sigma$를 직접 받지 않고
positive radial rate를 학습한다.

---

## 5. 결과

### 5.1 핵심 지표

| Dataset | W&B run | Final `val_d` | Best `val_d` (epoch) | Equiv. final | Rank | Median $\lambda_{min}$ | Median $\lambda_{max}$ | Max clamp |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| centro $\eta=0$ | [r9q0l7kb](https://wandb.ai/adjoint_equivariant_network/pc-se3-congruence/runs/r9q0l7kb) | 5.893e-4 | 5.795e-4 (147) | 4.43e-15 | 6 | 2.682e-3 | 1.057e-2 | 0 |
| centro $\eta=.02$ | [7djforvp](https://wandb.ai/adjoint_equivariant_network/pc-se3-congruence/runs/7djforvp) | 6.013e-4 | 5.950e-4 (147) | 3.79e-15 | 6 | 2.686e-3 | 1.059e-2 | 0 |
| centro $\eta=.1$ | [01mq5crt](https://wandb.ai/adjoint_equivariant_network/pc-se3-congruence/runs/01mq5crt) | 5.737e-4 | 5.698e-4 (144) | 3.53e-15 | 6 | 2.697e-3 | 1.058e-2 | 0 |
| centro $\eta=.5$ | [wuv5m4hv](https://wandb.ai/adjoint_equivariant_network/pc-se3-congruence/runs/wuv5m4hv) | 6.086e-4 | 5.994e-4 (144) | 3.78e-15 | 6 | 2.688e-3 | 1.071e-2 | 0 |
| C2 $\eta=0$ | [nvk1gle3](https://wandb.ai/adjoint_equivariant_network/pc-se3-congruence/runs/nvk1gle3) | 5.617e-4 | 5.610e-4 (148) | 3.42e-15 | 6 | 2.677e-3 | 1.063e-2 | 0 |
| tetra $\eta=0$ | [k6i04g3s](https://wandb.ai/adjoint_equivariant_network/pc-se3-congruence/runs/k6i04g3s) | 5.507e-4 | 5.492e-4 (148) | 3.62e-15 | 6 | 3.350e-3 | 4.973e-3 | 0 |
| IID $N=32$ | [azmb1twq](https://wandb.ai/adjoint_equivariant_network/pc-se3-congruence/runs/azmb1twq) | 6.020e-4 | 6.020e-4 (149) | 2.03e-15 | 6 | 5.936e-4 | 2.246e-2 | 0 |
| IID $N=128$ | [w8pnedfs](https://wandb.ai/adjoint_equivariant_network/pc-se3-congruence/runs/w8pnedfs) | 5.988e-4 | 5.973e-4 (147) | 3.87e-15 | 6 | 1.045e-3 | 1.916e-2 | 0 |
| IID $N=512$ | [bcog72cy](https://wandb.ai/adjoint_equivariant_network/pc-se3-congruence/runs/bcog72cy) | **3.762e-4** | **3.746e-4** (142) | 8.19e-15 | 6 | 1.128e-3 | 1.808e-2 | 0 |

모든 run은 W&B에서 `finished` 상태다. 9개 run의 W&B runtime 합은 약 681초
(11분 21초)였다. IID $N=512$는 약 256초로 가장 느렸고, 현재 `torch.cdist`
neighbor search의 $O(N^2)$ 비용이 주요 확장성 한계임을 보여준다.

### 5.2 Block 상대오차

| Dataset | `err_rel_ff` | `err_rel_fm` | `err_rel_mm` |
|---|---:|---:|---:|
| centro $\eta=0$ | 1.858e-4 | 1.465 | 2.522e-4 |
| centro $\eta=.02$ | 1.878e-4 | 1.787e-3 | 2.561e-4 |
| centro $\eta=.1$ | 1.688e-4 | 1.572e-3 | 2.274e-4 |
| centro $\eta=.5$ | 1.597e-4 | 1.285e-3 | 2.178e-4 |
| C2 | 1.536e-4 | 1.099e-3 | 2.100e-4 |
| tetra | 1.598e-4 | 3.586e-1 | 2.537e-4 |
| IID $N=32$ | 1.441e-4 | 2.209e-4 | 1.757e-4 |
| IID $N=128$ | 1.635e-4 | 2.557e-4 | 2.010e-4 |
| IID $N=512$ | 1.134e-4 | 1.652e-4 | 1.300e-4 |

`centro eta=0`과 `tetra`의 FM 상대오차는 모델 실패가 아니다. 동일 seed와
validation split을 재생성하여 target FM block의 절대 norm을 측정하면 다음과 같다.

| Dataset | Target FM norm 평균 | 중앙값 | 최대값 |
|---|---:|---:|---:|
| centro $\eta=0$ | 1.385e-19 | 1.303e-19 | 4.985e-19 |
| tetra $\eta=0$ | 4.887e-19 | 4.435e-19 | 1.570e-18 |
| centro $\eta=.02$ (비교) | 1.019e-4 | 9.806e-5 | 2.363e-4 |

정확한 대칭에서 target FM이 round-off 수준으로 0이므로, `num/den` 형태의
상대오차 분모가 붕괴한다. 이 두 조건은 전체 AIRM `val_d`, FF/MM 오차,
eigenvalue 및 절대 FM error로 판단해야 한다.

### 5.3 Analytic sanity와 단위 테스트

학습 전 구조 검증을 위해 analytic radial head와 동일한 kernel target을 사용한
quick suite도 실행했다. 9개 조건 모두 다음을 만족했다.

- `val_d = 0`
- `err_rel_ff = err_rel_mm = 0`
- `rank_pred = 6`
- equivariance `2.03e-15 ~ 7.42e-15`

또한 다음 테스트 4개가 모두 통과했다.

```text
test_analytic_second_moment_matches_all_pair_target
test_all_pair_second_moment_is_full_rank_and_equivariant_on_c2
test_analytic_second_moment_matches_local_kernel_target
test_local_kernel_tensor_is_permutation_invariant_and_equivariant_on_c2

4 passed
```

---

## 6. 결과 해석

### 6.1 Rank deficiency 해결

정확한 centro, C2, tetra 대칭에서 모두 `rank=6`, `clamped=0`이고
$\lambda_{min}$이 $10^{-3}$ 수준이다. 따라서 기존 global-vector encoder에서 보인
rank collapse는 Plücker line coordinate 자체의 필연적 한계가 아니라, edge vector의
1차 모멘트만 남긴 조기 pooling의 결과였다는 진단을 지지한다.

### 6.2 all-pairs 없이 tie 문제 해결

tetra처럼 구조적인 거리 tie를 갖는 데이터에서도 equivariance error가
$3.62\times10^{-15}$이다. 경계 edge의 compact weight를 정확히 0으로 만든 것이
arbitrary boundary selection의 영향을 제거했다. 따라서 모든 pair를 유지하지 않고도
bounded local representation으로 tie-robust tensor를 만들 수 있다.

### 6.3 Learned kernel의 최적화

analytic target을 직접 복사하지 않고 radial MLP가 positive inverse length scale을
학습했음에도 모든 validation AIRM이 $10^{-4}$ 수준으로 수렴했다. train/validation
차이도 작다. best epoch가 대부분 142~149에 있어 150 epoch가 과도하게 길지는 않았고,
cosine schedule 후반까지 미세하게 개선되었다.

### 6.4 Point 수 변화

IID의 $N=32,128,512$에서 모두 rank와 equivariance가 유지된다. 기존 global mean
encoder는 $N$ 증가와 함께 방향 신호가 통계적으로 감소했지만, second moment는
edge energy를 평균하므로 이러한 1차 평균 상쇄가 발생하지 않는다.

---

## 7. 이 실험이 증명하지 않는 것

### 7.1 Matched target

Target과 model이 모두

$$
\sum_{ij}\alpha(d_{ij})w_{ij}w_{ij}^\top
$$

형태다. 따라서 이번 실험은 이 함수족에서의 realizability와 최적화를 확인하지만,
형식이 다른 실제 stiffness에 대한 일반화는 별도 실험이 필요하다.

### 7.2 Second moment의 비단사성

서로 다른 wrench distribution이 같은 second moment를 가질 수 있고, $w$와 $-w$는
구분되지 않는다. stiffness처럼 방향축의 부호가 중요하지 않은 target에는 적합하지만,
oriented force, odd moment, 일부 chirality 정보를 필요로 하는 task에는 충분하지 않다.

### 7.3 Pure-force Plücker wrench의 한계

현재 edge는 $m=p\times f$인 zero-pitch pure-force wrench이므로 항상
$f^\top m=0$이다. 따라서 독립적인 pure couple, force 방향과 평행한 torque,
nonzero-pitch screw 또는 일반적인 임의의 $6\times6$ PSD stiffness를 모두 직접
표현한다고 볼 수 없다. 실제 target에 torsional stiffness가 있다면 couple/screw
feature를 추가해야 한다.

### 7.4 Full rank는 보장값이 아님

이번 데이터에서는 active wrench들이 $\mathbb R^6$을 span했기 때문에 rank 6이었다.
점들이 선이나 특정 평면에 놓여 wrench span이 작아지면 tensor도 실제 rank deficiency를
그대로 반영한다. 이는 정보 손실이 아니라 물리적으로 관측되지 않는 운동 방향이다.

### 7.5 Adaptive radius

현재 kernel support는 고정된 물리 radius가 아니라 32번째 후보 거리로 정해진다.
따라서 density가 달라도 bounded degree를 유지하지만, 실제 contact radius가 주어진
문제에서는 fixed-radius graph와 smooth cutoff가 더 물리적일 수 있다.

---

## 8. 재현 방법

### 8.1 빠른 analytic 구조 검증

```bash
conda run -n lieneurons python \
  experiment/pc_se3_congruence/run_tensor_kernel_suite.py \
  --quick \
  --phase sanity \
  --wandb-mode disabled
```

### 8.2 본 보고서의 full learned suite

```bash
conda run -n lieneurons python \
  experiment/pc_se3_congruence/run_tensor_kernel_suite.py \
  --recipe full \
  --phase train \
  --kernel-candidates 32 \
  --wandb-mode online
```

### 8.3 중간 규모 실험

```bash
conda run -n lieneurons python \
  experiment/pc_se3_congruence/run_tensor_kernel_suite.py \
  --phase train \
  --kernel-candidates 32 \
  --wandb-mode online \
  -- \
  --epochs 60 \
  --n-train 512 \
  --n-val 128 \
  --batch 32 \
  --lr 1e-3
```

### 8.4 단위 테스트

```bash
conda run -n lieneurons python -m pytest -q \
  test/test_pc_symmetry_tensor.py
```

---

## 9. 다음 권장 실험

1. **Graph/Pooling ablation**
   - hard-kNN + first moment
   - hard-kNN + second moment
   - compact kernel + second moment
   - all-pairs + second moment
   를 동일 target에서 비교하여 두 개선 요소의 기여를 분리한다.

2. **Kernel candidate 수 sweep**
   - $K_c\in\{8,16,32,64\}$에서 accuracy, equivariance, runtime,
     $\lambda_{min}$을 비교한다.

3. **Near-tie robustness curve**
   - 좌표 noise 크기를 변화시키며 neighbor set Jaccard와 output equivariance error를
     함께 측정한다.

4. **Mismatched target**
   - kernel encoder가 hard-kNN, radius, FEM 또는 실측 stiffness target을 얼마나
     근사하는지 확인한다. 단, exact symmetry의 hard-kNN target 자체가 tie-breaking에
     의존할 수 있으므로 target equivariance부터 별도로 검사해야 한다.

5. **General wrench 확장**
   - pure couple와 nonzero-pitch screw feature를 추가하여 pure-force Plücker cone의
     표현 제약을 측정한다.

6. **Sparse neighbor backend**
   - `torch.cdist`를 spatial hash/radius query로 교체하여 neighbor search까지
     $O(NK)$에 가깝게 만든다.

---

## 10. 최종 판단

이번 결과는 다음 설계 원칙을 지지한다.

$$
\boxed{
\text{local smooth graph}
\;\longrightarrow\;
\text{pointwise Plücker wrenches}
\;\longrightarrow\;
ww^\top\text{ before pooling}
\;\longrightarrow\;
\text{global stiffness}
}
$$

즉 point cloud를 먼저 하나의 global equivariant vector 집합으로 압축하기보다,
local edge 수준에서 stiffness의 에너지 기여인 second moment를 만든 뒤 합해야 한다.
이 방식은 본 실험의 모든 대칭 object에서 rank, equivariance, permutation robustness와
학습성을 동시에 만족했으며, all-pairs 없이도 동작했다.

# Peg-and-Hole PCD 데이터셋

**작성일:** 2026-08-08
**코드:** `experiment/pc_se3_congruence/peg_hole_synth.py`(장면·라벨),
`data_gen/gen_peg_hole_pcd.py`(생성 CLI), `data_loader/peg_hole_data_loader.py`(로더),
`test/test_peg_hole_dataset.py`(테스트 14종)
**현재 정의:** v2 — 라벨이 **면적 가중 Monte-Carlo 면적분**이라 해상도에 수렴한다(§2).
**산출물:** train 102,400 / val 10,240 / test 10,240 쌍, cloud당 **N = 2048점**,
총 122,880쌍 (~3.2 GB, npz 샤드 60개). 생성 명령은 §4.
`data/peg_hole/v1`은 구 라벨(해상도 의존)이라 **v2로 재생성해야 한다.**

---

## 0. 목적

선행 실험(`pointwise_pipeline.md`)의 cloud는 비등방 Gaussian blob·대칭 orbit 등
**추상적 합성 데이터**였다. 본 데이터셋은 같은 (cloud, $K$) supervised 회귀 구도를
유지하면서 cloud를 **실제 조립 작업처럼 생긴 장면** — 프리즘 peg + 관통 구멍 plate —
의 표면 PCD로 교체한다. 세 가지를 얻는다.

1. **실물성.** 표면-샘플 PCD(면적 가중, 센서 노이즈), CAD형 부품 형상, 조립 단계
   (접근/탐색/삽입)의 pose 분포.
2. **대칭 벤치의 실물 대응.** peg 단면이 곧 대칭 클래스다: 원($C_\infty$),
   정삼각·정사각·정육각($C_3,C_4,C_6$), 직사각·타원($C_2$), D형(비대칭 대조군).
   선행 rank-collapse·tie 실험의 추상 대칭(centro/c2/tetra)이 실물 부품 대칭으로
   번역된다.
3. **물리적으로 유의미한 라벨.** 접촉 강성 $K_{\text{contact}}$는 삽입 상태의
   **구속 구조**를 그대로 스펙트럼에 반영한다 — 무접촉이면 정확히 0, 표면 접촉이면
   저랭크, 깊은 삽입이면 벽 방향이 잠긴 고랭크.

---

## 1. 장면 생성

### 1.1 좌표계와 부품

- **canonical(hole) 프레임:** plate 윗면 $z=0$, plate는 $z\in[-T,0]$, 구멍 축 $+z$
  (원점 통과). **peg 프레임:** 압출 축 $+z$, tip 단면 $z=0$, 상단 $z=H$;
  pose $(R_{\rm peg}, p_{\rm peg})$, $p_{\rm peg}$는 tip 중심.
- **단면 = 볼록 다각형** (`ConvexProfile`). 매끄러운 프로파일(원/타원/D형)은
  128각형 — 현 오차(상대 면적 ~4e-4, 반경 ~7.5e-5·r)는 센서 노이즈(1e-3~5e-3)
  아래. 다각형 통일 덕에 경계/내부 균일 샘플링(호장·삼각 fan), 포함 판정
  (edge-plane margin), 스케일·회전이 전부 정확·벡터화된다.
- **구멍 = peg 단면을 $(1+c)$배 스케일 + hole_yaw 회전**, clearance 비율
  $c\sim U[0.02, 0.08]$. plate는 $W_{x,y}\sim U[1.8,3.0]$, $T\sim U[0.25,0.6]$ 블록.
- **peg:** 외접반경 $r\sim U[0.22,0.45]$, 높이 $H\sim U[0.9,1.6]$, 확률 0.6으로
  tip 챔퍼(높이 $U[0.05,0.12]H$, tip 스케일 $U[0.75,0.9]$).

### 1.2 Stage 혼합 (free 0.15 / search 0.25 / insert 0.60)

| stage | 정의 | pose 샘플링 |
|---|---|---|
| free | 접촉 없음 | 최저점 gap $\in[0.33,0.85]$ (**접촉 반경보다 크게** → $K_{\text{contact}}\equiv 0$), tilt $\le20^\circ$, yaw 자유 |
| search | tip이 구멍 주변 표면에 접촉/스침 | 최저점 gap $\in[0.002,0.02]$, offset $\in[0.3,1.3]\,r_{\rm hole}$ (rim 주변), tilt $\le8^\circ$ |
| insert | 부분~깊은 삽입 | depth $\in[0.06,1.0]\cdot d_{\max}$, $d_{\max}=\min(0.9H,\,1.15T)$; offset·yaw 불일치·tilt를 clearance 예산 안에서 제안 후 **무관통 검사**로 기각 샘플링 (40회, 실패 시 축소) |

**무관통 검사 (insert).** 프리즘 표면은 outline ring 사이의 ruled surface이므로,
볼록성에 의해 plate 밴드 $z\in[-T,0]$를 **감싸는** ring들의 xy-사영이 구멍 다각형
안에 있으면(margin $=0.1\,c\cdot r_{\rm in}$) 전 표면이 안에 있다. tilt $\le3^\circ$
여유를 포함해 ring 높이를 $[\max(0,d-T-0.05),\ d+0.08]$로 잡는다. free/search는
최저 ring을 원하는 gap에 정확히 놓아 구성상 관통이 불가능하다.
테스트가 전 stage·전 프로파일에서 점 단위로 재검증한다.

### 1.3 표면 PCD

- 부품별 점 배분: 표면적 비율(단, peg 최소 30%·최대 55% 클램프), 부품 내부는
  면(윗면−구멍, 아랫면−구멍, 외측 4면, **구멍 내벽**, peg 옆면·챔퍼 밴드·양 캡)
  면적-가중 multinomial.
- 등방 센서 노이즈 $\sigma\sim U[10^{-3},5\times10^{-3}]$, 장면 전체 랜덤 SE(3)
  ($R$ 균일, $\|p\|\sim1$ — 파이프라인의 $\sigma(P)\sim O(1)$ 규약 유지), 점 순서 셔플.
- `part` 마스크(1=peg, 0=plate)를 함께 저장 — 라벨 재계산·분석용. 모델 입력은
  cloud만으로도 되고(part는 형상에서 원리적으로 복원 가능), part를 쓰는 실험도 가능.

---

## 2. 라벨

### 2.1 $K_{\text{contact}}$ — 접촉 면적분의 Monte-Carlo 추정

물리적으로 접촉 강성은 두 표면에 대한 **이중 면적분**이다.

$$K=\iint_{\Gamma_{\rm peg}\times\Gamma_{\rm plate}}
\kappa(\lVert x-y\rVert)\;w\,w^{\top}\,dA(y)\,dA(x)$$

균일 표면 샘플에서 점 하나는 면적 $A/N$을 대표하므로 추정량은

$$K_{\text{contact}}=\frac{A_{\rm peg}}{N_{\rm peg}}\cdot\frac{A_{\rm plate}}{N_{\rm plate}}
\sum_{i\in{\rm peg}}\ \sum_{\substack{j\in{\rm plate}\\ \lVert d_{ij}\rVert<r_c}}
\kappa(\lVert d_{ij}\rVert)\,w_{ij}w_{ij}^{\top},$$

$$w_{ij}=\begin{bmatrix}u_{ij}\\ r_i\times u_{ij}\end{bmatrix},\quad
u_{ij}=\frac{d_{ij}}{\lVert d_{ij}\rVert},\qquad
\kappa(d)=e^{-d^2/2\sigma_c^2}\,\phi(d/r_c),\quad r_c=3\sigma_c .$$

설계 논거 넷:

1. **면적 가중치가 라벨을 물리량으로 만든다.** 종전의 $1/(N_{\rm peg}k)$ 정규화는
   해상도에 따라 라벨이 계통적으로 변했다 — 같은 접촉인데 $\lVert K_c\rVert$가
   $N$=2048/1024/512에서 9.6e-3 / 7.5e-3 / 4.8e-3 (AIRM 1.6–2.8 차이). 즉 "정답"의
   일부가 기하가 아니라 이산화였다. 면적 가중 후에는 8배 밀도 범위에서 **편향이 없고**
   ($\lVert K\rVert$ 변동 13%) 잔차는 순수 MC 분산이다 (§2.3).
2. **반경 안 쌍을 전부 센다.** $k$번째 이웃으로 자르면 그 절단 자체가 다시 밀도
   의존이 된다 (실측: $N$=2048에서 반경 안에 최대 63개인데 $k$=12로 잘림). 후보 예산이
   부족하면 `ValueError`로 즉시 중단한다 — 잘린 라벨은 조용히 샘플링의 함수가 된다.
3. **Unit 방향.** $w=(d,\ r\times d)$를 쓰면 기여가 $\lVert d\rVert^2$에 비례해
   접촉 신호가 표면 점 간격에 묻힌다. 단위 line-spring은 닿은 쌍이 gap 크기와 무관하게
   $O(1)$을 기여한다. $\lVert d\rVert$가 불변량이라 등변성은 그대로고, 두 끝점이 같은
   접촉선 위에 있어 anchor 선택도 무관하다 ($r_j\times u=r_i\times u$).
4. **Hard support** $r_c=3\sigma_c$. Gaussian 꼬리를 잘라 **무접촉 ⇒ 정확히 0**.
   free stage의 최소 gap 0.33 > $r_c+3\sigma_{\rm noise}$라 노이즈 하에서도 유지된다.

가중이 전부 SE(3)-불변이고 $w$가 coadjoint로 변환하므로 congruence 등변성이 **정확히**
성립한다 (테스트 실측 < 5e-9). 순열 불변·PSD.

**스펙트럼 = 구속 구조.** 무접촉 rank 0, 표면 탐색은 tip↔윗면 수직 springs(저랭크),
깊은 삽입은 벽 법선 방향들이 잠긴 고랭크.

### 2.2 $K_{\text{body}}$와 합성

$K_{\text{body}}$는 **같은 추정량에서 cross-part 제약만 뺀 것**(전 쌍, 자체 반경
$r_b=3\sigma_b$, $\sigma_b=0.05$)이다. 항상 rank 6 SPD라 합성 타깃이 SPD가 되어 AIRM이
정의된다. 학습 타깃은 로더에서

$$K \;=\; K_{\text{contact}} \;+\; \lambda\,K_{\text{body}},\qquad \lambda=0.005$$

두 항을 **분리 저장**하므로 $\lambda$는 재생성 없이 스윕 가능.

**캘리브레이션** (48 × N=2048): $\lVert K_c\rVert$ 중앙값 — free **정확히 0** /
search 4.7e-3 / insert 2.2e-2, $\lVert K_b\rVert$ 2.5e-1. $\lambda=0.005$에서 접촉 시
contact 항이 $\lambda K_{\rm body}$의 **17배**, $\lambda_{\min}=1.2\times10^{-5}$,
cond 중앙값 120 · 최대 1300 (v1의 242 / 2e4보다 개선).

### 2.3 해상도 수렴 — 이 데이터셋의 핵심 성질

같은 장면을 서브샘플해 각 해상도에서 재계산한 라벨:

| $N$ | $\lVert K_c\rVert$ 중앙값 | $N$=2048 라벨과 AIRM | (v1 방식) |
|---|---|---|---|
| 2048 | 7.96e-3 | 0.000 | — |
| 1024 | 8.38e-3 | **0.506** | 1.57 |
| 512 | 6.85e-3 | **1.018** | 2.79 |
| 256 | 7.20e-3 | 1.864 | — |

잔차가 **편향이 아니라 MC 분산**임을 확인했다: $N$=1024 서브샘플을 4번 다르게 뽑으면
뽑기끼리 AIRM이 0.64–0.70으로 $N$=2048과의 거리(0.51)보다 오히려 크고,
$\lVert K\rVert$의 뽑기 간 변동은 13%다. 즉 라벨은 물리량의 **불편추정량**이며,
$N$을 키우면 참값으로 수렴한다.

### 2.4 정밀도 프로토콜

points는 float32로 저장하되 **라벨은 float32로 반올림된 점에서 float64로 계산**
→ `저장 K == 라벨함수(저장 points)`가 비트 수준으로 성립 (테스트 < 1e-14).
면적 $A_{\rm peg},A_{\rm plate}$는 라벨의 정확한 가중치이므로 **float64로 저장**한다.

### 2.5 서브샘플과 재라벨 (필수)

저장 라벨은 2048점 cloud의 함수다. 이를 서브샘플과 짝지으면 **회귀가 ill-posed해진다**
— 답이 입력에 들어 있지 않다. v1에서 실측한 크기:

- 저장 라벨(2048점) vs 같은 장면 512점 라벨: AIRM **중앙값 2.22**
- 한 장면에서 512점을 다섯 번 다르게 뽑으면 서로 **1.38–3.30** 차이
- 이것이 정확도의 원리적 하한이 된다 (학습이 4.69에서 고착했고, 재라벨 후 4.16으로
  내려가며 블록 오차가 5.5 → 2.4로 절반이 됐다)

따라서 로더는 `n_points`가 주어지면 **서브샘플에서 라벨을 재계산한다**
(`relabel=True`, 기본값). 저장된 면적을 그대로 쓰므로 재계산본은 **같은 면적분의 같은
추정량**이고, 그래서 해상도를 바꿔도 다른 양이 되지 않는다 (§2.3). 재계산은
`(n_points, seed)`에 대해 결정론적이라 `<root>/cache/n<N>_seed<S>/`에 캐시된다
(1024개 24초 → 재사용 0.1초). `relabel=False`는 ill-posed 짝을 의도적으로 재현할
때만 쓴다.

**해상도 선택.** 균일 표면 샘플의 균일 부분집합은 그 자체로 균일 표면 샘플이라
downsample은 분포를 왜곡하지 않는다. 라벨은 어느 $N$에서도 같은 물리량의 불편추정량이고
(§2.3), 남는 것은 MC 분산과 학습 비용의 절충이다.

| $N$ | MC 분산 (AIRM) | 1 epoch (train 102,400) | 150 epoch |
|---|---|---|---|
| 2048 | 기준 | 93분 | 233시간 |
| **1024** | **0.51** | **24분** | **59시간** |
| 512 | 1.02 | 7.6분 | 19시간 |
| 256 | 1.86 | 3.3분 | 8시간 |

$O(N^2)$ 거리행렬이 비용을 지배한다. **권장 $N_0=1024$.**

**upsample은 권하지 않는다.** 보간으로 만든 점은 실제 표면 위에 없으므로 접촉
라벨을 왜곡한다. 더 조밀한 데이터가 필요하면 생성기의 `--n-points`를 올린다.

---

## 3. 저장 포맷과 재현성

```
data/peg_hole/v2/
  meta.json                # cfg 전체, 시드, 샤드 목록 (샤드마다 원자적 갱신)
  {split}_{idx:05d}.npz    # 샤드당 2048 샘플
    points     [S,2048,3] f32   part [S,2048] u8    K_contact [S,6,6] f64
    K_body     [S,6,6]    f64   area_peg [S] f64    area_plate [S] f64
    stage/profile [S] u8        n_peg [S] i32
    r_peg/H/T/clearance/depth/tilt/yaw_mismatch/noise [S] f32
```

`meta.json`의 `version`이 라벨 정의를 식별한다 — **v2 = 면적 가중**, v1 = 구
$1/(N_{\rm peg}k)$ 정규화. `area_*`가 없는 샤드는 v1이며 재라벨이 불가능하다.

- 샤드 시드 = `master·1000003 + split_offset·65536 + idx` — 순서 무관·재개 가능
  (`meta.json`에 기록된 샤드는 건너뜀), 같은 시드 → 동일 바이트 (테스트).
- 생성 순서 val → test → train: 부분 생성 중에도 로더가 즉시 사용 가능.
- 처리량 (RTX 4090, N=2048): **~27 샘플/s**, 전체 122,880쌍에 **약 75분**, 3.2 GB.
  v1의 57/s보다 느린 것은 반경 안 쌍을 전부 세기 때문이며, 그것이 라벨을 수렴하게
  만드는 바로 그 성질이다.

## 4. 사용법

```bash
# 1) 생성 — 약 75분, 3.2 GB.  중단해도 --out 을 그대로 주면 이어서 진행한다
#    (meta.json 에 기록된 샤드는 건너뛴다).  val -> test -> train 순이라
#    train 이 도는 동안에도 val 로 smoke 를 돌릴 수 있다.
python data_gen/gen_peg_hole_pcd.py --out data/peg_hole/v2 --device cuda:0

#    먼저 통계만 보고 싶을 때 (아무것도 쓰지 않음)
python data_gen/gen_peg_hole_pcd.py --inspect --inspect-n 96 --device cuda:0
#    규모/해상도 조절
python data_gen/gen_peg_hole_pcd.py --out data/peg_hole/v2 --device cuda:0 \
    --n-train 102400 --n-val 10240 --n-test 10240 --n-points 2048

# 로더 (n_points를 주면 그 점들에서 라벨을 재계산 + 캐시)
from data_loader.peg_hole_data_loader import load_peg_hole_split
P, K = load_peg_hole_split('data/peg_hole/v2', 'train', n=4096,
                           n_points=1024, seed=0, device='cuda')

# 학습 (pointwise 파이프라인, 저장 라벨) — 그래프 튜닝 플래그 불필요
python experiment/pc_se3_congruence/blockage_bench.py --dataset peghole \
    --encoder pointwise --method covector --target-graph stored \
    --peghole-n-points 1024 --recipe full
# realizability 통제군 (타깃이 모델 클래스 안에 있음이 보장)
#   ... --target-graph teacher
```

**그래프 반경 (이 데이터셋이 드러낸 문제와 그 수정).** 최초 smoke에서 기본값
(`global_scale` $\alpha=0.75$)이 truncation 0.99로 set-equivariance 보장을 깼다.
원인은 표면 분포에 국한되지 않았다 — `global_scale`은 degree가 $N$에 따라 발산하고
(iid **부피** N=512에서도 trunc 0.71), `density_scaled`의 $(k/N)^{1/3}$은 내재차원 3
가정이라 표면에서 degree가 16.5→24.3으로 드리프트하고 곡선에서는 59.9까지 간다.

이에 따라 `pointwise_graph.py`에 **`degree_matched`(신규 기본값)**를 추가했다:
평균 degree가 `target_k`가 되는 반경은 전체 쌍거리 중 $N\cdot k_{\rm target}$번째로
작은 값과 정확히 같으므로, 밀도 모형도 차원 지수도 없이 닫힌 형태로 구해진다.
설계 근거와 전체 실측은 `graph_radius.md`. 결과
(전 분포 `target_k=16`, `candidate_k=64`):

| cloud | degree | 필요 예산 | trunc |
|---|---|---|---|
| iid 부피 N=32/512/1024 | 15.9 / 16.0 / 16.0 | 26 / 44 / 57 | 0.000 |
| peg-hole 표면 N=512/2048 | 16.0 / 16.0 | 41 / 46 | 0.000 |
| 곡선 N=256 · 래티스 $5^3$ | 16.0 / 15.4 | 22 / 25 | 0.000 |
| centro · c2 · tetra | 16.0 / 16.0 / 15.9 | 38 / 37 / 34 | 0.000 |

따라서 **peg-hole 학습에 그래프 플래그를 줄 필요가 없다**. 순수 기본값 6 epoch
smoke에서 loss 4.97→4.84 단조 감소, rank 6, degree 16.0, trunc 0.000,
등변성 2.6e-15다.

`--peghole-n-points`(기본: 저장 해상도 2048 그대로)로 서브샘플. `--lambda-body`로
합성 가중 오버라이드. 시각화: `python experiment/pc_se3_congruence/visualize_peg_hole.py`
→ `figs/peghole_scenes.png`, `figs/peghole_labels.png`.

## 5. 검증 (test/test_peg_hole_dataset.py, 14종 전부 통과)

프로파일 기하(면적/경계/margin) · 장면 구성(part 비율, dtype) · **무관통**(전
stage 점 단위) · stage별 gap 일관성 · **라벨 등변성**(비영 병진 SE(3), <5e-9) ·
순열 불변 · PSD/SPD·cond · free ⇒ contact 정확히 0 / insert ⇒ $O(1)$ ·
생성 재현성 · **저장-재계산 비트 일치** · 재개가 기존 샤드를 건드리지 않음 ·
로더 서브샘플 결정론 · **재라벨이 입력의 함수임**(재계산과 $10^{-14}$ 일치) ·
재라벨 캐시 일관성. 전체 스위트 63종 무회귀.

## 6. 한계와 확장 방향

- **가시성:** 전 표면 샘플링(CAD-스캔형). 단일 시점 depth 카메라의 occlusion은
  미구현 — 시점 기반 hidden-point removal을 생성 옵션으로 추가 가능.
- **법선 미저장** (points에서 국소 PCA로 복원 가능; 디스크 여유 51GB 고려).
- **접촉 모델:** 근접-기반 proximity springs이지 물리 시뮬레이션(마찰·변형)이
  아니다. 실측 강성으로 가는 중간 단계의 "정답 함수" 위치는 선행 실험과 동일.
- **정확한 tie:** 노이즈 때문에 정확한 등거리 tie는 없다 — tie 극한 검증은
  기존 `lattice` 벤치가 계속 담당.
- **MC 분산:** 라벨은 불편추정량이지만 $N$=1024에서 표준편차가 AIRM ~0.5다. 이보다
  정밀한 "물리 정답"이 필요하면 $N$을 올리거나 여러 뽑기를 평균해야 한다.
- **모델 클래스 불일치(미해결):** 타깃은 raw wrench의 edge 단위 2차 모멘트, pointwise
  모델은 latent covector의 point 단위 2차 모멘트다. encoder가 shell **평균**(1차
  모멘트)만 넘기므로 이웃 방향의 퍼짐이 17–66% 소실된다. 단, edge 2차 모멘트 항을
  모델에 직접 넣는 것은 생성 공식을 역공학하는 것이라 PoC로서 부적절하다.
- **부품 구분 정보:** 타깃은 cross-part 쌍만 쓰는데 모델이 보는 kNN edge의 93.6%가
  같은 부품끼리이고, `part` 마스크는 모델 입력에 넣지 않고 있다.
- 다음 버전 후보: 시점 occlusion, 다중 구멍/다중 부품, nut-bolt·shaft-bearing 등
  형상 확장, 접촉 여부를 넘어선 힘 균형 기반 라벨.

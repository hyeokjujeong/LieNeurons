# `pc_se3_congruence` — 실행 가이드

Point cloud $P$에서 $6\times6$ 강성 $K$를 예측하되, 좌표계를 바꾸면 $K$가 **congruence**로
따라 변해야 한다는 제약을 학습이 아니라 **구조**로 만족시킨다.

$$K(T\cdot P)=\mathrm{Ad}_T^{-\top}\,K(P)\,\mathrm{Ad}_T^{-1},\qquad T=(R,p)\in SE(3)$$

이 문서는 **어떻게 돌리는지와 기본값이 무엇인지만** 다룬다. 설계 근거·증명·실험 결과는
아래 [상세 문서](#상세-문서)에 있다.

[빠른 시작](#빠른-시작) → [학습하기](#학습하기) → [결과 읽는 법](#결과-읽는-법) 순으로
읽으면 된다. 데이터를 아직 형식에 맞추지 않았으면 [데이터 준비](#데이터-준비)를 먼저
보고, 이 저장소의 합성 데이터셋을 쓰려면 [peg-and-hole 데이터셋](#peg-and-hole-데이터셋)으로.

**표기 — angular/moment first.** 저장소 전체가 twist $\xi=[\omega;v]$, wrench
$F=[m;f]$를 쓴다 (`core.lie_alg_util.HatLayer('se3')`부터 라벨까지 동일). 따라서

$$\mathrm{Ad}_T=\begin{bmatrix}R&0\\\hat pR&R\end{bmatrix},\qquad
\mathrm{Ad}_T^{-\top}=\begin{bmatrix}R&\hat pR\\0&R\end{bmatrix},\qquad
K=\begin{bmatrix}K_{mm}&K_{mf}\\K_{fm}&K_{ff}\end{bmatrix}$$

이고 `K[0:3,0:3]`가 **회전** 강성, `K[3:6,3:6]`가 **병진** 강성이다.

---

## 빠른 시작

모든 명령은 **저장소 루트**에서 실행한다 (`core/`, `data_loader/` 등을 루트 기준으로
import한다).

```bash
conda env create -f environment.yml
conda activate lieneurons

# 설치 확인 — 데이터셋 없이 1분 안에 끝난다
python experiment/pc_se3_congruence/train.py selftest
```

`selftest`는 합성 클라우드로 5 epoch을 돌린다. 학습 후 equivariance가 `1e-15` 수준이고
`rank 6`, `clamp 0`이면 설치와 경로가 정상이다 (정확도는 보지 않는다).

---

## 학습하기

데이터가 이미 있다고 가정한다. 필요한 건 `(point cloud, 6×6 강성)` 쌍이 담긴 파일
하나고, 경로는 `--data-path` 하나로 준다. 아직 없으면 [데이터 준비](#데이터-준비)부터.

```bash
python experiment/pc_se3_congruence/train.py --data-path mydata.npz
```

이게 전부다. `--data-path`를 주면 **내 데이터셋용 설정(`custom`)으로 알아서 붙는다** —
표본은 파일에 있는 만큼 전부, 60 epoch, 저장된 해상도 그대로. 실험 이름을 따로 칠 필요
없다. **학습이 실제로 돌고 있는지는 loss가 아니라
[매 epoch 로그의 진단 열](#결과-읽는-법)로 확인한다** — 이 구조에서는 loss가 정상적으로
내려가면서 모델이 입력을 전혀 안 읽는 상태가 실제로 발생한다.

돌리기 전에 확인만:

```bash
python experiment/pc_se3_congruence/train.py --data-path mydata.npz --dry-run
python experiment/pc_se3_congruence/train.py --list      # 정의된 실험 목록
```

`--dry-run`은 실제로 실행될 `blockage_bench.py` 명령을 그대로 출력하고 끝낸다.

값을 바꾸고 싶으면 그냥 뒤에 붙인다. 실험이 정한 기본값은 **내가 직접 준 플래그를 절대
덮지 않는다.**

```bash
python experiment/pc_se3_congruence/train.py --data-path mydata.npz \
    --epochs 200 --batch 16 --device cuda:1
```

이름을 주고 싶으면 **반드시 첫 인자**여야 한다 (`train.py cmpA --epochs 60`).
`--data-path`도 이름도 없으면 기본 실험 `cmp2`가
[peg-and-hole 데이터셋](#peg-and-hole-데이터셋) 경로로 실행된다.

---

## 기본 설정 `cmp2`

`cmp2`와 `custom`은 **같은 모델 설정**이고 데이터 경로와 예산만 다르다. 아래가 그 설정이다.

`--pw-force-invariant`가 켜져 있다. gate와 $\beta$에 공급되던 Klein 불변량이 **항등적으로
0**이 되는 경우가 있다 — 인코더가 이웃 wrench를 모두 같은 앵커 점에 걸면 잠재 특징이
전부 pitch 0인 wrench가 되는데, Klein form은 정확히 pitch를 재기 때문이다. 그때
파라미터의 70%가 상수만 출력한다. pitch 0에서 죽지 않는 $f_a\cdot f_b$로 바꾼 것이 이
플래그이고, 대조 실험에서 $d$ 2.888 → **1.634**였다. 자기 데이터에서 이 경로가 살아
있는지는 로그의 [`inv` 열](#결과-읽는-법)로 본다.

| 항목 | 기본값 | 왜 이 값인가 |
|---|---|---|
| `--data-path` | `data/peg_hole/v2` | 경로 하나로 통일. `custom`에서는 **필수 인자**다 |
| `--target-graph` | `stored` | 파일에 저장된 라벨을 그대로 타깃으로 쓴다 |
| `--n-points` | 1024 | 학습 비용이 $O(N^2)$이라 절충점. **서브샘플은 라벨을 재계산할 수 있는 형식에서만 허용**되므로 ([데이터 준비 §5](#5-저장-전-확인)) `custom`에서는 저장된 해상도를 그대로 쓴다 |
| `--n-train` / `--n-val` | 20480 / 2048 | `custom`은 `0 0` — 파일에 있는 만큼 전부 |
| `--epochs` | 150 | `custom`은 60 |
| `--batch` | 32 | |
| `--eval-batch` | 64 | 그래프가 $[B,N,N]$ 거리행렬을 만들어 **최대 GPU 메모리를 이 값이 지배**한다 ($N$=1024, 64 → 537 MB). 숫자 자체는 불변 |
| `--lr` | 1e-3 | |
| `--pw-channels` | 16 32 64 32 | $[C_0(\text{set pooling}), C_1, \dots]$ |
| `--pw-factors` | 16 | factor 채널 $H$ |
| `--seed` | 100 | |
| `--device` | `cuda:0` | torch는 기본이 FASTEST_FIRST라 `nvidia-smi` 인덱스와 다를 수 있다 |
| 인코더/표현 | `pointwise` / `covector` | 고정. 이 파이프라인은 전부 $\mathfrak{se}(3)^*$ 안에서 동작한다 |
| 그래프 반경 | `degree_matched`, `candidate_k` 64 | 평균 degree를 `target_k`로 고정하는 닫힌 형태. 이 데이터에서 degree ~16, truncation 0 |

---

## 다른 실험

| 이름 | 설명 | 추가 플래그 |
|---|---|---|
| **`cmp2`** | 기본 | `--pw-force-invariant` |
| `cmpA` | 대조군 — 불변량 교체만 뺀 초기 구성 | 없음 |
| `cmpC` | 제거군 — gate와 $\beta$를 통째로 제거 (파라미터 51%↓) | `--pw-gate none --pw-beta uniform` |
| `teacher` | realizability 통제군 — 타깃이 같은 클래스의 고정 난수 모델 | `--pw-force-invariant` |
| `custom` | **내 데이터셋** — `--data-path`만 주면 이름 없이 여기로 붙는다 | `--pw-force-invariant` |
| `selftest` | 설치 확인 스모크 — 데이터 불필요, 1분 | `--pw-force-invariant` |

`teacher`를 먼저 보라. 정답 가중치가 존재함이 보장되므로 여기서 0 근처로 내려가지
않으면 최적화 문제이고, **그러면 `stored` 결과는 해석할 수 없다.**

`cmpC`는 반증 시험이다. `cmp2`의 진단이 맞다면 죽은 gate 출력은 채널별 상수이고 뒤의
LNLinear가 흡수하므로 `cmpC`는 `cmpA`와 거의 같아야 한다. 달랐다면 진단이 틀린 것이다.

`custom`과 `selftest`를 뺀 나머지는 데이터 경로 기본값이 peg-and-hole이다. 자기
데이터로 같은 대조를 돌리려면 `--data-path`를 같이 준다.

### 래퍼에 없는 옵션

`train.py`가 모르는 플래그는 `blockage_bench.py`로 그대로 전달된다. 래퍼가 노출하지
않는 옵션도 기다릴 필요 없이 바로 쓸 수 있고, 오타는 `blockage_bench.py`가 거부한다.

```bash
python experiment/pc_se3_congruence/train.py cmp2 --pw-gate full
```

### 실험 추가

`train.py`의 `EXPERIMENTS` 딕셔너리에 항목 하나를 넣으면 끝이다. 생략한 값은 공용
기본값으로 채워진다.

---

## 결과 읽는 법

매 epoch 한 줄이 이렇게 찍힌다.

```
ep    2  train d 2.543  val d 1.968  scale 1.465  shape 1.135  mm 1.164  ff 0.963
         rank 6.0  clamp 0  |g| 3.89e+00  βσ 7.32e-05  inv 1.49e-02  |f_c| 0.8374
         deg 16.0  trunc 0.000
```

### 먼저 볼 것 — 학습이 실제로 돌고 있는가

loss가 정상적으로 내려가면서 **모델이 입력에서 아무것도 읽지 않는** 상태가 이 구조에서는
실제로 발생한다. 이 프로젝트의 핵심 발견이 그거였다 — Klein 불변량이 항등적으로 0이라
파라미터의 70%가 상수만 출력했는데 등변성·rank·truncation·clamp·loss 곡선이 **전부
정상**이었다. 그래서 아래 열들이 학습 루프에 상시 계측으로 들어가 있다.

| 열 | 살아 있으면 | 죽었으면 |
|---|---|---|
| `inv` (`head_inv_abs`) | $10^{-2}$ 수준 | $10^{-17}$ — 불변 스칼라 경로가 상수만 낸다 |
| `βσ` (`beta_scene_std`) | $>0$ | `0.00e+00` — 장면별 크기 변조가 없다 |
| `\|f_c\|` (`f_signal`) | $O(1)$ | 0 — 인코더가 방향 정보를 못 뽑는다 |

구조가 유지되는지도 함께 본다: `rank` 6, `clamp` 0, `trunc` 0.000.

### 그 다음 — 무엇을 못 맞추는가

- `scale` / `shape` — 강성의 **크기**를 틀렸는지 **방향 구조**를 틀렸는지. 모델의 전역
  handle은 $e^g$ 하나뿐이라 `scale`이 크면 "장면별 크기를 못 읽는다"는 뜻이다.
- `mm` / `ff` — 회전 강성(`K[0:3,0:3]`) / 병진 강성(`K[3:6,3:6]`) 블록의 상대 오차.
  **어느 물리 응답이 문제인지 국소화하는 진단용이지 판정용이 아니다** (타깃 블록 norm이
  작은 곳에서 분모가 무너져 폭발한다).
- 거리는 $\exp(d/\sqrt 6)$로 읽는다 — $d$가 6개 log-eigenvalue의 Frobenius norm이므로
  방향당 몇 배 어긋났는지가 나온다.

전체 지표 목록은 `peghole_training_report.md` 부록 D.

### wandb

기본이 `--wandb-mode online`이고, entity/project는 `metrics.py`에
`adjoint_equivariant_network/pc-se3-congruence`로 **하드코딩**되어 있다. 다른 계정으로
돌린다면 넘겨서 덮는다 (`train.py`가 모르는 플래그는 그대로 전달된다):

```bash
python .../train.py --wandb-entity my-team --wandb-project my-project
python .../train.py --wandb-mode disabled              # 기록 없이
```

초기화에 실패하면(미로그인·권한 없음) `[wandb 비활성화: ...]`만 찍히고 **학습은
계속되지만 지표는 남지 않는다.** 남의 계정에서 처음 돌릴 때는 시작 직후 이 줄이
없는지 확인할 것.

---

## 데이터 준비

자기 **(point cloud, 6×6 강성)** 쌍을 이 형식으로 맞추면 된다. 데이터셋 경로는
`--data-path` **하나뿐**이고, 형식은 경로를 보고 판별한다 (`meta.json`이 있으면
[peg-and-hole 샤드 데이터셋](#peg-and-hole-데이터셋), 아니면 아래 형식).

### 1. 데이터 구조

배열 **두 개**. 그 외에는 아무것도 필요 없다.

```
points   float  [S, N, 3]     S개 장면 × N개 점 × (x, y, z)
K        float  [S, 6, 6]     각 장면의 6×6 강성 (대칭 SPD)
```

- **`points[i]` 와 `K[i]` 가 같은 장면.** 인덱스 대응이 유일한 연결 고리다.
  어떤 파일 형식을 쓰든 **장면 순서가 두 배열에서 같아야** 한다.
- **`N`은 모든 장면에서 같아야 한다** (한 배치로 쌓는다). 점 개수가 제각각이면
  잘라내거나 패딩하지 말고 — 라벨이 그 점들의 함수다 — 같은 `N`으로 다시 만든다.
- `float32`로 저장해도 되고 내부에서 `float64`로 올린다.
- **장면 하나짜리 메타데이터(재질, 포즈, 시각 …)는 넣어도 무시된다.** 필수는 위
  둘뿐이다.

이건 **논리적 구조**다. 이걸 어느 컨테이너에 어떻게 담는지는 §2.

split은 둘 중 하나로 준다.

```
mydata.npz                    ← 파일 하나. --val-frac (기본 0.1) 로 나눈다
mydata/                       ← 디렉터리. 미리 나눠 둔 경우
├── train.npz
└── val.npz
```

파일 전체를 한 번에 메모리에 올린다. 필요한 RAM은 `S × N × 3 × 8` 바이트
(20480장면 × 1024점이면 503 MB) + `S × 288` 바이트.

### 2. 파일 형식

**로더가 직접 읽는 것은 `.npz`와 `.pt` 둘뿐이다.** 나머지는 아래 스니펫으로 한 번
변환한다 — 배열 두 개짜리라 어느 형식이든 3~4줄이다.

| 형식 | `--data-path`에 바로 | 비고 |
|---|:---:|---|
| `.npz` | **O** | 권장. numpy만 있으면 된다 |
| `.pt` | **O** | dict 또는 `(points, K)` 튜플 |
| `.safetensors` | 변환 | 이름→텐서 dict라 대응이 그대로 |
| `.h5` / `.hdf5` | 변환 | dataset 두 개 |
| `.parquet` | 변환 | 행 기반이라 레이아웃 선택이 필요 |
| `.json` / `.jsonl` | 변환 | 중첩 리스트 |
| `.csv` | 변환 | 3차원 배열이 안 들어간다. 비권장 |

> 변환에 쓰는 `pandas` / `pyarrow` / `h5py` / `safetensors`는 `environment.yml`에
> **없다.** 변환할 때만 있으면 되고 학습에는 필요 없다.

#### `.npz` — 이 저장소의 기본형

```
mydata.npz
├── points   [S, N, 3]
└── K        [S, 6, 6]
```

```python
np.savez('mydata.npz', points=points, K=K)              # 또는 savez_compressed
```

**키 이름은 별칭을 받는다**: `points` / `P` / `cloud` / `xyz`,
`K` / `stiffness` / `K_gt`. 기존 파일이 이 중 하나를 쓰고 있으면 손대지 않아도 된다.
못 찾으면 파일에 실제로 있는 키 이름을 같이 출력하고 멈춘다.

#### `.pt` — torch

```python
torch.save({'points': torch.from_numpy(points), 'K': torch.from_numpy(K)}, 'mydata.pt')
torch.save((torch.from_numpy(points), torch.from_numpy(K)), 'mydata.pt')   # 튜플도 됨
```

dict면 위의 키 별칭이 그대로 적용되고, 길이 2짜리 tuple/list면 `(points, K)` 순서로 읽는다.

#### `.safetensors`

이름→텐서의 평평한 dict다. 논리 구조가 그대로 들어간다.

```
mydata.safetensors
├── "points"   F64  [S, N, 3]
└── "K"        F64  [S, 6, 6]
```

```python
from safetensors.numpy import load_file
d = load_file('mydata.safetensors')
np.savez('mydata.npz', points=d['points'], K=d['K'])
```

torch 쪽(`safetensors.torch`)으로 **쓸** 때는 텐서가 contiguous여야 저장된다
(`.contiguous()`). 읽을 때는 상관없다.

#### `.h5` / `.hdf5`

루트에 dataset 두 개. 그룹 안에 넣었으면 경로만 바꾸면 된다.

```
mydata.h5
├── /points   [S, N, 3]
└── /K        [S, 6, 6]
```

```python
import h5py
with h5py.File('mydata.h5', 'r') as f:
    np.savez('mydata.npz', points=f['points'][:], K=f['K'][:])   # [:] 로 실제 읽기
```

#### `.parquet`

행 기반이라 `[S, N, 3]`이 그대로 안 들어간다. 두 레이아웃 중 하나다.

**(a) 장면당 한 행, list 컬럼** — 장면 경계가 행으로 보존돼서 안전하다. 배열을
평평하게 편다 (`points` 길이 `3N`, `K` 길이 36, **행 우선**).

```
row i │ points = [x₀,y₀,z₀, x₁,y₁,z₁, …]   (길이 3N)
      │ K      = [K₀₀,K₀₁,…,K₀₅, K₁₀,…]    (길이 36)
```

```python
df = pd.read_parquet('mydata.parquet')
points = np.stack(df['points'].to_numpy()).reshape(len(df), -1, 3)
K      = np.stack(df['K'].to_numpy()).reshape(len(df), 6, 6)
np.savez('mydata.npz', points=points, K=K)
```

**(b) 점당 한 행 (long)** — 스캐너 출력이 보통 이 모양이다. 장면 id 컬럼이 필요하고,
`K`는 장면당 한 행이라 테이블이 따로 있다.

```
points.parquet │ scene  x  y  z        ← S·N 행
K.parquet      │ scene  K00 K01 … K55  ← S 행
```

```python
p = pd.read_parquet('points.parquet').sort_values('scene', kind='stable')
k = pd.read_parquet('K.parquet').sort_values('scene', kind='stable')
points = p[['x', 'y', 'z']].to_numpy().reshape(-1, N, 3)
K = k[[f'K{i}{j}' for i in range(6) for j in range(6)]].to_numpy().reshape(-1, 6, 6)
np.savez('mydata.npz', points=points, K=K)
```

> long 형식은 **정렬이 전부다.** `scene`으로 stable sort 하지 않으면 점이 엉뚱한
> 장면으로 섞이는데, 형상은 멀쩡한 `[S, N, 3]`이라 **에러가 안 난다.** 그리고
> 장면마다 행 수가 정확히 `N`이어야 `reshape`이 맞다 — 다르면 여기서 터진다
> (다행히 터진다). `df.groupby('scene').size().nunique() == 1`로 미리 확인.

#### `.json` / `.jsonl`

중첩 리스트. 두 레이아웃 다 흔하다.

```json
{"points": [[[x, y, z], …N개], …S개],
 "K":      [[[k, k, k, k, k, k], …6개], …S개]}
```

```python
d = json.load(open('mydata.json'))
np.savez('mydata.npz', points=np.array(d['points']), K=np.array(d['K']))
```

장면 레코드 리스트(`[{"points": …, "K": …}, …]`)나 jsonl(한 줄 = 한 장면)이면:

```python
recs = [json.loads(l) for l in open('mydata.jsonl')]      # json.load(...) for .json
np.savez('mydata.npz',
         points=np.array([r['points'] for r in recs]),
         K=np.array([r['K'] for r in recs]))
```

#### `.csv`

3차원 배열이 안 들어가므로 long 형식밖에 없다 — parquet (b)와 같은 스키마를 읽으면
된다. 텍스트라 **npz의 2.6배 크기에 58배 느리다.** 가진 게 csv뿐이면 한 번 변환하고
npz로 다시 쓰라.

읽을 때 `float_precision='round_trip'`을 주라. pandas의 기본 파서는 마지막 자리가
1 ulp 틀리고(실측 3.6e-15), 이 플래그를 주면 정확히 0이 된다. 쓰는 쪽은 기본값으로
충분하다 — 어긋나는 건 **파서**지 `float_format`이 아니다.

```python
pd.read_csv('points.csv', float_precision='round_trip')
```

`K`에서는 이 1 ulp가 실제로 문제가 된다: 대칭성이 깨지고, 고윳값이 0 근처인 표본은
SPD 판정이 뒤집힐 수 있다. §5의 대칭화를 반드시 거치라.

#### 무엇을 고를까

`S=200`, `N=1024` (float64 5.0 MB)에서 실측한 파일 크기와 로드 시간:

| 형식 | 크기 | 로드 | npz 대비 |
|---|---|---|---|
| `.npz` | 5.0 MB | 0.7 ms | 1.00× |
| `.npz` float32 | 2.5 MB | 0.4 ms | 0.53× |
| `.pt` | 5.0 MB | 0.4 ms | 0.55× |
| `.safetensors` | 5.0 MB | 0.5 ms | 0.82× |
| `.h5` | 5.0 MB | 0.4 ms | 0.57× |
| `.parquet` | 5.3 MB | 3.1 ms | 4.7× |
| `.json` | 13.2 MB | 131 ms | **197×** |
| `.csv` | 12.8 MB | 39 ms | **58×** |

바이너리 형식끼리는 사실상 차이가 없다. 텍스트 형식만 두 자릿수로 갈린다.
`.npz`를 쓰고, 디스크가 아쉬우면 `float32`로 저장하라 (내부에서 `float64`로 올리고,
라벨 정밀도는 손실 1e-7 수준이라 손실 값 1.6 근처에서는 무해하다).
`savez_compressed`는 점 좌표가 잘 안 눌려서 이득이 거의 없고 로드만 20배 느려진다.

### 3. `K` 6×6의 내부 구조

`K`는 twist를 wrench로 보내는 행렬이다. 행과 열이 무엇인지가 전부다.

```
                    ω_x   ω_y   ω_z  │  v_x   v_y   v_z      ← 열: 입력 twist
                  ┌──────────────────┼──────────────────┐
            m_x   │                  │                  │
   행:      m_y   │       K_mm       │       K_mf       │
   출력     m_z   │    회전 강성       │      결합        │
   wrench         ├──────────────────┼──────────────────┤
            f_x   │                  │                  │
            f_y   │       K_fm       │       K_ff       │
            f_z   │      결합         │    병진 강성      │
                  └──────────────────┴──────────────────┘
```

$$\begin{bmatrix}m\\f\end{bmatrix}
=K\begin{bmatrix}\omega\\v\end{bmatrix}$$

즉 `K[0:3,0:3]`가 **회전** 강성(비틀었을 때 모멘트), `K[3:6,3:6]`가 **병진**
강성(밀었을 때 힘)이다. 대칭이므로 `K_fm = K_mf^T`.

**force-first로 저장했다면** — `K[0:3,0:3]`가 병진인 경우 — 블록을 바꿔야 한다.
바꾸지 않아도 대칭 SPD라서 **아무 에러 없이 돌고 숫자만 틀린다.**

```python
K = K[:, [3,4,5,0,1,2]][:, :, [3,4,5,0,1,2]]
```

### 4. 모멘트를 어느 점 기준으로 쟀는가

wrench $F=[m;f]$에서 **힘 $f$는 기준점과 무관하지만 모멘트 $m=r\times f$는 기준점에
따라 달라진다.** 같은 물리 상황이라도 어느 점에 대해 모멘트를 쟀느냐로 `K`의 숫자가
달라진다는 뜻이다. 실측 — 같은 cloud, 원점 기준 vs 무게중심 기준:

| | 두 `K`의 상대 차이 |
|---|---|
| 전체 | **0.855** — 서로 다른 행렬 |
| `K_ff` (힘 블록) | 9e-17 — 힘은 기준점과 무관하므로 동일 |
| `K_mm` (모멘트 블록) | **0.822** — 모멘트만 달라진다 |

**이 저장소의 규약은 `points`에 적힌 좌표계의 원점이다.** 라벨이
$m_i=r_i\times f_i$로 만들어지고 $r_i$가 저장된 좌표 그대로여야 한다. 그래야 장면을
움직였을 때 `K`가 congruence로 따라 변한다 (실측 잔차 1.7e-15). 무게중심 기준으로 잰
`K`는 평행이동에 **불변**이 되어 버려서, 모델이 구조로 갖고 있는 법칙과 어긋난다.

로보틱스에서는 무게중심이나 TCP 기준으로 재는 경우가 흔하다. 그렇게 잰 `K_c`(기준점
$c$)를 원점 기준으로 옮기려면:

```python
# c: 기준점의 좌표 [3].  A = Ad^{-T}(I, c),  [m; f] 순서
cx = np.array([[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]])
A  = np.block([[np.eye(3), cx], [np.zeros((3, 3)), np.eye(3)]])
K_origin = A @ K_c @ A.T
```

(검증: 이 식이 원점 기준 라벨을 2.8e-17로 재현한다. 대칭과 SPD는 congruence라 보존된다.)

> **물체를 원점 근처에 두라.** 모멘트가 $|r|$에 비례해 커지므로 물체가 원점에서 멀면
> 회전 블록이 병진 블록을 덮어 버린다. 같은 cloud를 $x$축으로 옮기며 잰 값:
>
> | ‖중심‖ | 0 | 1 | 10 | 100 |
> |---|---|---|---|---|
> | cond(`K`) | 9.8 | 2.1e1 | 1.3e4 | **1.2e8** |
> | ‖`K_mm`‖/‖`K_ff`‖ | 3.8 | 4.9 | 88 | **8238** |

### 5. 저장 전 확인

```python
K = 0.5 * (K + K.transpose(0, 2, 1))                    # 대칭화
assert np.linalg.eigvalsh(K)[:, 0].min() > 0            # SPD
assert points.shape[0] == K.shape[0]                    # 장면 수 일치
np.savez('mydata.npz', points=points, K=K)
```

SPD를 여기서 확인하는 이유: 손실이 타깃의 Cholesky 인수를 쓰기 때문에 **표본 하나만
어긋나도 실행이 죽는다.** rank가 모자라면 물리 모델에 body 항을 더해 정칙화한다
(peg-and-hole 데이터셋이 $K=K_{\rm contact}+\lambda K_{\rm body}$로 하는 것과 같다).

**해상도.** `--n-points` 서브샘플은 이 형식에서 거부된다. 부분집합에 원래 라벨을
붙이면 회귀가 ill-posed해지기 때문이다 (입력이 답을 결정하지 않는다). 점을 줄이려면
cloud를 줄이고 `K`를 **다시 계산해서** 저장한다.

이제 [학습하기](#학습하기)로 간다.

---

## 선택: 절대 눈금이 필요할 때

AIRM 거리에는 절대 눈금이 없다. 실행끼리 비교하거나 학습이 도는지 보는 데는
[결과 읽는 법](#결과-읽는-법)의 지표로 충분하지만, **새 데이터셋에서 "이 숫자가 좋은
건가"를 판정**해야 한다면 대조군이 필요하다.

```bash
python experiment/pc_se3_congruence/peghole_baseline.py \
    --data-path mydata.npz --n-train 0 --n-val 0 --out baseline.json

python experiment/pc_se3_congruence/train.py \
    --data-path mydata.npz --baseline-json baseline.json
```

`--baseline-json`을 주면 매 epoch `rel`(기준선 대비 비율)이 붙고 끝에
`최저 val d ... (기준선 대비 N)`이 나온다.

**기준선은 실행마다가 아니라 데이터셋 설정당 한 번이면 된다** — `(데이터셋, split 크기,
N, λ_body)`만의 함수이고 모델·seed·하이퍼파라미터와 무관하다. 설정을 바꾸면 다시 뽑아야
한다 (json 재사용은 이걸 검사하지 않는다).

### 판정이 자명하지 않은 이유

**이 문제에는 자명한 예측자가 둘이고 서로 정반대로 잘한다.** 상수 예측자는 강성의
*크기*를 맞추고 방향을 못 맞추며, 학습 안 된 랜덤 초기화 모델은 그 반대다 — 등변 구성
자체가 방향 이방성을 공짜로 준다. 따라서 "기준선을 이겼다"만으로는 판정이 안 되고,
**두 예측자를 각자의 장기에서 동시에 이기는지**를 봐야 한다.

아래는 peg-and-hole 데이터셋(N=1024)의 실측값이다. 다른 데이터셋에서는 값이 달라지므로
그대로 쓰면 안 되고, 같은 방식으로 다시 재야 한다.

| 예측자 | $d$ | $d_{\rm scale}$ | $d_{\rm shape}$ |
|---|---|---|---|
| 학습 안 된 모델 | 7.380 | 6.973 ✗ | 2.361 ✓ |
| Fréchet 평균 (최선의 상수) | 5.174 | **1.677** ✓ | 4.713 ✗ |
| 학습 안 된 모델 + 전역 스칼라 1개 | 3.208 | 1.798 | 2.361 |

| 층 | 조건 |
|---|---|
| 필요조건 | $d<5.174$ |
| **주 기준** | $d_{\rm scale}<1.677$ **AND** $d_{\rm shape}<2.361$ |
| 순 기여 | $d<3.208$ |
| 의심 구간 | $d<0.53$ — 물리가 아니라 뽑기를 외운 것 |

> 주의: `peghole_baseline.py`가 내는 것은 $d$(=5.174)뿐이고, **주 기준의 임계값
> $d_{\rm scale}$/$d_{\rm shape}$는 자동 계산되지 않는다.** 위 표는 일회성 측정치다.

MC 라벨 잡음 **하한**(같은 장면의 다른 서브샘플끼리의 거리 — 이 아래로 내려가면 물리가
아니라 그 뽑기를 외운 것)은 라벨을 재계산할 수 있는 형식, 즉 peg-and-hole에서만 나온다.

---

## 구조 검증과 테스트

학습 숫자를 읽기 전에 구조부터 확인한다. 학습이 필요 없다.

```bash
python experiment/pc_se3_congruence/verify_pointwise.py          # A–F
python experiment/pc_se3_congruence/verify_pointwise.py --full   # 상세
python -m pytest test/ -q                                        # 63 tests
```

`verify_pointwise.py`가 보는 것: (A) 등변성, (B) 점 순열 불변성, (C) 정확한 거리 tie
견고성 — rank-channel 비교군이 여기서 깨진다, (D) 대칭 cloud에서 rank 보존,
(E) 그래프 건전성, (F) near-tie 곡선.

---

## 파일 지도

각 파일 docstring 첫 줄에 `[CURRENT ...]` / `[COMPARISON ARMS]` / `[SUPERSEDED ...]`
태그가 붙어 있다.

### 현행

| 파일 | 역할 |
|---|---|
| `train.py` | **학습 진입점.** 실험 레지스트리 → `blockage_bench.py` 호출 |
| `peghole_baseline.py` | **선택.** 기준선(Fréchet 평균)과 라벨 MC 잡음 — 절대 눈금이 필요할 때 데이터셋 설정당 한 번 |
| `verify_pointwise.py` | 구조 검증 A–F |
| `blockage_bench.py` | 공용 학습 드라이버. 플래그가 많아 `--help`가 그룹으로 나뉘어 있다 |
| `run_experiments.sh` | **전체 프로토콜 하나로 통합.** 단계를 인자로 고른다 — peg-and-hole `baseline teacher stored compare`, 합성 suite `verify synth-teacher synth-analytic`. 보고서 수치의 출처 |
| `run_pointwise_suite.py` | 합성 object suite 실행 (verify / teacher / analytic / ablation) |
| `pointwise_models.py` | set encoder / message passing / Klein gate / late Gram head |
| `pointwise_graph.py` | tie-safe local graph, Wendland window, edge invariant |
| `peg_hole_synth.py` | peg/hole 장면·표면 PCD·contact/body 라벨 생성 |
| `data_synth.py` | 합성 cloud 생성기(centro/c2/tetra/iid/lattice)와 analytic 타깃 |
| `../../data_loader/pc_stiffness_data_loader.py` | **모든 디스크 데이터셋의 단일 로더** — 경로로 형식 판별, 블록 순서·해상도 주의사항 |
| `spd_loss.py` | AIRM 손실 (`affine_invariant_d`가 실제 학습이 최소화하는 것) |
| `se3_utils.py` `metrics.py` | $\mathfrak{se}(3)$ 유틸 / 진단·wandb 헬퍼 |
| `visualize_peg_hole*.py` | 데이터셋 figure |

저장소 루트 쪽: `data_gen/gen_peg_hole_pcd.py`(샤드 생성),
`data_loader/peg_hole_data_loader.py`(서브샘플 재라벨 + 캐시).

### 비교군 (현행 아님, 삭제 불가)

`models.py`, `encoders.py` — 전역 풀링 세대의 모델·인코더. `blockage_bench.py`가
비교 대상으로 계속 인스턴스화한다. 특히 `WrenchSecondMomentModel`은 이웃 **rank**를
채널 인덱스로 읽어 정확한 거리 tie에서 등변성이 깨지는 팔이고, 테스트가 그 실패를
단언한다 (순열 오차 1.9e-01 vs pointwise 3.0e-16).

### `legacy/`

폐기된 진입점. 현행 경로에 없지만, 보고서 표의 출처이거나 현재 설계를 정당화하는
**negative control**이라 남겨 둔다. 세대별 폐기 사유는 `legacy/__init__.py` 참조.

---

## peg-and-hole 데이터셋

이 저장소가 설계와 검증에 쓴 합성 데이터셋이다. 보고서의 수치는 전부 여기서 나왔다.
**자기 데이터로 돌리는 데는 필요 없다.**

**1. 생성.**

```bash
python data_gen/gen_peg_hole_pcd.py --out data/peg_hole/v2 \
    --n-train 102400 --n-val 10240 --n-test 10240 --n-points 2048
```

샤드 단위로 재개 가능하다. 중단해도 `meta.json`에 기록된 샤드는 건너뛰고 이어서 만든다.
생성 전에 캘리브레이션 통계만 보려면 `--inspect`.

**2. 학습.**

```bash
python experiment/pc_se3_congruence/train.py
```

인자 없이 돌리면 기본 실험 **`cmp2`**가 `--data-path data/peg_hole/v2`로 실행된다
([기본 설정](#기본-설정-cmp2)).

**3. 전체 프로토콜.**

기준선 → realizability → 본학습 → 대조를 순서대로 도는 5–6시간짜리(GPU 2장) 실행은
별도 스크립트다. `peghole_training_report.md`의 수치는 여기서 나왔다.

```bash
nohup bash experiment/pc_se3_congruence/run_experiments.sh > peghole.log 2>&1 &
```

설계·캘리브레이션·검증은 `peg_hole_dataset.md`.

---

## 상세 문서

| 문서 | 내용 |
|---|---|
| `peghole_training_report.md` | peg-and-hole 학습 실험 — 판정 기준, 대조 실험, 진단 과정 |
| `peg_hole_dataset.md` | 데이터셋 설계·캘리브레이션·검증 |
| `pointwise_pipeline.md` | pointwise 파이프라인 설계와 구조 검증 |
| `tensor_kernel_experiment_report.md` | compact kernel + second moment (선행 세대) |
| `pc_se3_congruence_report.md` | O/L/A/B/C 구조 검증 상세 (선행 세대) |
| `covector_ln_framework.md` | covector-native 설계와 물리적 해석 |
| `train_report.md` | 초기 학습·loss·dtype 보고서 |
| `../../0803_project_summary.md` | 2026-08-03 시점 종합 정리 |

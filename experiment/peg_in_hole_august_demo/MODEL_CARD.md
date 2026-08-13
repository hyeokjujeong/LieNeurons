# `khat_pointwise.pt` — 로드/사용 가이드 (Model Card)

Peg-in-hole 8월 데모용 stiffness 회귀 모델. **이 문서 하나로 다른 컴퓨터에서
weight 파일만 옮겨 바로 로드**할 수 있도록 필요한 전부를 적었다.

- 학습일: 2026-08-14 · 커밋 `0961b07` 기준 코드
- 원본 위치: `experiment/peg_in_hole_august_demo/khat_pointwise.pt` (414 KB)
- 파일을 어디로 옮겨도 됨 — 로드 시 경로만 지정 (아래)

---

## 1. 무엇을 하는 모델인가

hole 부품(90×90×50 mm 블록, 관통 원통 구멍)의 표면 point cloud를 입력받아
6×6 SPD stiffness $K$를 출력한다. 학습 라벨은 body frame 기준

$$K_{\rm body} = \mathrm{diag}(30,\,30,\,30,\;30,\,30,\,\mathbf{500})$$

(**[m; f] 순서**: 회전 3 + 병진 3, 구멍 축 = body $z$ = 마지막 슬롯).
즉 "구멍 축 병진만 뻣뻣, 나머지는 부드러움"을 부품 pose에 맞춰 congruence
수송한 값이다. 아키텍처가 다음을 **학습과 무관하게 보장**한다:

- $K(T\cdot P) = \mathrm{Ad}_T^{-\top}\,K(P)\,\mathrm{Ad}_T^{-1}$ — 정확한
  SE(3) congruence equivariance (실측 잔차 ~1e-11)
- $K = LL^\top$ — SPD 항상 보장 (임피던스 게인으로 안전)

## 2. 의존성

| 항목 | 요구 |
|---|---|
| 코드 | 이 repo, **커밋 `0961b07` 이후** (2026-08-10 리팩토링 `83bb536`의 [m; f] 규약 필수 — 그 이전 코드는 [f; m]이라 로드해도 잘못된 결과) |
| 모듈 | `experiment/pc_se3_congruence/pointwise_models.py` (+ 같은 폴더의 `se3_utils.py`, `pointwise_graph.py` 등 내부 의존) |
| 파이썬 | torch ≥ 2.x. GPU 불필요 (CPU 추론 ~수십 ms) |
| dtype | **float64 고정** — float32 캐스팅 시 등변성 잔차 증가 |

## 3. 체크포인트 파일 구조

`torch.save`된 dict:

| 키 | 내용 |
|---|---|
| `state_dict` | 모델 가중치 (48,769 params) |
| `model_kwargs` | 생성자 인자 — `{'channels': (16, 64, 64, 32), 'factors': 32, 'use_force_invariant': True}` (나머지는 클래스 기본값) |
| `contract` | 입출력 계약 (아래 §4와 동일 내용) |
| `train_meta` | 학습 데이터/epoch/최종 val_d/시드/라벨 diag |

`model_kwargs`가 파일 안에 있으므로 **아키텍처 설정을 외부에서 알 필요 없이**
아래 스니펫으로 로드된다.

## 4. 입출력 계약

- **입력**: `P` — `torch.float64`, `[B, 128, 3]`.
  **N=128 고정** (샘플링 밀도에 대한 ‖K‖ 드리프트가 알려진 이슈라 학습 해상도
  유지). 좌표는 **canonical 정규화**: 중심(표본 평균) 이동 + 최대 반지름 1.
- **출력**: `K` — `torch.float64`, `[B, 6, 6]`, SPD,
  **[m; f] 순서** (회전 블록 = 좌상단 3×3, 병진 블록 = 우하단 3×3).
  제어기가 [force; torque] 등 다른 순서면 블록 치환 필요.

## 5. 로드 — 최소 스니펫 (repo 루트에서)

```python
import torch
from experiment.pc_se3_congruence.pointwise_models import PointwiseStiffnessModel

torch.set_default_dtype(torch.float64)

ck = torch.load('/원하는/경로/khat_pointwise.pt', map_location='cpu',
                weights_only=False)
model = PointwiseStiffnessModel(**ck['model_kwargs']).double()
model.load_state_dict(ck['state_dict'])
model.eval()

K = model(P[None])[0]        # P: [128, 3] canonical cloud -> K: [6, 6]
```

## 6. 로드 — 권장 경로 (`KhatEstimator`)

canonical cloud 생성·캐시·pose 수송·등변성 인증까지 포함된 래퍼.
**checkpoint를 어느 경로에 두든 `ckpt_path`로 지정**하면 된다. canonical
cloud는 repo의 `real_objects/hole6{2,3,4}.stl`에서 자동 재생성된다
(별도 데이터 파일 불필요):

```python
from experiment.peg_in_hole_august_demo.khat_infer import KhatEstimator

est = KhatEstimator(ckpt_path='/원하는/경로/khat_pointwise.pt')
K_body = est.k_body('hole64')          # [6,6] — 임피던스 task-frame 게인
K_base = est.k_base('hole64', R, p)    # base-frame 표현 필요 시 (p는 m 단위)
print(est.certificate('hole64'))       # 등변성 잔차, ~1e-11 기대
```

## 7. 정상 동작 기준값 (sanity check)

로드 직후 아래와 일치해야 한다 (`python experiment/peg_in_hole_august_demo/khat_infer.py`):

| 부품 | 병진 블록 고유값 | 축 정렬 \|v·z\| | 회전 블록 고유값 |
|---|---|---|---|
| hole62 | (31.7, 32.5, **355.3**) | 1.0000 | 27~36 |
| hole63 | (31.9, 32.7, **327.5**) | 1.0000 | 27~40 |
| hole64 | (31.7, 32.2, **312.4**) | 0.9999 | 28~41 |

공통: SPD (모든 고유값 > 0), 등변성 잔차 < 1e-10.

**알려진 한계**: 축 병진 강성이 라벨 500 대비 ~310–355에서 포화한다
(pointwise 인코더의 이방성 표현 상한 — capacity 무관 재현 확인). 축 *방향*은
정확하므로, 절대 크기는 제어기 쪽 스칼라 게인 $\alpha$로 조정할 것.
$K_{\rm cmd} = \alpha K$는 equivariance를 깨지 않는다.

## 8. 학습 재현

```bash
# canonical-3 데이터셋 생성은 train_khat.py --data 기본 경로 참조 (STL에서 재생성 가능)
python experiment/peg_in_hole_august_demo/train_khat.py \
    --data data/real_objects/holes_canonical_axis.pt \
    --epochs 4000 --batch 3 --lr 2e-3 --channels 16 64 64 32 --factors 32
```

학습 표본은 hole62/63/64의 canonical cloud 3개 (identity pose, seed 7,
jitter 0) — pose 증강은 AIRM loss의 congruence 불변성 + 모델 등변성 때문에
gradient 기여가 정확히 0이라 의도적으로 생략했다.

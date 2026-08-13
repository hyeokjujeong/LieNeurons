"""K̂ 추론 래퍼 — 로봇 컴퓨터용 (README.md §4 계약).

사용 예 (repo 루트에서):

    from experiment.peg_in_hole_august_demo.khat_infer import KhatEstimator

    est = KhatEstimator()                      # 체크포인트 + canonical cloud 로드
    K_body = est.k_body('hole64')              # [6, 6] — task frame 게인으로 그대로 사용
    K_base = est.k_base('hole64', R, p)        # base-frame 표현이 필요할 때만
    print(est.certificate('hole64'))           # 등변성 잔차 (~1e-12)

규약: **[m; f] 순서** (rot 0:3, trans 3:6 — se3_utils.coadjoint와 동일),
float64. K_body는 ICP로 얻은 hole pose를 임피던스 제어기의 task frame으로
지정한 뒤 그 frame의 게인으로 넣는다 (README §4.2). 제어기가 [f; m]이나
[force; torque] 순서를 쓰면 블록 치환 후 투입할 것.
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, '..', '..'))

from experiment.pc_se3_congruence.pointwise_models import PointwiseStiffnessModel
from experiment.pc_se3_congruence.se3_utils import coadjoint, random_SE3

torch.set_default_dtype(torch.float64)

_CKPT = os.path.join(_HERE, 'khat_pointwise.pt')
_DATA = os.path.join(_HERE, '..', '..', 'data', 'real_objects',
                     'holes_canonical_axis.pt')
_NAMES = ('hole62', 'hole63', 'hole64')


def _build_canonical():
    """STL에서 canonical cloud를 재생성 (README §3.2와 동일: seed 7, N=128,
    무 jitter, 중심 이동 + 최대 반지름 1).  데이터 .pt가 없어도 (data/는
    gitignore) repo의 STL만으로 추론이 자립하도록 하는 폴백이다."""
    from experiment.pc_se3_congruence.mesh_pcd import (load_stl,
                                                       sample_mesh_surface)
    mesh_dir = os.path.join(_HERE, '..', '..', 'real_objects')
    P, c, s = {}, {}, {}
    for n in _NAMES:
        tri = load_stl(os.path.join(mesh_dir, f'{n}.stl'))
        gen = torch.Generator().manual_seed(7)
        P_mm = sample_mesh_surface(tri, 128, gen)
        c[n] = P_mm.mean(0)
        s[n] = (P_mm - c[n]).norm(dim=-1).max()
        P[n] = (P_mm - c[n]) / s[n]
    return P, c, s


class KhatEstimator:
    def __init__(self, ckpt_path=_CKPT, data_path=_DATA, device='cpu'):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model = PointwiseStiffnessModel(**ck['model_kwargs']).double()
        self.model.load_state_dict(ck['state_dict'])
        self.model.to(device).eval()
        self.meta = ck.get('train_meta', {})

        if os.path.exists(data_path):
            d = torch.load(data_path, map_location=device, weights_only=False)
            # canonical cloud (identity pose) — 학습 입력과 bit-identical
            self.P_canon = {n: d['P_train'][i] for i, n in enumerate(_NAMES)}
            self.center_mm = {n: d['center_mm'][i] for i, n in enumerate(_NAMES)}
            self.scale_mm = {n: d['scale_mm'][i] for i, n in enumerate(_NAMES)}
        else:
            self.P_canon, self.center_mm, self.scale_mm = _build_canonical()
        self._cache = {}

    def k_body(self, name):
        """[6, 6] body-frame K̂ — task frame 게인으로 직접 사용 (SI 해석은
        README §4.4: 병진 N/m, 회전 N·m/rad, 필요시 스칼라 α 배)."""
        if name not in self._cache:
            with torch.no_grad():
                self._cache[name] = self.model(self.P_canon[name][None])[0]
        return self._cache[name]

    def k_base(self, name, R, p):
        """base-frame 표현. R [3,3], p [3] = ICP hole pose (p는 미터).
        제어기가 task frame 지정을 지원하면 이 함수는 필요 없다."""
        G = coadjoint(torch.as_tensor(R, dtype=torch.float64),
                      torch.as_tensor(p, dtype=torch.float64))
        K = self.k_body(name)
        return G @ K @ G.T

    def certificate(self, name, n_T=5, trans_scale=0.5, seed=7):
        """등변성 인증: max_T |K(T·P) - G K(P) G^T|.  ~1e-12 기대."""
        gen = torch.Generator().manual_seed(seed)
        P = self.P_canon[name]
        K0 = self.k_body(name)
        worst = 0.0
        with torch.no_grad():
            for _ in range(n_T):
                R, p = random_SE3(trans_scale, gen)
                G = coadjoint(R, p)
                K1 = self.model((P @ R.T + p)[None])[0]
                worst = max(worst, float((K1 - G @ K0 @ G.T).abs().max()))
        return worst


if __name__ == '__main__':
    est = KhatEstimator()
    z = torch.tensor([0., 0., 1.])
    for n in _NAMES:
        K = est.k_body(n)
        lam, V = torch.linalg.eigh(K[3:6, 3:6])     # 병진 블록 ([m; f] 저장)
        print(f'{n}: K_ff eig {[round(float(x), 1) for x in lam]}  '
              f'axis align {abs(float(V[:, -1] @ z)):.4f}  '
              f'equivariance {est.certificate(n):.2e}  '
              f'(scale_mm {float(est.scale_mm[n]):.2f})')

"""hole62/63/64 STL -> K̂ 학습용 canonical 데이터셋 (.pt).

`experiment/peg_in_hole_august_demo/train_khat.py` 와 `khat_infer.py` 가 읽는
파일을 만든다.  레시피는 MODEL_CARD.md §8 그대로다:

  train  mesh 당 1개, **identity pose**, seed 7, jitter 0, N=128.
         pose 증강은 일부러 넣지 않는다 — AIRM loss 가 congruence 불변이고
         모델이 등변이므로 pose 를 돌린 표본의 gradient 기여가 정확히 0이다.
  val    mesh 당 --n-val-per-mesh 개.  **샘플링 seed 를 바꾼다** — 이것이
         이 문제의 실제 일반화 축이다 (MODEL_CARD §4 의 '샘플링 밀도에 대한
         ‖K‖ 드리프트').  pose 도 무작위로 주는데, 이쪽은 일반화가 아니라
         등변성의 수치 인증으로 읽어야 한다 (구조상 손실이 불변).

라벨은 body frame 의 K_body = diag(30,30,30, 30,30,500) 을 각 표본 pose 로
congruence 수송한 값이다 (**[m; f] 순서**, 구멍 축 = body z = 마지막 슬롯).
identity pose 인 train 표본에서는 라벨이 곧 K_body 다.

주의: 축 병진 500 은 이 형상에서 물리적으로 실현 불가능한 값이다 — 기본
아키텍처(`--pitch none`)의 상한이 ~300 이다.  왜 그런지, 그리고 무엇을 바꿔야
하는지는 experiment/peg_in_hole_august_demo/STIFFNESS_CEILING.md 에 있다.

repo 루트에서:
    python data_gen/gen_holes_canonical.py
"""

import argparse
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from experiment.pc_se3_congruence.data_synth import random_SO3
from experiment.pc_se3_congruence.mesh_pcd import (body_frame_stiffness_labels,
                                                   load_stl,
                                                   sample_mesh_surface)

MESHES = ('hole62', 'hole63', 'hole64')


def canonical(tri, n_points, seed):
    """표면 area-weighted 샘플 -> 중심 이동 + 최대 반지름 1.  (P, center, scale)."""
    gen = torch.Generator().manual_seed(seed)
    P_mm = sample_mesh_surface(tri, n_points, gen)
    c = P_mm.mean(0)
    s = (P_mm - c).norm(dim=-1).max()
    return (P_mm - c) / s, c, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh-dir', default='real_objects')
    ap.add_argument('--out', default='data/real_objects/holes_canonical_axis.pt')
    ap.add_argument('--n-points', type=int, default=128)
    ap.add_argument('--sample-seed', type=int, default=7,
                    help='train(=canonical) 표본의 샘플링 seed. khat_infer.py의 '
                         'STL 폴백과 bit-identical 해야 하므로 바꾸지 말 것')
    ap.add_argument('--n-val-per-mesh', type=int, default=32)
    ap.add_argument('--val-seed', type=int, default=1000)
    ap.add_argument('--pose-seed', type=int, default=11)
    ap.add_argument('--trans-scale', type=float, default=0.5)
    ap.add_argument('--k-body', type=float, nargs=6,
                    default=[30., 30., 30., 30., 30., 500.],
                    help='[m; f] 순서. 마지막 슬롯 = 구멍 축 병진')
    a = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    K_body = torch.diag(torch.tensor(a.k_body))
    tris = {m: load_stl(os.path.join(a.mesh_dir, f'{m}.stl')) for m in MESHES}

    # ---- train: identity pose, mesh 당 1개
    P_tr, cen, sca = [], [], []
    for m in MESHES:
        P, c, s = canonical(tris[m], a.n_points, a.sample_seed)
        P_tr.append(P)
        cen.append(c)
        sca.append(s)
    P_tr = torch.stack(P_tr)
    K_tr = K_body.expand(len(MESHES), 6, 6).clone()          # identity pose

    # ---- val: 샘플링 seed 를 바꾸고 random SE(3) 로 수송
    gp = torch.Generator().manual_seed(a.pose_seed)
    P_va, Rs, ps, mid = [], [], [], []
    for k, m in enumerate(MESHES):
        for j in range(a.n_val_per_mesh):
            P, _, _ = canonical(tris[m], a.n_points,
                                a.val_seed + 1000 * k + j)
            R = random_SO3(gp)
            p = torch.randn(3, generator=gp) * a.trans_scale
            P_va.append(P @ R.T + p)
            Rs.append(R)
            ps.append(p)
            mid.append(k)
    P_va = torch.stack(P_va)
    K_va = body_frame_stiffness_labels(torch.stack(Rs), torch.stack(ps), K_body)

    out = {
        'P_train': P_tr, 'K_train': K_tr,
        'P_val': P_va, 'K_val': K_va,
        'mesh_id_val': torch.tensor(mid),
        'center_mm': torch.stack(cen), 'scale_mm': torch.stack(sca),
        'meta': {'meshes': list(MESHES), 'n_points': a.n_points,
                 'K_body_diag': list(a.k_body), 'ordering': '[m; f]',
                 'sample_seed': a.sample_seed, 'val_seed': a.val_seed,
                 'pose_seed': a.pose_seed, 'trans_scale': a.trans_scale,
                 'train_pose': 'identity', 'jitter': 0.0},
    }
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    torch.save(out, a.out)

    # 저장한 라벨이 실제로 SPD 이고 수송이 규약과 맞는지 즉시 확인한다.
    lam = torch.linalg.eigvalsh(K_va)
    print(f'[out] {a.out}')
    print(f'  train P {tuple(P_tr.shape)}  K {tuple(K_tr.shape)}  (identity pose)')
    print(f'  val   P {tuple(P_va.shape)}  K {tuple(K_va.shape)}  '
          f'mesh당 {a.n_val_per_mesh}개')
    print(f'  scale_mm {[round(float(x), 2) for x in torch.stack(sca)]}')
    print(f'  val 라벨 고유값 min {float(lam.min()):.4g}  '
          f'(SPD 이어야 함)  cond max {float((lam[:, -1] / lam[:, 0]).max()):.4g}')
    print(f'  K_body diag {a.k_body}  [m; f]')


if __name__ == '__main__':
    main()

"""STL mesh -> surface point cloud sampling (dependency-free).

Binary STL parser + area-weighted surface sampler, mirroring the sampling
philosophy of the peg-hole v2 labels (area-weighted, resolution-convergent).
Outputs float64 torch tensors ready for the pc_se3_congruence pipeline
(scale normalisation + random SE(3) pose + jitter happen downstream, exactly
as in :func:`data_synth.object_clouds`).
"""

import struct

import numpy as np
import torch


def load_stl(path):
    """Read an STL file (binary or ASCII) -> triangle vertices [T, 3, 3]."""
    with open(path, 'rb') as fh:
        head = fh.read(80)
        rest = fh.read()
    n = struct.unpack('<I', rest[:4])[0]
    if len(rest) == 4 + 50 * n:                      # binary layout check
        rec = np.frombuffer(rest[4:], dtype=np.uint8).reshape(n, 50)
        tri = rec[:, 12:48].copy().view('<f4').reshape(n, 3, 3)
        return np.asarray(tri, dtype=np.float64)
    # ASCII fallback: "vertex x y z" lines in facet order
    text = (head + rest).decode('ascii', errors='ignore')
    vs = [list(map(float, ln.split()[1:4]))
          for ln in text.splitlines() if ln.strip().startswith('vertex')]
    if len(vs) < 3 or len(vs) % 3:
        raise ValueError(f'{path}: not a valid STL file')
    return np.asarray(vs, dtype=np.float64).reshape(-1, 3, 3)


def sample_mesh_surface(tri, n_points, gen=None, dtype=torch.float64):
    """Area-weighted uniform surface sampling.  tri [T, 3, 3] -> [n_points, 3].

    Triangles are chosen with probability proportional to their area, then a
    point is drawn uniformly inside each via the sqrt barycentric trick.
    """
    t = torch.as_tensor(tri, dtype=dtype)
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    area = torch.linalg.cross(b - a, c - a).norm(dim=-1)         # 2x, cancels
    idx = torch.multinomial(area, n_points, replacement=True, generator=gen)
    u = torch.rand(n_points, 1, generator=gen, dtype=dtype).sqrt()
    v = torch.rand(n_points, 1, generator=gen, dtype=dtype)
    return (1 - u) * a[idx] + u * (1 - v) * b[idx] + u * v * c[idx]


def mesh_clouds(n_samples, n_points, mesh_paths, gen=None, dtype=torch.float64,
                jitter=0.01, trans_scale=0.5, center=True, unit_scale=True,
                return_pose=False):
    """STL 파일들 -> object_clouds와 동일 규격의 [S, N, 3] 데이터셋.

    각 표본마다 mesh 하나를 고르고(mixed), 표면 area-weighted 샘플 후
    중심 이동 + 단위 스케일 정규화(최대 반지름 1) + random SE(3) + jitter.

    return_pose=True 이면 (P, R, p)를 돌려준다 — body-frame 라벨을 같은
    pose로 congruence 수송할 때 필요 (난수 소비 순서는 동일하므로 같은
    seed에서 P는 return_pose와 무관하게 bit-identical).
    """
    from experiment.pc_se3_congruence.data_synth import random_SO3
    tris = [load_stl(p) for p in mesh_paths]
    out, Rs, ps = [], [], []
    for _ in range(n_samples):
        k = int(torch.randint(len(tris), (1,), generator=gen))
        P = sample_mesh_surface(tris[k], n_points, gen, dtype)
        if center:
            P = P - P.mean(dim=0, keepdim=True)
        if unit_scale:
            P = P / P.norm(dim=-1).max().clamp_min(1e-12)
        P = P + jitter * torch.randn(P.shape, generator=gen, dtype=dtype)
        R = random_SO3(gen, dtype)
        p = torch.randn(3, generator=gen, dtype=dtype) * trans_scale
        out.append(P @ R.T + p)
        Rs.append(R)
        ps.append(p)
    if return_pose:
        return torch.stack(out), torch.stack(Rs), torch.stack(ps)
    return torch.stack(out)


def body_frame_stiffness_labels(R, p, K_body):
    """Body-frame 강성 K_body를 각 표본 pose (R, p)로 congruence 수송.

    저장 규약 [m; f] (se3_utils.coadjoint와 동일)에서 wrench는
    w' = Ad_T^{-T} w, 즉 m' = Rm + p x Rf, f' = Rf 이므로
    G = [[R, hat(p)R], [0, R]] 이고 K_world = G K_body G^T.
    모델의 equivariance K(T.P) = Ad_T^{-T} K(P) Ad_T^{-1} 과 같은 규칙이다.
    se3_utils.coadjoint의 배치 버전 — 규약 검증 테스트는 두 구현을 대조할 것.
    """
    S = R.shape[0]
    G = torch.zeros(S, 6, 6, dtype=R.dtype)
    G[:, 0:3, 0:3] = R
    G[:, 3:6, 3:6] = R
    ph = torch.zeros(S, 3, 3, dtype=R.dtype)
    ph[:, 0, 1], ph[:, 0, 2] = -p[:, 2], p[:, 1]
    ph[:, 1, 0], ph[:, 1, 2] = p[:, 2], -p[:, 0]
    ph[:, 2, 0], ph[:, 2, 1] = -p[:, 1], p[:, 0]
    G[:, 0:3, 3:6] = ph @ R
    return G @ K_body.to(R.dtype) @ G.transpose(-1, -2)

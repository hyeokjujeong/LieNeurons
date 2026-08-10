"""[CURRENT DATA]
One loader for any on-disk (point cloud, stiffness) dataset.

``load_split(path, split)`` is the single entry point.  It looks at ``path`` and
picks the format, so callers never branch on which dataset they were given:

    <dir>/meta.json exists   peg-and-hole shard dataset
                             (data_gen/gen_peg_hole_pcd.py; see
                             peg_hole_data_loader.py for its own details)
    anything else            generic (points, K) file or directory, below

GENERIC FORMAT.  A ``.npz`` or ``.pt`` file holding two arrays:

    points   [S, N, 3]   S clouds of N points
    K        [S, 6, 6]   the matching stiffness, symmetric positive definite

Key aliases so an existing file usually loads unchanged: ``points`` / ``P`` /
``cloud`` / ``xyz``, and ``K`` / ``stiffness`` / ``K_gt``.  A ``.pt`` may hold a
dict with those keys or a plain ``(points, K)`` tuple.  Point ``path`` at a
directory holding ``train.npz`` and ``val.npz``, or at a single file, which is
split by ``val_frac`` after a deterministic shuffle seeded by ``seed``.

``.npz`` and ``.pt`` are the only containers read directly.  json / parquet /
safetensors / hdf5 / csv hold the same two arrays and convert in three lines --
the experiment README has the recommended layout and snippet for each.

Everything returns ``(P, K, info)``.  ``info`` carries whatever extra the format
has -- ``stage`` for peg-and-hole, empty for the generic one.

``K`` is the map from a twist to a wrench, ``[m; f] = K [omega; v]``.  N must be
the same for every scene (they are stacked into one batch).

THREE THINGS THAT DO NOT RAISE IF YOU GET THEM WRONG:

1.  BLOCK ORDER.  The model works in the wrench basis with MOMENT first,
    ``F = [m; f]``, so ``K[0:3, 0:3]`` is ROTATIONAL and ``K[3:6, 3:6]``
    TRANSLATIONAL stiffness.  Force-first matrices are also symmetric 6x6 SPD,
    so nothing complains -- the numbers are just wrong.  Permute before saving:
    ``K = K[:, [3,4,5,0,1,2]][:, :, [3,4,5,0,1,2]]``.

2.  MOMENT REFERENCE POINT -- the ORIGIN of the frame the points are given in,
    not the centroid.  Labels must be built as m_i = r_i x f_i with r_i the
    coordinate stored in ``points``; that is what makes K transform by
    congruence when the scene moves (measured residual 1.7e-15).  A K measured
    about the centroid would be translation INVARIANT instead, which is not the
    law the model is built around.
    To move a K measured about some point c to the origin (A = Ad^{-T}(I, c) in
    [m; f] order; congruence, so symmetry and definiteness survive):
        cx = [[0, -c2, c1], [c2, 0, -c0], [-c1, c0, 0]]
        A  = [[I, cx], [0, I]];   K_origin = A @ K_c @ A.T
    Keep the object near the origin: moments grow with |r|, so the same cloud
    translated along x gives cond(K) 9.8 / 2.1e1 / 1.3e4 / 1.2e8 at |centre| =
    0 / 1 / 10 / 100, with the mm block swamping ff by 8000x at the far end.

3.  RESOLUTION.  ``n_points`` subsamples only where the label can be RECOMPUTED
    from the subsample (peg-and-hole).  For a generic dataset K is a function of
    the cloud you measured it on, so pairing a subset with the full cloud's
    label would make the regression ill-posed; ``n_points`` is refused there.
    Downsample the clouds and recompute K yourself before saving.

Check symmetry and definiteness when you WRITE the file, not at training time:
the loss takes a Cholesky factor of the target, so one bad sample kills the run.

Memory: the local graph builds a [B, N, N] distance matrix, so N decides
whether a run fits.  N=1024 with an eval chunk of 64 is ~537 MB.
"""
import os

import numpy as np
import torch

POINT_KEYS = ('points', 'P', 'cloud', 'xyz')
K_KEYS = ('K', 'stiffness', 'K_gt')


def is_peg_hole(path):
    return os.path.isfile(os.path.join(path, 'meta.json'))


def _first_key(container, names, path, what):
    for k in names:
        if k in container:
            return container[k]
    raise KeyError(f'{path}: {what} 배열이 없다. 찾아본 이름 {names}, '
                   f'파일에 있는 이름 {sorted(container)}')


def _read_file(path):
    if path.endswith('.pt'):
        obj = torch.load(path, map_location='cpu')
        if isinstance(obj, (tuple, list)) and len(obj) == 2:
            pts, K = obj
        else:
            pts = _first_key(obj, POINT_KEYS, path, 'point cloud')
            K = _first_key(obj, K_KEYS, path, 'stiffness')
        return torch.as_tensor(pts).double(), torch.as_tensor(K).double()
    with np.load(path) as z:
        pts = _first_key(z, POINT_KEYS, path, 'point cloud')
        K = _first_key(z, K_KEYS, path, 'stiffness')
        return (torch.from_numpy(np.asarray(pts)).double(),
                torch.from_numpy(np.asarray(K)).double())


def _load_generic(path, split, n, n_points, seed, val_frac):
    if n_points is not None:
        raise ValueError(
            f'{path}: 이 형식은 서브샘플을 지원하지 않는다. K 는 측정한 그 '
            'cloud 의 함수라서 부분집합에 원래 라벨을 붙이면 회귀가 ill-posed '
            '해진다 (입력이 답을 결정하지 않는다). cloud 를 줄이고 K 를 다시 '
            '계산해서 저장할 것')
    if os.path.isdir(path):
        src = next((os.path.join(path, split + ext)
                    for ext in ('.npz', '.pt')
                    if os.path.isfile(os.path.join(path, split + ext))), None)
        if src is None:
            raise FileNotFoundError(
                f'{path} 에 {split}.npz 도 {split}.pt 도 meta.json 도 없다')
        pts, K = _read_file(src)
    else:
        src = path
        pts, K = _read_file(src)
        # 같은 seed 면 train/val 호출이 정확히 상보적인 조각을 받는다.
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(pts.shape[0], generator=g)
        n_val = max(1, round(pts.shape[0] * val_frac))
        sel = perm[:-n_val] if split == 'train' else perm[-n_val:]
        pts, K = pts[sel], K[sel]
    if n is not None:
        pts, K = pts[:n], K[:n]
    return pts, K, {}, src


def load_split(path, split, n=None, n_points=None, seed=0, val_frac=0.1,
               lambda_body=None, relabel=True, device='cpu', verbose=True):
    """Load ``split`` ('train' / 'val') -> (P [S,N,3], K [S,6,6], info)."""
    if is_peg_hole(path):
        from data_loader.peg_hole_data_loader import load_peg_hole_split
        pts, K, info = load_peg_hole_split(
            path, split, n=n, n_points=n_points, seed=seed,
            lambda_body=lambda_body, relabel=relabel, extras=True,
            device=device)
        src = path
    else:
        pts, K, info, src = _load_generic(path, split, n, n_points, seed,
                                          val_frac)
    if verbose:
        print(f'[data] {split}: {src}  P {tuple(pts.shape)}  '
              f'K {tuple(K.shape)}', flush=True)
    return pts.to(device), K.to(device), info

"""[CURRENT DATA]
Loader for the peg-and-hole PCD -> stiffness dataset.

Shards are written by ``data_gen/gen_peg_hole_pcd.py``; the scene/label
definitions live in ``experiment/pc_se3_congruence/peg_hole_synth.py``.

The training target is composed here as

    K = K_contact + lambda_body * K_body

so the contact-dominance / conditioning trade-off can be swept without
regenerating data (``lambda_body=None`` uses the calibrated default recorded
in ``meta.json``).

Subsampling and RELABELLING.  ``n_points`` draws a per-sample random subset
with a deterministic per-sample seed.  The stored labels are functions of the
FULL stored cloud, so pairing them with a subsample would make the regression
ill-posed -- the answer is not determined by the input.  Measured on this
dataset, the stored 2048-point label sits AIRM 2.2 away (median) from the
label of its own 512-point subsample, and two different subsamples of one
scene differ from each other by 1.4-3.3, so no model can do better than that
floor.  ``relabel=True`` (the default whenever ``n_points`` is set) therefore
recomputes ``K_contact`` and ``K_body`` from the subsampled points, restoring
"label = f(input)".

Because a uniform subsample of a uniform surface sample is itself a uniform
surface sample, downsampling does not distort the distribution; the label
structure is preserved down to N ~ 256 (stage contrast insert/search 3.3-3.8,
free contact exactly 0) and only collapses at N = 128, where the surface
pitch exceeds the contact kernel sigma_c.

Recomputation is deterministic given (n_points, seed), so results are cached
under ``<root>/cache/n<N>_seed<S>/``.
"""
import json
import os

import numpy as np
import torch

# The first dataset version whose stored K is in the ANGULAR/MOMENT-FIRST
# basis [m; f].  Anything older was written force-first and is converted on
# load by the block swap Pi = [[0, I], [I, 0]]: K_new = Pi K_old Pi.  That is a
# congruence by an orthogonal involution, so symmetry, definiteness and the
# whole eigenvalue spectrum are untouched -- only which block is called ff.
MOMENT_FIRST_VERSION = 3
_SWAP = [3, 4, 5, 0, 1, 2]


def swap_blocks(K):
    """Pi K Pi for K [..., 6, 6] — force-first <-> moment-first."""
    return K[..., _SWAP, :][..., :, _SWAP]


SCALAR_KEYS = ('r_peg', 'H', 'T', 'clearance', 'depth', 'tilt',
               'yaw_mismatch', 'noise')


def read_meta(root):
    path = os.path.join(root, 'meta.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found -- generate the dataset first with '
            'python data_gen/gen_peg_hole_pcd.py --out ' + root)
    with open(path) as f:
        return json.load(f)


def _subsample_index(n_total, n_points, seed, global_idx):
    g = torch.Generator().manual_seed(seed * 1_000_003 + global_idx)
    return torch.randperm(n_total, generator=g)[:n_points]


def _cache_dir(root, n_points, seed):
    # 규약 버전을 경로에 넣는다: force-first 시절 캐시를 조용히 재사용하면
    # 라벨만 옛 basis 로 섞여 들어간다.
    return os.path.join(root, 'cache',
                        f'n{n_points}_seed{seed}_v{MOMENT_FIRST_VERSION}')


def relabel_subsample(P, part, area_peg, area_plate, cfg, device='cpu'):
    """Recompute (K_contact, K_body) from the points actually being fed.

    ``cfg`` is ``meta['cfg']`` -- the exact configuration the dataset was
    generated with -- and the areas are the scene's TRUE surface areas, so the
    recomputed label is the same Monte-Carlo estimator of the same surface
    integral, just at a different sample size.  That is what makes it
    resolution-convergent rather than a different quantity.
    """
    from experiment.pc_se3_congruence.peg_hole_synth import (body_K_areas,
                                                             peg_contact_K)
    Pd, pd = P.double().to(device), part.to(device)
    ap, al = area_peg.double().to(device), area_plate.double().to(device)
    Kc = peg_contact_K(Pd, pd, ap, al, sigma_c=cfg['sigma_c'],
                       contact_radius=cfg['contact_radius'],
                       candidate_k=cfg['contact_candidates'])
    Kb = body_K_areas(Pd, ap + al, sigma_b=cfg['body_sigma'],
                      body_radius=cfg['body_radius'],
                      candidate_k=cfg['body_candidates'])
    return Kc.cpu(), Kb.cpu()


def _relabelled_shard(root, split, sh, offset, n_points, seed, cfg,
                      device='cpu', use_cache=True):
    """[S,N,3] points, [S,N] part, and labels recomputed on the subsample."""
    cdir = _cache_dir(root, n_points, seed)
    cpath = os.path.join(cdir, sh['file'])
    with np.load(os.path.join(root, sh['file'])) as z:
        pts = torch.from_numpy(z['points']).double()
        part = torch.from_numpy(z['part'])
        ap = torch.from_numpy(z['area_peg'])
        al = torch.from_numpy(z['area_plate'])
    sel = torch.stack([_subsample_index(pts.shape[1], n_points, seed,
                                        offset + i)
                       for i in range(pts.shape[0])])
    pts = torch.gather(pts, 1, sel.unsqueeze(-1).expand(-1, -1, 3))
    part = torch.gather(part, 1, sel)
    if use_cache and os.path.exists(cpath):
        with np.load(cpath) as z:
            return pts, part, (torch.from_numpy(z['K_contact']),
                               torch.from_numpy(z['K_body']))
    Kc, Kb = relabel_subsample(pts, part, ap, al, cfg, device=device)
    if use_cache:
        os.makedirs(cdir, exist_ok=True)
        tmp = cpath + '.tmp.npz'
        np.savez(tmp, K_contact=Kc.numpy(), K_body=Kb.numpy())
        os.replace(tmp, cpath)
    return pts, part, (Kc, Kb)


def load_peg_hole_split(root, split, n=None, n_points=None, seed=0,
                        lambda_body=None, extras=False, relabel=True,
                        device='cpu', cache=True):
    """Load the first ``n`` samples of a split into memory (CPU tensors).

    Returns ``(P, K)`` with P [S, N, 3] float64 and K [S, 6, 6] float64, or
    ``(P, K, info)`` with ``extras=True`` where ``info`` carries part masks,
    stage / profile ids, the raw label terms and per-scene scalars.

    ``n_points`` subsamples each cloud deterministically.  With ``relabel``
    (the default) the labels are then recomputed FROM the subsample, which is
    what keeps the regression well posed -- see the module docstring.  Pass
    ``relabel=False`` only to reproduce the ill-posed pairing on purpose.
    """
    meta = read_meta(root)
    shards = meta['splits'][split]['shards']
    have = sum(s['n'] for s in shards)
    if n is None:
        n = have
    if have < n:
        raise ValueError(f'split {split!r} has {have} samples on disk, '
                         f'{n} requested ({root})')
    if lambda_body is None:
        lambda_body = meta['cfg']['lambda_body']
    do_relabel = relabel and n_points is not None \
        and n_points < meta['n_points']
    # 재라벨 경로는 현행 peg_hole_synth 를 다시 돌리므로 이미 새 basis 다.
    old_basis = meta.get('version', 1) < MOMENT_FIRST_VERSION
    if old_basis and not do_relabel:
        print(f"[peg_hole] v{meta.get('version')} 데이터셋 — 저장 라벨을 "
              f'moment-first 로 변환해서 읽는다 (Pi K Pi)', flush=True)

    P, K, info = [], [], {k: [] for k in
                          ('part', 'stage', 'profile', 'n_peg', 'area_peg',
                           'area_plate', 'K_contact', 'K_body') + SCALAR_KEYS}
    taken = 0
    for sh in shards:
        if taken >= n:
            break
        take = min(sh['n'], n - taken)
        if do_relabel:
            pts, part, (Kc, Kb) = _relabelled_shard(
                root, split, sh, taken, n_points, seed, meta['cfg'],
                device=device, use_cache=cache)
            pts, part, Kc, Kb = pts[:take], part[:take], Kc[:take], Kb[:take]
        with np.load(os.path.join(root, sh['file'])) as z:
            if not do_relabel:
                pts = torch.from_numpy(z['points'][:take]).double()
                part = torch.from_numpy(z['part'][:take])
                Kc = torch.from_numpy(z['K_contact'][:take])
                Kb = torch.from_numpy(z['K_body'][:take])
                if old_basis:
                    Kc, Kb = swap_blocks(Kc), swap_blocks(Kb)
                if n_points is not None:
                    sel = torch.stack([
                        _subsample_index(pts.shape[1], n_points, seed,
                                         taken + i) for i in range(take)])
                    pts = torch.gather(pts, 1,
                                       sel.unsqueeze(-1).expand(-1, -1, 3))
                    part = torch.gather(part, 1, sel)
            P.append(pts)
            K.append(Kc + lambda_body * Kb)
            if extras:
                info['part'].append(part)
                info['K_contact'].append(Kc)
                info['K_body'].append(Kb)
                for k in ('stage', 'profile', 'n_peg', 'area_peg',
                          'area_plate') + SCALAR_KEYS:
                    info[k].append(torch.from_numpy(z[k][:take]))
            taken += take
    P, K = torch.cat(P), torch.cat(K)
    if not extras:
        return P, K
    info = {k: torch.cat(v) for k, v in info.items() if v}
    info['lambda_body'] = lambda_body
    info['meta'] = meta
    return P, K, info


class PegHoleDataset(torch.utils.data.Dataset):
    """Streaming per-sample view (keeps one shard cached), for future
    DataLoader-based training at full dataset scale."""

    def __init__(self, root, split, n_points=None, lambda_body=None, seed=0,
                 relabel=True, device='cpu'):
        self.root, self.split, self.seed = root, split, seed
        self.n_points, self.device = n_points, device
        meta = read_meta(root)
        self.cfg = meta['cfg']
        self.lambda_body = (meta['cfg']['lambda_body']
                            if lambda_body is None else lambda_body)
        self.relabel = (relabel and n_points is not None
                        and n_points < meta['n_points'])
        self.old_basis = meta.get('version', 1) < MOMENT_FIRST_VERSION
        self.shards = meta['splits'][split]['shards']
        self.offsets = np.cumsum([0] + [s['n'] for s in self.shards])
        self._cache_idx, self._cache = None, None

    def __len__(self):
        return int(self.offsets[-1])

    def _shard(self, si):
        if self._cache_idx != si:
            if self.relabel:
                pts, part, (Kc, Kb) = _relabelled_shard(
                    self.root, self.split, self.shards[si],
                    int(self.offsets[si]), self.n_points, self.seed,
                    self.cfg, device=self.device)
                self._cache = {'points': pts, 'part': part,
                               'K_contact': Kc, 'K_body': Kb}
            else:
                path = os.path.join(self.root, self.shards[si]['file'])
                with np.load(path) as z:
                    self._cache = {k: torch.from_numpy(z[k]) for k in
                                   ('points', 'part', 'K_contact', 'K_body')}
                self._cache['points'] = self._cache['points'].double()
                if self.old_basis:
                    for k in ('K_contact', 'K_body'):
                        self._cache[k] = swap_blocks(self._cache[k])
            self._cache_idx = si
        return self._cache

    def __getitem__(self, i):
        si = int(np.searchsorted(self.offsets, i, side='right') - 1)
        z = self._shard(si)
        j = i - int(self.offsets[si])
        P, part = z['points'][j], z['part'][j]
        K = z['K_contact'][j] + self.lambda_body * z['K_body'][j]
        if self.n_points is not None and not self.relabel:
            sel = _subsample_index(P.shape[0], self.n_points, self.seed, i)
            P, part = P[sel], part[sel]
        return P, K, part

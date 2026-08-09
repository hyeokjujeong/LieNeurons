"""Generate the peg-and-hole point-cloud -> stiffness dataset (npz shards).

Scenes, labels and calibration are defined in
``experiment/pc_se3_congruence/peg_hole_synth.py``; this script only handles
sharded, resumable, reproducible on-disk generation.

Reproducibility: every shard is generated from its own deterministic seed
``master_seed * 1_000_003 + split_index * 65_536 + shard_index``, so shards
can be (re)generated in any order and interrupted runs resume by skipping
shard files already recorded in ``meta.json``.  ``meta.json`` is rewritten
atomically after every shard, so a partially generated dataset is loadable at
any time (the loader simply sees fewer shards).

Precision: points are stored float32; labels are float64 computed FROM the
float32-rounded points, so ``stored K == label(stored points)`` exactly.

Usage (repo root, conda env ``lieneurons``):
  python data_gen/gen_peg_hole_pcd.py --out data/peg_hole/v1 \
      --n-train 102400 --n-val 10240 --n-test 10240 --n-points 2048
  python data_gen/gen_peg_hole_pcd.py --inspect     # calibration stats only
"""
import argparse
import json
import os
import sys
import time

sys.path.append('.')

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from experiment.pc_se3_congruence.peg_hole_synth import (DEFAULT_CFG, STAGES,
                                                         PROFILE_TYPES,
                                                         compose_K,
                                                         generate_batch,
                                                         make_cfg)

VERSION = 2   # v2: area-weighted (resolution-convergent) labels
SPLITS = ('train', 'val', 'test')
# generation order only -- val/test first so training smoke tests can start
# while the train split is still streaming in; seeds stay order-independent
GEN_ORDER = ('val', 'test', 'train')
_SEED_OFFSET = {'train': 0, 'val': 1, 'test': 2}
ARRAY_KEYS = ('points', 'part', 'K_contact', 'K_body', 'area_peg',
              'area_plate', 'stage', 'profile', 'n_peg', 'r_peg', 'H', 'T',
              'clearance', 'depth', 'tilt', 'yaw_mismatch', 'noise')


def shard_seed(master, split, idx):
    return master * 1_000_003 + _SEED_OFFSET[split] * 65_536 + idx


def atomic_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def load_meta(out):
    path = os.path.join(out, 'meta.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def make_meta(args, cfg):
    return {
        'version': VERSION,
        'n_points': args.n_points,
        'shard_size': args.shard_size,
        'master_seed': args.seed,
        'cfg': {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in cfg.items()},
        'stages': list(STAGES),
        'profile_types': list(PROFILE_TYPES),
        'array_keys': list(ARRAY_KEYS),
        'splits': {s: {'n_target': getattr(args, f'n_{s}'), 'shards': []}
                   for s in SPLITS},
    }


def write_shard(out, split, idx, batch):
    fname = f'{split}_{idx:05d}.npz'
    arrays = {k: batch[k].numpy() for k in ARRAY_KEYS}
    tmp = os.path.join(out, fname + '.tmp.npz')
    np.savez(tmp, **arrays)
    os.replace(tmp, os.path.join(out, fname))
    return fname


def spd_check(batch, cfg):
    K = compose_K(batch['K_contact'], batch['K_body'], cfg['lambda_body'])
    lam = torch.linalg.eigvalsh(K)
    lam_min = lam[:, 0].min().item()
    cond_max = (lam[:, -1] / lam[:, 0]).max().item()
    if lam_min <= 1e-10:
        raise RuntimeError(f'composed K not safely SPD: lam_min={lam_min:.3e}')
    return lam_min, cond_max


def generate(args, cfg):
    out = args.out
    os.makedirs(out, exist_ok=True)
    meta = load_meta(out)
    if meta is None:
        meta = make_meta(args, cfg)
    else:
        for key in ('n_points', 'shard_size', 'master_seed'):
            if meta[key] != getattr(args, key if key != 'master_seed'
                                    else 'seed'):
                raise ValueError(
                    f'{out}/meta.json has {key}={meta[key]}, which conflicts '
                    'with the requested settings; use a fresh --out')
        for s in SPLITS:
            meta['splits'][s]['n_target'] = getattr(args, f'n_{s}')
    total_todo = sum(getattr(args, f'n_{s}') for s in SPLITS)
    done_before = sum(sh['n'] for s in SPLITS
                     for sh in meta['splits'][s]['shards'])
    print(f'[gen] out={out}  target={total_todo}  already={done_before}  '
          f'N={args.n_points}  shard={args.shard_size}  device={args.device}')

    t0, done_run = time.time(), 0
    for split in GEN_ORDER:
        n_target = getattr(args, f'n_{split}')
        entry = meta['splits'][split]
        idx = 0
        remaining = n_target - sum(sh['n'] for sh in entry['shards'])
        while remaining > 0:
            fname = f'{split}_{idx:05d}.npz'
            if any(sh['file'] == fname for sh in entry['shards']):
                idx += 1
                continue
            n = min(args.shard_size, remaining)
            seed = shard_seed(args.seed, split, idx)
            t1 = time.time()
            batch = generate_batch(n, args.n_points, seed, cfg,
                                   device=args.device)
            lam_min, cond_max = spd_check(batch, cfg)
            write_shard(out, split, idx, batch)
            entry['shards'].append({'file': fname, 'n': n, 'seed': seed})
            atomic_json(os.path.join(out, 'meta.json'), meta)
            done_run += n
            dt = time.time() - t1
            rate = n / dt
            left = (total_todo - done_before - done_run) / max(rate, 1e-9)
            print(f'[{split} {idx:05d}] n={n}  {dt:6.1f}s ({rate:5.1f}/s)  '
                  f'lam_min {lam_min:.2e}  cond_max {cond_max:.1e}  '
                  f'ETA {left/60:6.1f} min', flush=True)
            remaining -= n
            idx += 1
    print(f'[gen] complete in {(time.time()-t0)/60:.1f} min')


def inspect(args, cfg):
    """Calibration / sanity statistics on a throwaway batch."""
    b = generate_batch(args.inspect_n, args.n_points, args.seed, cfg,
                       device=args.device)
    Kc, Kb, stage = b['K_contact'], b['K_body'], b['stage']
    print(f'--- inspect: {args.inspect_n} scenes, N={args.n_points} ---')
    frac = b['n_peg'].double() / args.n_points
    print(f'peg point fraction: {frac.min():.3f}..{frac.max():.3f}')
    print(f'stage counts: '
          + ', '.join(f'{s}={int((stage == i).sum())}'
                      for i, s in enumerate(STAGES)))
    print(f'profile counts: '
          + ', '.join(f'{p}={int((b["profile"] == i).sum())}'
                      for i, p in enumerate(PROFILE_TYPES)))
    for i, s in enumerate(STAGES):
        m = stage == i
        if m.any():
            nc = Kc[m].norm(dim=(1, 2))
            print(f'|K_contact| {s:7s} median {nc.median():.3e}  '
                  f'p10 {nc.quantile(0.1):.3e}  p90 {nc.quantile(0.9):.3e}')
    print(f'|K_body| median {Kb.norm(dim=(1, 2)).median():.3e}')
    for lam in (0.02, 0.05, 0.1):
        K = compose_K(Kc, Kb, lam)
        ev = torch.linalg.eigvalsh(K)
        cond = ev[:, -1] / ev[:, 0]
        print(f'lambda_body={lam}: lam_min {ev[:, 0].min():.2e}  '
              f'cond median {cond.median():.1e}  max {cond.max():.1e}')
    # nearest cross-part gap per stage: the contact kernel must fire at
    # search/insert (gap << sigma_c) and stay silent at free (gap >> sigma_c)
    P, part = b['points'].double(), b['part'].bool()
    for i, s in enumerate(STAGES):
        idx = (stage == i).nonzero().flatten()[:32]
        if idx.numel() == 0:
            continue
        gaps = []
        for j in idx.tolist():
            d = torch.cdist(P[j][part[j]], P[j][~part[j]])
            gaps.append(d.min().item())
        g = torch.tensor(gaps)
        print(f'min cross gap {s:7s}: median {g.median():.4f}  '
              f'max {g.max():.4f}  (sigma_c={cfg["sigma_c"]})')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='data/peg_hole/v1')
    ap.add_argument('--n-train', type=int, default=102400)
    ap.add_argument('--n-val', type=int, default=10240)
    ap.add_argument('--n-test', type=int, default=10240)
    ap.add_argument('--n-points', type=int, default=2048)
    ap.add_argument('--shard-size', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device',
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--sigma-c', type=float, default=None)
    ap.add_argument('--shard-chunk', type=int, default=8,
                    help='라벨 계산 chunk (VRAM 조절)')
    ap.add_argument('--lambda-body', type=float, default=None)
    ap.add_argument('--inspect', action='store_true',
                    help='print calibration stats, write nothing')
    ap.add_argument('--inspect-n', type=int, default=96)
    args = ap.parse_args()

    over = {}
    if args.sigma_c is not None:
        over['sigma_c'] = args.sigma_c
    if args.lambda_body is not None:
        over['lambda_body'] = args.lambda_body
    cfg = make_cfg(**over)
    if args.inspect:
        inspect(args, cfg)
    else:
        generate(args, cfg)


if __name__ == '__main__':
    main()

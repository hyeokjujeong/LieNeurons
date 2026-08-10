"""[CURRENT TOOL] Sanity/report figures for the peg-and-hole dataset.

  fig 1  peghole_scenes.png   example scenes (canonical frame, same generator
                              distribution as the dataset) -- rows: stage,
                              columns: assorted profiles
  fig 2  peghole_labels.png   label structure measured on the STORED val
                              split: composed-K eigenspectra per stage,
                              active contact rank, |K_contact| vs depth

Usage:  python experiment/pc_se3_congruence/visualize_peg_hole.py \
            [--root data/peg_hole/v1] [--out experiment/pc_se3_congruence/figs]
"""
import argparse
import os
import sys

sys.path.append('.')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_loader.peg_hole_data_loader import load_peg_hole_split
from experiment.pc_se3_congruence.peg_hole_synth import (STAGES,
                                                         PROFILE_TYPES,
                                                         generate_batch,
                                                         make_cfg)

# dataviz reference palette (validated, first three slots / all-pairs safe)
C_PLATE, C_PEG = '#2a78d6', '#eb6834'
STAGE_C = {'free': '#2a78d6', 'search': '#eb6834', 'insert': '#1baf7a'}
SURFACE, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#d8d7d2')
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color='#ececea', linewidth=0.6)
    ax.set_axisbelow(True)


def fig_scenes(out):
    cfg = make_cfg(trans_scale=0.0)
    b = generate_batch(160, 2048, seed=202, cfg=cfg, device='cpu',
                       return_canonical=True)
    stage, prof = b['stage'], b['profile']
    fig = plt.figure(figsize=(11, 10.5), facecolor=SURFACE)
    picks = []
    for si in range(3):                       # rows: free / search / insert
        idx = (stage == si).nonzero().flatten().tolist()
        seen, row = set(), []
        for i in idx:                         # prefer distinct profiles
            if int(prof[i]) not in seen:
                seen.add(int(prof[i]))
                row.append(i)
            if len(row) == 3:
                break
        picks.append(row)
    for r, row in enumerate(picks):
        for c, i in enumerate(row):
            ax = fig.add_subplot(3, 3, 3 * r + c + 1, projection='3d')
            # canonical (upright) frame for readability; stored clouds are
            # additionally rotated by a random SE(3), which reads poorly
            P = b['canonical'][i].numpy() + \
                b['noise'][i].item() * np.random.default_rng(i).standard_normal(
                    b['canonical'][i].shape)
            peg = b['part'][i].numpy().astype(bool)
            ax.scatter(*P[~peg].T, s=1.2, c=C_PLATE, alpha=0.45,
                       linewidths=0, rasterized=True)
            ax.scatter(*P[peg].T, s=1.6, c=C_PEG, alpha=0.8,
                       linewidths=0, rasterized=True)
            spec = b['specs'][i]
            info = {'free': f'gap>{0.33}', 'search': 'tip on surface',
                    'insert': f'depth {spec["depth"]:.2f}'}[spec['stage']]
            ax.set_title(f'{spec["stage"]} · {spec["kind"]} · '
                         f'clr {spec["clearance"]:.3f} · {info}',
                         fontsize=8.5, color=INK, pad=0)
            lim = 1.4
            ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim)
            ax.set_zlim(-0.8, 2.0)
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=16, azim=30 + 25 * c)
            ax.set_axis_off()
            ax.set_facecolor(SURFACE)
    fig.suptitle('peg-and-hole scenes -- surface PCD, N=2048 '
                 '(blue plate / orange peg)', color=INK, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path = os.path.join(out, 'peghole_scenes.png')
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    print('wrote', path)


def fig_labels(root, out):
    P, K, info = load_peg_hole_split(root, 'val', extras=True)
    stage = info['stage'].long()
    lam = torch.linalg.eigvalsh(K)                          # [S, 6] ascending
    lam_c = torch.linalg.eigvalsh(info['K_contact'])
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), facecolor=SURFACE)

    ax = axes[0]
    style_ax(ax)
    x = np.arange(1, 7)
    for si, sname in enumerate(STAGES):
        m = stage == si
        if not m.any():
            continue
        med = lam[m].median(dim=0).values.numpy()[::-1]
        q1 = lam[m].quantile(0.25, dim=0).numpy()[::-1]
        q3 = lam[m].quantile(0.75, dim=0).numpy()[::-1]
        ax.fill_between(x, q1, q3, color=STAGE_C[sname], alpha=0.15,
                        linewidth=0)
        ax.plot(x, med, color=STAGE_C[sname], linewidth=2, marker='o',
                markersize=4, label=sname)
        ax.annotate(sname, (x[-1], med[-1]), textcoords='offset points',
                    xytext=(6, 0), fontsize=8, color=INK2)
    ax.set_yscale('log')
    ax.set_xlabel('eigenvalue index (desc)', color=INK2, fontsize=9)
    ax.set_title('spectrum of K = K_contact + λ·K_body', color=INK,
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

    ax = axes[1]
    style_ax(ax)
    thresh = 1e-3 * lam_c[:, -1].clamp_min(1e-30)
    arank = (lam_c > thresh[:, None]).sum(-1)
    arank[lam_c[:, -1] < 1e-12] = 0
    width = 0.27
    for si, sname in enumerate(STAGES):
        m = stage == si
        cnt = torch.bincount(arank[m], minlength=7).double()
        cnt = cnt / cnt.sum().clamp_min(1)
        ax.bar(np.arange(7) + (si - 1) * width, cnt.numpy(), width * 0.93,
               color=STAGE_C[sname], label=sname)
    ax.set_xlabel('active rank of K_contact (λ > 1e-3·λ_max)', color=INK2,
                  fontsize=9)
    ax.set_title('contact constraint structure by stage', color=INK,
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

    ax = axes[2]
    style_ax(ax)
    nc = info['K_contact'].norm(dim=(1, 2))
    ins = stage == 2
    depth_frac = info['depth'][ins] / np.minimum(
        0.9 * info['H'][ins].numpy(), 1.15 * info['T'][ins].numpy())
    ax.scatter(depth_frac, nc[ins].clamp_min(1e-8).numpy(), s=5,
               c=STAGE_C['insert'], alpha=0.35, linewidths=0)
    ax.set_yscale('log')
    ax.set_xlabel('insertion depth fraction', color=INK2, fontsize=9)
    ax.set_title('|K_contact| vs insertion depth', color=INK, fontsize=10)

    for a in axes:
        a.set_facecolor(SURFACE)
    fig.tight_layout()
    path = os.path.join(out, 'peghole_labels.png')
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    n = P.shape[0]
    print(f'wrote {path}  (val split, {n} scenes)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/peg_hole/v1')
    ap.add_argument('--out', default='experiment/pc_se3_congruence/figs')
    ap.add_argument('--scenes-only', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_scenes(args.out)
    if not args.scenes_only:
        fig_labels(args.root, args.out)


if __name__ == '__main__':
    main()

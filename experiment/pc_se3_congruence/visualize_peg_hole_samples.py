"""[CURRENT TOOL]
Per-sample view of the peg-and-hole dataset: the cloud and ITS stiffness.

One row per scene.  Left: the point cloud the model actually receives (peg /
plate coloured, at the training resolution).  Middle: the 6x6 target K as a
signed heatmap with the ff / fm / mm blocks marked.  Right: the eigenvalue
spectrum on a log axis, which is what the AIRM loss actually compares.

K is displayed normalised by its own largest magnitude because ||K|| spans
~5000x across contact stages; the true norm is printed in the panel title.

Usage:
  python experiment/pc_se3_congruence/visualize_peg_hole_samples.py \
      [--root data/peg_hole/v1] [--n-points 1024] [--out .../figs]
"""
import argparse
import os
import sys

sys.path.append('.')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = ['Noto Sans CJK KR', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

from data_loader.peg_hole_data_loader import load_peg_hole_split
from experiment.pc_se3_congruence.peg_hole_synth import PROFILE_TYPES, STAGES

# dataviz reference palette
C_PLATE, C_PEG = '#2a78d6', '#eb6834'
DIV_LO, DIV_HI = '#2a78d6', '#eb6834'          # cool / warm poles
INK, INK2, SURFACE, GRID = '#0b0b0b', '#52514e', '#fcfcfb', '#ececea'
BAR = '#1baf7a'


def diverging_cmap():
    """Two-hue diverging ramp with a NEUTRAL GRAY midpoint (never a hue)."""
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        'kdiv', [DIV_LO, '#8fb4e2', '#efeeea', '#f2b193', DIV_HI])


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#d8d7d2')
    ax.tick_params(colors=INK2, labelsize=8)


def upright(pts, peg):
    """Rotate FOR DISPLAY ONLY so the plate lies flat and the peg points up.

    The clouds are stored at a random SE(3) pose (that is the point -- the
    model is equivariant), which makes a raw scatter unreadable.  The plate is
    a slab, so its smallest-variance direction is the surface normal; align
    that with +z and flip so the peg sits above.  Nothing downstream uses this.
    """
    plate = pts[~peg]
    c = plate.mean(0)
    # svd rows are principal directions in DECREASING variance, so vt[2] is
    # the slab normal; keeping vt as-is sends it to the z axis.
    _, _, vt = np.linalg.svd(plate - c, full_matrices=False)
    R = vt.copy()
    if np.linalg.det(R) < 0:
        R[0] = -R[0]
    q = (pts - c) @ R.T
    if q[peg][:, 2].mean() < 0:                    # peg must end up on top
        q[:, [0, 2]] = -q[:, [0, 2]]               # 180 deg about y: det = +1
    return q


def pick_scenes(info, n_per_stage=1):
    """One shallow and one deep insertion, plus one free and one search."""
    stage, depth = info['stage'].long(), info['depth']
    picks = []
    for si in (0, 1):
        idx = (stage == si).nonzero().flatten()
        picks += idx[:n_per_stage].tolist()
    ins = (stage == 2).nonzero().flatten()
    d = depth[ins]
    picks.append(int(ins[d.argsort()[len(d) // 10]]))       # shallow
    picks.append(int(ins[d.argsort()[-1]]))                 # deep
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/peg_hole/v1')
    ap.add_argument('--out', default='experiment/pc_se3_congruence/figs')
    ap.add_argument('--n-points', type=int, default=1024)
    ap.add_argument('--n-load', type=int, default=512)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    P, K, info = load_peg_hole_split(args.root, 'val', n=args.n_load,
                                     n_points=args.n_points, seed=0,
                                     extras=True, device=args.device)
    picks = pick_scenes(info)
    cmap = diverging_cmap()

    fig = plt.figure(figsize=(12.5, 3.05 * len(picks)), facecolor=SURFACE)
    gs = fig.add_gridspec(len(picks), 3, width_ratios=[1.15, 1.0, 1.25],
                          hspace=0.36, wspace=0.42)

    for r, i in enumerate(picks):
        pts = P[i].numpy()
        peg = info['part'][i].numpy().astype(bool)
        Ki = K[i].numpy()
        stage = STAGES[int(info['stage'][i])]
        prof = PROFILE_TYPES[int(info['profile'][i])]
        nrm = np.abs(Ki).max()

        # ---- cloud
        ax = fig.add_subplot(gs[r, 0], projection='3d')
        q = upright(pts, peg)
        ax.scatter(*q[~peg].T, s=1.6, c=C_PLATE, alpha=0.55,
                   linewidths=0, rasterized=True)
        ax.scatter(*q[peg].T, s=2.0, c=C_PEG, alpha=0.9,
                   linewidths=0, rasterized=True)
        lim = np.abs(q).max() * 0.60
        ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim), ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=10, azim=-72)
        ax.set_axis_off()
        ax.set_facecolor(SURFACE)
        extra = (f' · depth {info["depth"][i]:.2f}' if stage == 'insert'
                 else '')
        ax.set_title(f'{stage} · {prof} · N={pts.shape[0]}{extra}',
                     fontsize=9, color=INK, pad=-2)

        # ---- K heatmap
        ax = fig.add_subplot(gs[r, 1])
        style(ax)
        im = ax.imshow(Ki / nrm, cmap=cmap, vmin=-1, vmax=1)
        for e in (2.5,):
            ax.axhline(e, color=INK, lw=1.1)
            ax.axvline(e, color=INK, lw=1.1)
        ax.set_xticks([1, 4]), ax.set_yticks([1, 4])
        ax.set_xticklabels(['f', 'm'], fontsize=9, color=INK2)
        ax.set_yticklabels(['f', 'm'], fontsize=9, color=INK2)
        for (a, b, lab) in ((1, 1, 'ff'), (4, 1, 'fm'), (1, 4, 'fm'),
                            (4, 4, 'mm')):
            ax.text(a, b, lab, ha='center', va='center', fontsize=11,
                    color=INK, alpha=0.30, fontweight='bold')
        ax.set_title(f'K / |K|max     ‖K‖ = {np.linalg.norm(Ki):.2e}',
                     fontsize=9, color=INK)
        cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
        cb.ax.tick_params(labelsize=7, colors=INK2)
        cb.outline.set_visible(False)

        # ---- spectrum
        ax = fig.add_subplot(gs[r, 2])
        style(ax)
        ax.grid(True, axis='y', color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        lam = np.linalg.eigvalsh(Ki)[::-1]
        ax.bar(np.arange(1, 7), lam, 0.62, color=BAR)
        ax.set_yscale('log')
        ax.set_xticks(np.arange(1, 7))
        ax.tick_params(axis='y', pad=1)
        ax.set_title(f'spectrum   cond = {lam[0] / lam[-1]:.0f}',
                     fontsize=9, color=INK)
        kc = info['K_contact'][i].numpy()
        share = np.linalg.norm(kc) / max(np.linalg.norm(Ki), 1e-30)
        ax.annotate(f'contact 기여 {share:.1%}', (0.97, 0.90),
                    xycoords='axes fraction', ha='right', fontsize=8,
                    color=INK2)

    fig.suptitle('peg-and-hole: 입력 point cloud와 대응하는 target stiffness K',
                 color=INK, fontsize=12.5, y=0.995)
    path = os.path.join(args.out, 'peghole_samples.png')
    fig.savefig(path, dpi=165, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path)


if __name__ == '__main__':
    main()

"""Run the local-kernel + second-moment experiment on the full object suite.

The suite contains centro (four asymmetry levels), C2, tetrahedral, and IID
clouds (three point counts).  ``--phase both`` first verifies that the analytic
head exactly reproduces the matching target, then trains the learned radial
kernel.  Each case is recorded as a separate W&B run by blockage_bench.py.

Examples:
  # Fast smoke test, no W&B
  python experiment/pc_se3_congruence/run_tensor_kernel_suite.py \
      --quick --phase both --wandb-mode disabled

  # Recommended learned experiment
  python experiment/pc_se3_congruence/run_tensor_kernel_suite.py \
      --recipe full --phase train --wandb-mode online
"""
import argparse
import subprocess
import sys
from pathlib import Path


def command(args, weight):
    bench = Path(__file__).with_name('blockage_bench.py')
    cmd = [
        sys.executable, str(bench),
        '--suite',
        '--recipe', args.recipe,
        '--encoder', 'tensor',
        '--method', 'covector',
        '--tensor-graph', 'kernel',
        '--target-graph', args.target_graph,
        '--tensor-weight', weight,
        '--kernel-candidates', str(args.kernel_candidates),
        '--wandb-mode', args.wandb_mode,
    ]
    if args.quick:
        cmd.append('--quick')
    if args.device is not None:
        cmd.extend(['--device', args.device])
    if args.wandb_project is not None:
        cmd.extend(['--wandb-project', args.wandb_project])
    if args.wandb_entity is not None:
        cmd.extend(['--wandb-entity', args.wandb_entity])
    cmd.extend(args.extra)
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['sanity', 'train', 'both'],
                    default='both',
                    help=('sanity=analytic exact-match, train=learned radial '
                          'kernel, both=둘 다 순서대로'))
    ap.add_argument('--recipe', choices=['toy', 'full'], default='toy')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--kernel-candidates', type=int, default=32)
    ap.add_argument('--target-graph', choices=['kernel', 'knn', 'all'],
                    default='kernel',
                    help=('kernel=권장 matching smooth target; knn=기존 hard-kNN '
                          'target에 대한 비교; all=nonlocal 대조 target'))
    ap.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'],
                    default='online')
    ap.add_argument('--wandb-project')
    ap.add_argument('--wandb-entity')
    ap.add_argument('--device')
    ap.add_argument('extra', nargs=argparse.REMAINDER,
                    help='-- 뒤에 blockage_bench.py 추가 옵션 전달')
    args = ap.parse_args()
    if args.extra and args.extra[0] == '--':
        args.extra = args.extra[1:]
    if args.kernel_candidates < 2:
        ap.error('--kernel-candidates must be >= 2')
    if args.phase in ('sanity', 'both') and args.target_graph != 'kernel':
        ap.error('analytic sanity phase requires --target-graph kernel')

    weights = {
        'sanity': ['analytic'],
        'train': ['learned'],
        'both': ['analytic', 'learned'],
    }[args.phase]
    for weight in weights:
        title = ('analytic exact-match sanity' if weight == 'analytic'
                 else 'learned local radial kernel')
        print(f'\n===== tensor-kernel suite: {title} =====', flush=True)
        subprocess.run(command(args, weight), check=True)


if __name__ == '__main__':
    main()

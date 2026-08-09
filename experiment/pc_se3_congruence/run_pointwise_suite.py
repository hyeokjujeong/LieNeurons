"""Run the pointwise wrench -> stiffness experiment over the object suite.

Pipeline under test (axis convention: N points, k neighbours, C channels,
H factors, K the 6x6 stiffness):

    P -> tie-safe local graph -> learned invariant SET pooling over k
      -> pointwise LN-Linear / covector-bracket / Klein-gate blocks (N kept)
      -> late second moment over (i, h) -> K

Phases:
  verify    structural checks only, no training (verify_pointwise.py):
            equivariance, permutation invariance, exact-tie robustness against
            the rank-channel comparison arm, rank on symmetric clouds.
  teacher   REALIZABILITY: the target is a frozen randomly-initialised model of
            the same class.  Separates optimisability from expressivity -- a
            residual here cannot be blamed on the target being outside the
            model class.  Run this before reading any analytic-target number.
  analytic  the compact-kernel contact-spring target.  NOT matched to this
            model (it is an edge-level second moment of RAW wrenches, while the
            model forms a second moment of LATENT covectors), so a nonzero
            floor is expected and informative.
  ablation  the architecture ablations, on the same target as --ablation-target.

Examples:
  # 1. structure first, no training
  python experiment/pc_se3_congruence/run_pointwise_suite.py --phase verify

  # 2. can it fit its own class on every symmetric object?
  python experiment/pc_se3_congruence/run_pointwise_suite.py \
      --phase teacher --wandb-mode online

  # 3. full recipe against the analytic target
  python experiment/pc_se3_congruence/run_pointwise_suite.py \
      --phase analytic --recipe full --wandb-mode online

  # 4. what does each component buy?
  python experiment/pc_se3_congruence/run_pointwise_suite.py --phase ablation
"""
import argparse
import subprocess
import sys
from pathlib import Path

# name -> extra blockage_bench.py flags.  'default' must stay first: every other
# row differs from it in exactly one design decision.
ABLATIONS = {
    'default': [],
    'separable-encoder-bracket': ['--pw-bracket', 'separable'],
    'pairwise-encoder-bracket': ['--pw-bracket', 'pairwise',
                                 '--pw-bracket-channels', '4'],
    'no-backbone-bracket': ['--pw-no-bracket-layers'],
    'no-gate': ['--pw-gate', 'none'],
    'full-gram-gate': ['--pw-gate', 'full'],
    'no-global-context': ['--pw-no-global-context'],
    'message-passing': ['--pw-message-passing'],
    'attention-pool': ['--pw-pool', 'attention'],
    'unnormalized-basis-pool': ['--pw-pool', 'basis'],
    'sum-pool': ['--pw-pool', 'sum'],
    'mean-pool': ['--pw-pool', 'mean'],
    'knn-adaptive-radius': ['--pw-radius-mode', 'knn_adaptive'],
    'knn-shell-radius': ['--pw-radius-mode', 'knn_shell'],
    'density-scaled-radius': ['--pw-radius-mode', 'density_scaled'],
    'global-scale-radius': ['--pw-radius-mode', 'global_scale'],
    'uniform-beta': ['--pw-beta', 'uniform'],
    'force-invariant': ['--pw-force-invariant'],
    'beta-normalized': ['--pw-normalize', 'beta'],
    # The comparison arm, not a variant of this model: neighbour rank as the
    # channel index, which is the construction that fails on exact ties.
    'rank-channel-baseline': ['--encoder', 'tensor', '--tensor-graph', 'kernel',
                              '--tensor-backbone', 'covector'],
}


def bench_command(args, target_graph, extra):
    bench = Path(__file__).with_name('blockage_bench.py')
    cmd = [sys.executable, str(bench), '--suite', '--recipe', args.recipe,
           '--method', 'covector', '--target-graph', target_graph,
           '--kernel-candidates', str(args.kernel_candidates),
           '--pw-candidates', str(args.pw_candidates),
           '--pw-channels', *[str(c) for c in args.pw_channels],
           '--pw-factors', str(args.pw_factors),
           '--pw-radius-mode', args.pw_radius_mode,
           '--pw-target-k', str(args.pw_target_k),
           '--pw-pool', args.pw_pool,
           '--wandb-mode', args.wandb_mode]
    if args.pw_radius_alpha is not None:
        cmd += ['--pw-radius-alpha', str(args.pw_radius_alpha)]
    if '--encoder' not in extra:
        cmd += ['--encoder', 'pointwise']
    if args.quick:
        cmd.append('--quick')
    if args.device is not None:
        cmd += ['--device', args.device]
    if args.wandb_project is not None:
        cmd += ['--wandb-project', args.wandb_project]
    if args.wandb_entity is not None:
        cmd += ['--wandb-entity', args.wandb_entity]
    return cmd + list(extra) + list(args.extra)


def verify_command(args):
    script = Path(__file__).with_name('verify_pointwise.py')
    cmd = [sys.executable, str(script), '--full',
           '--candidates', str(args.pw_candidates),
           '--radius-mode', args.pw_radius_mode,
           '--target-k', str(args.pw_target_k)]
    if args.pw_radius_alpha is not None:
        cmd += ['--radius-alpha', str(args.pw_radius_alpha)]
    if args.device is not None:
        cmd += ['--device', args.device]
    if args.out is not None:
        cmd += ['--out', args.out]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', default='verify',
                    choices=['verify', 'teacher', 'analytic', 'ablation',
                             'all'])
    ap.add_argument('--recipe', choices=['toy', 'full'], default='toy')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--ablations', default='all',
                    help='쉼표로 구분한 ablation 이름 목록 또는 all')
    ap.add_argument('--ablation-target', default='teacher',
                    choices=['teacher', 'kernel', 'knn', 'all'],
                    help='ablation phase가 사용할 target')
    ap.add_argument('--kernel-candidates', type=int, default=32,
                    help='analytic target의 compact-kernel 후보 수')
    ap.add_argument('--pw-candidates', type=int, default=64)
    ap.add_argument('--pw-channels', type=int, nargs='+',
                    default=[8, 16, 32, 16])
    ap.add_argument('--pw-factors', type=int, default=8)
    # degree_matched는 분포·내재차원·N과 무관하게 평균 degree를 target_k로 고정한다.
    # §5.5의 published 수치는 density_scaled + alpha 1.15 + target_k 16으로 얻었고
    # (N=48/128/512에서 평균 degree 9.6/13.5/15.3), 그 조합은
    # run_pointwise_gpu_experiments.sh가 플래그로 고정하므로 재현에는 영향이 없다.
    ap.add_argument('--pw-radius-mode', default='degree_matched',
                    choices=['degree_matched', 'global_scale',
                             'density_scaled', 'fixed', 'knn_adaptive',
                             'knn_shell'])
    ap.add_argument('--pw-radius-alpha', type=float, default=None,
                    help='기본: 모드별 기본값 (degree_matched 1.0, 나머지 0.75)')
    ap.add_argument('--pw-target-k', type=int, default=16)
    ap.add_argument('--pw-pool', default='basis_mean',
                    choices=['basis', 'basis_mean', 'attention', 'sum', 'mean'])
    ap.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'],
                    default='online')
    ap.add_argument('--wandb-project')
    ap.add_argument('--wandb-entity')
    ap.add_argument('--device')
    ap.add_argument('--out', help='verify phase 결과 json 경로')
    ap.add_argument('extra', nargs=argparse.REMAINDER,
                    help='-- 뒤에 blockage_bench.py 추가 옵션 전달')
    args = ap.parse_args()
    if args.extra and args.extra[0] == '--':
        args.extra = args.extra[1:]

    if args.ablations == 'all':
        names = list(ABLATIONS)
    else:
        names = [n.strip() for n in args.ablations.split(',') if n.strip()]
        unknown = [n for n in names if n not in ABLATIONS]
        if unknown:
            ap.error(f'unknown ablation(s) {unknown}; '
                     f'available: {list(ABLATIONS)}')

    phases = (['verify', 'teacher', 'analytic', 'ablation']
              if args.phase == 'all' else [args.phase])
    for phase in phases:
        if phase == 'verify':
            print('\n===== pointwise suite: structural verification =====',
                  flush=True)
            subprocess.run(verify_command(args), check=True)
        elif phase in ('teacher', 'analytic'):
            target = 'teacher' if phase == 'teacher' else 'kernel'
            title = ('realizability (frozen teacher of the same class)'
                     if phase == 'teacher'
                     else 'analytic compact-kernel contact-spring target')
            print(f'\n===== pointwise suite: {title} =====', flush=True)
            subprocess.run(bench_command(args, target, []), check=True)
        else:
            for name in names:
                extra = ABLATIONS[name]
                target = args.ablation_target
                if '--encoder' in extra and target == 'teacher':
                    # The teacher target is a frozen model of the POINTWISE
                    # class; comparing another architecture against it would be
                    # a different experiment, so the baseline uses the analytic
                    # target instead.  Reported explicitly, never silently.
                    target = 'kernel'
                    print(f'[{name}] target teacher -> kernel '
                          '(teacher는 pointwise 클래스 전용)', flush=True)
                print(f'\n===== pointwise suite: ablation {name} '
                      f'(target={target}) =====', flush=True)
                subprocess.run(bench_command(args, target, extra), check=True)


if __name__ == '__main__':
    main()

"""[CURRENT ENTRY POINT]
Train a named experiment.  Default: ``cmp2``.

    python .../train.py --data-path mydata.npz   # 내 데이터셋 (= custom)
    python .../train.py                          # 이름 없고 경로도 없으면 cmp2
    python .../train.py cmpA --epochs 60         # 이름을 준다면 첫 인자
    python .../train.py --list                   # 정의된 실험 목록
    python .../train.py --dry-run                # 명령만 출력

``--data-path`` 를 주면 실험 이름을 생략할 수 있다 -- 내 데이터셋이라는 뜻이므로
``custom`` 으로 붙는다.

WHY THIS EXISTS.  ``blockage_bench.py`` is the shared training driver for every
experiment in this folder (five synthetic cloud families, four encoder
generations, two representations), so it exposes ~65 flags of which only about
a dozen apply to any single run.  This script names the configurations worth
running, fills in the rest, and forwards everything else, so a newcomer does not
have to work out which flags are theirs.  It runs ``blockage_bench.py`` as a
subprocess -- there is no second training loop to keep in sync.

HOW TO ADD AN EXPERIMENT.  Add one entry to ``EXPERIMENTS`` below.  A entry only
has to say what makes it different; everything it omits falls back to the shared
defaults in ``DEFAULTS`` and to the CLI flags.  Nothing else in this file needs
to change.  Any flag this script does not define is passed straight through to
``blockage_bench.py``, so an experiment never has to wait for a wrapper flag:

    python experiment/pc_se3_congruence/train.py cmp2 --pw-gate full

For the full multi-stage protocol (baseline -> teacher -> stored -> compare,
5-6 hours, two GPUs) use ``run_experiments.sh`` instead.  That script
is the source of the numbers in ``peghole_training_report.md``; this one runs a
single configuration.

Read the resulting ``val_d`` only against ``peghole_baseline.py``: on its own an
AIRM distance has no scale.  Pass that script's json via ``--baseline-json`` and
the run records ``val_d_rel`` for you.
"""
import argparse
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().with_name('blockage_bench.py')

# name -> experiment definition
#   desc      one line, shown by --list
#   dataset   blockage_bench --dataset
#   target    blockage_bench --target-graph
#   flags     extra blockage_bench flags specific to this experiment
#   overrides CLI defaults this experiment changes.  Applied ONLY where the
#             user did not pass the flag themselves, so an override never
#             silently beats something typed on the command line.
#   note      printed before the run; why this configuration exists
#   dataset   blockage_bench --dataset (합성 생성기).  --data-path 를 쓰면 생략
PEGHOLE = dict(overrides=dict(data_path='data/peg_hole/v2',
                              n_points=1024))   # 저장 2048점 -> 1024 서브샘플
EXPERIMENTS = {
    'cmp2': dict(
        **PEGHOLE,
        desc='기본. peg-hole 저장 라벨 + f_a·f_b 불변량 (compare arm B)',
        target='stored',
        flags=['--pw-force-invariant'],
        note=('gate와 beta에 공급되던 Klein 불변량이 이 데이터에서 항등적으로 0'
              '이다 — 인코더가 이웃 wrench를 모두 같은 앵커에 걸어 잠재 특징이 '
              'pitch 0 wrench가 되고, Klein form은 정확히 pitch를 재기 때문. '
              '그 결과 파라미터의 70%가 상수만 출력했다. pitch 0에서 죽지 않는 '
              'f_a·f_b로 교체한 것이 이 팔이며, 대조 실험에서 d 2.888→1.634.'),
    ),
    'cmpA': dict(
        **PEGHOLE,
        desc='대조군. cmp2에서 불변량 교체만 뺀 초기 구성',
        target='stored',
        flags=[],
        note='cmp2의 개선폭을 읽기 위한 기준. 나머지 조건은 전부 동일하다.',
    ),
    'cmpC': dict(
        **PEGHOLE,
        desc='제거군. gate와 beta를 통째로 제거 (파라미터 51% 감소)',
        target='stored',
        flags=['--pw-gate', 'none', '--pw-beta', 'uniform'],
        note=('반증 시험. cmp2의 진단이 맞다면 죽은 gate 출력은 채널별 상수이고 '
              '뒤의 LNLinear가 흡수하므로 cmpC는 cmpA와 거의 같아야 한다. '
              '달랐다면 진단이 틀린 것이다.'),
    ),
    'teacher': dict(
        **PEGHOLE,
        desc='realizability 통제군. 타깃이 같은 클래스의 고정 난수 모델',
        target='teacher',
        flags=['--pw-force-invariant'],
        note=('정답 가중치가 존재함이 보장된다. 여기서 0 근처로 내려가지 않으면 '
              '최적화 문제이고, 그러면 stored 타깃의 결과는 해석할 수 없다. '
              'stored 숫자를 읽기 전에 먼저 확인한다.'),
    ),
    'custom': dict(
        desc='내 데이터셋. --data-path 만 주면 이름 없이도 이걸로 붙는다',
        target='stored',
        flags=['--pw-force-invariant'],
        # 표본 수는 파일에 있는 만큼(0 = 전부), 해상도는 저장된 그대로.
        overrides=dict(n_train=0, n_val=0, epochs=60),
        note=('직접 준비한 (point cloud, 6x6 강성) 쌍으로 검증한다. 형식은 '
              'data_loader/pc_stiffness_data_loader.py 참조 — K 는 [m; f] '
              '순서의 SPD 여야 한다 (force-first 로 저장했다면 블록을 바꿔서 '
              '저장할 것. 안 바꿔도 실행은 되고 숫자만 조용히 틀린다). '
              '판정을 하려면 같은 --data-path 로 peghole_baseline.py 를 먼저 '
              '돌려 기준선을 뽑을 것.'),
    ),
    'selftest': dict(
        dataset='iid',
        desc='설치 확인용 스모크. 데이터셋 없이 즉시 돈다 (합성 클라우드)',
        target='knn',
        flags=['--pw-force-invariant'],
        overrides=dict(n_points=48, n_train=64, n_val=32, epochs=5, batch=16,
                       eval_batch=16, device='cpu', wandb_mode='disabled'),
        note=('합성 클라우드에 analytic contact-spring 타깃. 1분 안에 끝난다. '
              '학습 후 equivariance 가 1e-15 수준이고 rank 6, clamp 0 이면 '
              '설치와 경로가 정상이다. 정확도는 보지 않는다.'),
    ),
}
DEFAULT_EXPERIMENT = 'cmp2'


def _drop_overridden(flags, passthrough):
    """Remove entries of ``flags`` that ``passthrough`` respecifies.

    argparse would keep the last occurrence anyway, but emitting a flag twice
    makes the printed command misleading about what actually ran.  A flag is
    dropped together with the values that follow it.
    """
    overridden = {t.split('=')[0] for t in passthrough if t.startswith('--')}
    out, drop = [], False
    for tok in flags:
        if tok.startswith('--'):
            drop = tok in overridden
        if not drop:
            out.append(tok)
    return out


def apply_overrides(exp, a, rest):
    """실험이 정한 기본값을 적용한다 — 단, 사용자가 직접 준 플래그는 건드리지 않는다."""
    used = {t.split('=')[0] for t in rest if t.startswith('--')}
    applied = {}
    for key, val in exp.get('overrides', {}).items():
        if '--' + key.replace('_', '-') in used:
            continue
        setattr(a, key, val)
        applied[key] = val
    return applied


def build_argv(exp, a, passthrough):
    flags = [
        # 이 파이프라인은 전부 se(3)* 안에서 동작한다 (Q가 등장하지 않는다).
        '--encoder', 'pointwise',
        '--method', 'covector',
        '--target-graph', exp['target'],
        '--n-train', str(a.n_train),
        '--n-val', str(a.n_val),
        '--epochs', str(a.epochs),
        '--batch', str(a.batch),
        # 둘 다 [B, N, N] 거리행렬을 만드는 chunk 크기 — 최대 메모리를 지배한다.
        '--eval-batch', str(a.eval_batch),
        '--target-batch', str(a.eval_batch),
        '--lr', str(a.lr),
        '--pw-channels', *[str(c) for c in a.pw_channels],
        '--pw-factors', str(a.pw_factors),
        '--data-seed', str(a.seed),
        '--device', a.device,
        '--wandb-mode', a.wandb_mode,
    ]
    # n_points=None 이면 플래그를 아예 생략한다 — 디스크 데이터셋은 저장된
    # 해상도를 그대로 쓰고, 합성 데이터셋은 blockage_bench 의 기본값을 쓴다.
    if a.n_points is not None:
        flags += ['--n-points', str(a.n_points)]
    if a.data_path:
        flags += ['--data-path', a.data_path, '--val-frac', str(a.val_frac)]
    if 'dataset' in exp:
        flags += ['--dataset', exp['dataset']]
    flags += exp['flags']
    if a.baseline_json:
        flags += ['--baseline-json', a.baseline_json]
    if a.ckpt_out:
        flags += ['--ckpt-out', a.ckpt_out]
    if a.quick:
        flags.append('--quick')
    # passthrough 가 이긴다. argparse 도 마지막 값을 쓰지만, 같은 플래그를 두 번
    # 찍으면 출력된 명령이 실제로 돈 설정을 잘못 알려주게 된다.
    return [sys.executable, str(BENCH)] \
        + _drop_overridden(flags, passthrough) + passthrough


def pop_experiment(argv):
    """실험 이름은 주는 경우 반드시 첫 인자다.  (이름, 나머지 argv)를 돌려준다."""
    if argv and not argv[0].startswith('-'):
        return argv[0], argv[1:]
    # --data-path 를 줬다는 것 자체가 '내 데이터셋'이라는 뜻이다.  이름을 요구하지
    # 않는다.  cmp2 로 떨어뜨리면 peg-hole 기본값(1024 서브샘플, 20480/2048 표본,
    # 150 epoch)이 남의 파일에 붙어서 로더가 서브샘플을 거부한다.
    if any(t == '--data-path' or t.startswith('--data-path=') for t in argv):
        return 'custom', argv
    return DEFAULT_EXPERIMENT, argv


def print_experiments():
    print('정의된 실험 (train.py <이름>):\n')
    for name, e in EXPERIMENTS.items():
        mark = '  (기본)' if name == DEFAULT_EXPERIMENT else ''
        print(f'  {name}{mark}\n    {e["desc"]}')
        print(f'    --target-graph {e["target"]}'
              + (f' {" ".join(e["flags"])}' if e['flags'] else ''))
        print()


def main():
    ap = argparse.ArgumentParser(
        description=('pc_se3_congruence 학습 진입점. 인자 없이 돌리면 '
                     f'{DEFAULT_EXPERIMENT} 를 기본 설정으로 실행한다.'),
        epilog=('여기서 정의하지 않은 플래그는 blockage_bench.py 로 그대로 '
                '전달된다:  train.py cmp2 --pw-gate full'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 실험 이름은 argparse 이전에 sys.argv 에서 직접 떼어 낸다.  nargs='?' 위치
    # 인자로 두면 parse_known_args 가 모르는 플래그의 VALUE 를 이름으로 삼켜서
    # (`--dataset iid` -> experiment='iid') 조용히 잘못 동작한다.
    ap.add_argument('--list', action='store_true',
                    help='정의된 실험과 각각의 플래그를 출력하고 끝낸다')

    d = ap.add_argument_group('데이터')
    d.add_argument('--data-path',
                   help=('디스크 데이터셋 경로 — 하나로 통일되어 있다. '
                         'meta.json 이 있으면 peg-and-hole 샤드 데이터셋, '
                         '아니면 points [S,N,3] 와 K [S,6,6] 이 든 .npz/.pt '
                         '파일(또는 train/val 이 든 디렉터리). 형식과 K 의 '
                         '블록 순서는 data_loader/pc_stiffness_data_loader.py'))
    d.add_argument('--val-frac', type=float, default=0.1,
                   help='경로가 파일 하나일 때 val 로 뗄 비율')
    d.add_argument('--n-points', type=int, default=None,
                   help=('cloud 해상도. peg-hole 은 1024 로 서브샘플하고 라벨을 '
                         '그 점들에서 재계산한다. 라벨을 재계산할 수 없는 '
                         '형식은 저장된 해상도를 그대로 쓴다'))
    d.add_argument('--n-train', type=int, default=20480,
                   help='0 이하면 있는 만큼 전부 (custom 기본)')
    d.add_argument('--n-val', type=int, default=2048,
                   help='0 이하면 있는 만큼 전부 (custom 기본)')

    m = ap.add_argument_group('모델')
    m.add_argument('--pw-channels', type=int, nargs='+',
                   default=[16, 32, 64, 32],
                   help='[C_0(set pooling), C_1, ...]')
    m.add_argument('--pw-factors', type=int, default=16, help='factor 채널 H')

    t = ap.add_argument_group('학습')
    t.add_argument('--epochs', type=int, default=150)
    t.add_argument('--batch', type=int, default=32)
    t.add_argument('--eval-batch', type=int, default=64,
                   help=('검증/타깃 생성 chunk. 그래프가 [B,N,N] 거리행렬을 '
                         '만들므로 최대 GPU 메모리를 이 값이 지배한다 '
                         '(N=1024, 64 -> 537 MB). 숫자 자체는 불변'))
    t.add_argument('--lr', type=float, default=1e-3)
    t.add_argument('--seed', type=int, default=100)
    t.add_argument('--device', default='cuda:0')

    o = ap.add_argument_group('출력')
    o.add_argument('--baseline-json',
                   help=('peghole_baseline.py의 출력. 주면 val_d_rel(기준선 '
                         '대비 비율)이 기록된다. 판정에 필요하다'))
    o.add_argument('--ckpt-out', help='val_d 최저점 state_dict 저장 경로')
    o.add_argument('--wandb-mode', default='online',
                   choices=['online', 'offline', 'disabled'])
    o.add_argument('--quick', action='store_true',
                   help='smoke 설정 (10 epoch, 소량 표본)')
    o.add_argument('--dry-run', action='store_true',
                   help='실행 대신 blockage_bench.py 명령을 출력만 한다')

    # 모르는 플래그는 blockage_bench.py 로 그대로 넘긴다 (잘못된 이름은 거기서
    # 거부되므로 오타가 조용히 통과하지는 않는다).
    name, rest = pop_experiment(sys.argv[1:])
    a, passthrough = ap.parse_known_args(rest)
    if a.list:
        print_experiments()
        return 0
    if name not in EXPERIMENTS:
        ap.error(f"'{name}' 은 정의된 실험이 아니다 "
                 f"(가능: {', '.join(EXPERIMENTS)}).  실험 이름을 준다면 반드시 "
                 f"첫 인자여야 한다")
    if name == 'custom' and not a.data_path:
        # 없으면 blockage_bench 가 --dataset 기본값(합성 centro)으로 조용히 돈다.
        ap.error('custom 은 --data-path 가 필요하다')

    exp = EXPERIMENTS[name]
    applied = apply_overrides(exp, a, rest)
    argv = build_argv(exp, a, passthrough)

    print(f'===== {name}: {exp["desc"]} =====')
    print(exp['note'])
    if applied:
        print('실험 기본값: '
              + '  '.join(f'{k}={v}' for k, v in applied.items()))
    if passthrough:
        print(f'전달된 추가 플래그: {" ".join(passthrough)}')
    # 한 줄에 플래그 하나와 그 값 전부를 붙여서 복사-붙여넣기 가능하게 출력한다.
    chunks = []
    for tok in argv[1:]:
        if tok.startswith('--') or not chunks:
            chunks.append([tok])
        else:
            chunks[-1].append(tok)
    printable = ' \\\n    '.join(' '.join(c) for c in chunks)
    print(f'\n{sys.executable} {printable}\n', flush=True)
    if a.dry_run:
        return 0
    return subprocess.call(argv)


if __name__ == '__main__':
    raise SystemExit(main())

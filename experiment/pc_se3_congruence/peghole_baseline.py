"""Phase 0 for the peg-and-hole experiment: the two numbers every trained
AIRM distance has to be read against.

Neither is a model.  Reporting ``val_d`` on its own says nothing — the AIRM
distance has no absolute scale, so a run is only interpretable between these
two brackets:

  UPPER  Frechet (Karcher) mean baseline.  The best possible CONSTANT
         predictor under the same metric the model is trained on.  A model
         that does not beat it has learned nothing about the input.
         Fitted on train, evaluated on val (the honest held-out number); the
         val-fitted mean is also reported as the oracle constant, i.e. the
         floor no constant predictor can go below.

  LOWER  Monte-Carlo label noise.  The label is an MC estimate of a surface
         integral, so two different subsamples of the SAME scene carry
         different labels.  d(K_seedA, K_seedB) on identical scenes is the
         resolution of the target itself; going below it means the run has
         memorised one particular draw rather than the physics.

Both are reported overall and per stage (free / search / insert), because the
stages differ by orders of magnitude in contact content.

Read distances through ``exp(d / sqrt(6))``: the per-direction factor by which
the prediction is off, since d is a Frobenius norm over 6 log-eigenvalues.

  python experiment/pc_se3_congruence/peghole_baseline.py \
      --peghole-root data/peg_hole/v2 --n-points 1024 \
      --n-train 20480 --n-val 2048 --device cuda:0
"""
import argparse
import json
import math
import sys

sys.path.append('.')

import torch

torch.set_default_dtype(torch.float64)

from data_loader.peg_hole_data_loader import load_peg_hole_split, read_meta
from experiment.pc_se3_congruence.peg_hole_synth import STAGES
from experiment.pc_se3_congruence.train import EIG_CLAMP, affine_invariant_d


# ------------------------------------------------------------ SPD utilities
def _eig_map(A, fn):
    lam, U = torch.linalg.eigh(0.5 * (A + A.transpose(-1, -2)))
    return U @ torch.diag_embed(fn(lam)) @ U.transpose(-1, -2)


def spd_log(A):
    return _eig_map(A, lambda l: l.clamp_min(EIG_CLAMP).log())


def spd_exp(A):
    return _eig_map(A, torch.exp)


def spd_pow(A, p):
    return _eig_map(A, lambda l: l.clamp_min(EIG_CLAMP).pow(p))


def _tangent_mean(M, K, chunk):
    """mean log(M^{-1/2} K_i M^{-1/2}) — the Riemannian gradient direction."""
    Mih = spd_pow(M, -0.5)
    S = torch.zeros_like(M)
    for i in range(0, K.shape[0], chunk):
        S = S + spd_log(Mih @ K[i:i + chunk] @ Mih).sum(0)
    return S / K.shape[0]


def karcher_mean(K, iters=200, tol=1e-12, step=0.5, chunk=8192):
    """Frechet mean of SPD matrices under the affine-invariant metric.

    Damped gradient step M <- M^{1/2} exp(t*S) M^{1/2} from the log-Euclidean
    mean, S the tangent mean.  The undamped (t=1) fixed-point iteration is the
    textbook form but DIVERGES on these labels: measured here it descends to
    ||S|| ~ 1e-2 and then blows back up past 1e0, because the spread is large
    (the labels sit AIRM ~5 apart) and the full step overshoots.  t=0.5 with
    backtracking reaches 1e-14 in ~50 iterations.

    ||S|| is the gradient of the Frechet functional, so it is returned as the
    convergence witness rather than assumed small.
    """
    M = spd_exp(spd_log(K).mean(0))
    S = _tangent_mean(M, K, chunk)
    grad, t = S.norm().item(), step
    for it in range(iters):
        if grad < tol:
            break
        Mh = spd_pow(M, 0.5)
        M_new = Mh @ spd_exp(t * S) @ Mh
        M_new = 0.5 * (M_new + M_new.transpose(-1, -2))
        S_new = _tangent_mean(M_new, K, chunk)
        g_new = S_new.norm().item()
        if g_new > grad:          # overshoot: shrink and retry from the same M
            t *= 0.5
            if t < 1e-6:
                break
            continue
        M, S, grad = M_new, S_new, g_new
    return M, grad, it + 1


def airm(K_gt, K_other):
    """Per-sample AIRM distance; K_other broadcasts against K_gt."""
    L = torch.linalg.cholesky(K_gt)
    d, n_clamped = affine_invariant_d(L, K_other.expand_as(K_gt))
    return d, n_clamped


def summarize(d, stage=None):
    """mean / median / p90 overall and per stage, plus the per-direction factor
    exp(d / sqrt(6)) that makes the numbers physically readable."""
    def one(x):
        return {'mean': x.mean().item(), 'median': x.median().item(),
                'p90': x.quantile(0.9).item(), 'n': int(x.numel()),
                'factor_mean': math.exp(x.mean().item() / math.sqrt(6))}
    out = {'all': one(d)}
    if stage is not None:
        for si, name in enumerate(STAGES):
            m = stage == si
            if m.any():
                out[name] = one(d[m])
    return out


def _fmt(tag, s):
    head = s['all']
    line = (f'  {tag:28s} d {head["mean"]:7.3f}  (median {head["median"]:7.3f}'
            f'  p90 {head["p90"]:7.3f})  factor x{head["factor_mean"]:.2f}')
    per = '  '.join(f'{n} {s[n]["mean"]:6.3f}' for n in STAGES if n in s)
    return line + (f'\n  {"":28s} per-stage: {per}' if per else '')


# ------------------------------------------------------------------- phases
def frechet_baseline(args, dev):
    """Best constant predictor under AIRM: fit on train, evaluate on val."""
    kw = dict(seed=args.data_seed, lambda_body=args.lambda_body,
              relabel=not args.no_relabel, device=args.device)
    _, K_tr = load_peg_hole_split(args.peghole_root, 'train', n=args.n_train,
                                  n_points=args.n_points, **kw)
    _, K_va, info = load_peg_hole_split(args.peghole_root, 'val', n=args.n_val,
                                        n_points=args.n_points, extras=True,
                                        **kw)
    K_tr, K_va = K_tr.to(dev), K_va.to(dev)
    stage = info['stage'].to(dev)

    M_tr, g_tr, it_tr = karcher_mean(K_tr)
    M_va, g_va, it_va = karcher_mean(K_va)
    d_held, n_cl = airm(K_va, M_tr.unsqueeze(0))
    d_oracle, _ = airm(K_va, M_va.unsqueeze(0))
    # The trivial constant, for scale: how much of the baseline is just "K is
    # not the identity" rather than "the mean is a good constant".
    d_ident, _ = airm(K_va, torch.eye(6, dtype=K_va.dtype,
                                      device=dev).unsqueeze(0))

    res = {
        'frechet_train_fit_on_val': summarize(d_held, stage),
        'frechet_val_oracle': summarize(d_oracle, stage),
        'identity_predictor': summarize(d_ident, stage),
        'karcher_grad_train': g_tr, 'karcher_iters_train': it_tr,
        'karcher_grad_val': g_va, 'karcher_iters_val': it_va,
        'clamped_val': n_cl,
        'K_train_mean_eigs': torch.linalg.eigvalsh(M_tr).tolist(),
    }
    print('\n[baseline] constant predictors on val '
          f'(n={K_va.shape[0]}, N={args.n_points})')
    print(_fmt('Frechet mean (train-fit)', res['frechet_train_fit_on_val']))
    print(_fmt('Frechet mean (val oracle)', res['frechet_val_oracle']))
    print(_fmt('identity', res['identity_predictor']))
    print(f'  Karcher gradient: train {g_tr:.2e} ({it_tr} it)  '
          f'val {g_va:.2e} ({it_va} it)   clamped {n_cl}')
    if max(g_tr, g_va) > 1e-8:
        print('  [경고] Karcher 평균이 수렴하지 않았다 — 기준선은 진짜 Frechet '
              '평균이 아니라 그 상계다 (실제 기준선은 이보다 낮다)')
    return res


def label_mc_noise(args, dev):
    """Resolution of the target itself: the same scenes, two different
    subsample draws.  Any val_d below this is fitting one draw, not physics."""
    if args.n_points is None or args.no_relabel:
        print('\n[mc-noise] --n-points 없이/relabel 없이는 두 번째 뽑기를 '
              '정의할 수 없어 건너뜁니다')
        return None
    n = min(args.n_val, args.mc_n)
    common = dict(n=n, n_points=args.n_points, lambda_body=args.lambda_body,
                  relabel=True, device=args.device)
    _, K_a, info = load_peg_hole_split(args.peghole_root, 'val',
                                       seed=args.data_seed, extras=True,
                                       **common)
    _, K_b = load_peg_hole_split(args.peghole_root, 'val',
                                 seed=args.data_seed + args.mc_seed_offset,
                                 **common)
    K_a, K_b = K_a.to(dev), K_b.to(dev)
    stage = info['stage'].to(dev)
    d, n_cl = airm(K_a, K_b)
    rel_norm = ((K_b.flatten(1).norm(dim=1) - K_a.flatten(1).norm(dim=1)).abs()
                / K_a.flatten(1).norm(dim=1))
    res = {'mc_noise': summarize(d, stage),
           'norm_rel_spread_mean': rel_norm.mean().item(),
           'seed_pair': [args.data_seed, args.data_seed + args.mc_seed_offset],
           'clamped': n_cl}
    print(f'\n[mc-noise] 같은 장면, 다른 서브샘플 (n={n}, N={args.n_points}, '
          f'seed {res["seed_pair"][0]} vs {res["seed_pair"][1]})')
    print(_fmt('label draw A vs draw B', res['mc_noise']))
    print(f'  ||K|| 뽑기 간 상대 변동: {res["norm_rel_spread_mean"]:.3f}')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--peghole-root', default='data/peg_hole/v2')
    ap.add_argument('--n-points', type=int, default=1024)
    ap.add_argument('--n-train', type=int, default=20480)
    ap.add_argument('--n-val', type=int, default=2048)
    ap.add_argument('--lambda-body', type=float, default=None)
    ap.add_argument('--data-seed', type=int, default=100)
    ap.add_argument('--no-relabel', action='store_true',
                    help='서브샘플 재라벨 없이 저장 라벨 사용 (ill-posed 대조군)')
    ap.add_argument('--mc-n', type=int, default=512,
                    help='MC 분산 추정에 쓸 val 장면 수 (두 번째 뽑기는 새로 '
                         '재라벨하므로 캐시가 없으면 비싸다)')
    ap.add_argument('--mc-seed-offset', type=int, default=1)
    ap.add_argument('--skip-mc', action='store_true')
    ap.add_argument('--device',
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out', default=None, help='결과 json 경로')
    args = ap.parse_args()

    dev = torch.device(args.device)
    meta = read_meta(args.peghole_root)
    lam = (meta['cfg']['lambda_body'] if args.lambda_body is None
           else args.lambda_body)
    print(f'[dataset] {args.peghole_root}  version {meta.get("version")}  '
          f'stored N {meta["n_points"]} -> {args.n_points}  lambda {lam}')

    res = {'args': vars(args), 'dataset_version': meta.get('version'),
           'lambda_body': lam}
    res.update(frechet_baseline(args, dev))
    if not args.skip_mc:
        mc = label_mc_noise(args, dev)
        if mc:
            res.update(mc)

    lo = res.get('mc_noise', {}).get('all', {}).get('mean')
    hi = res['frechet_train_fit_on_val']['all']['mean']
    print('\n[해석 구간] 학습된 val_d는 이 사이에 있어야 한다:')
    print(f'  기준선(넘어야 함)  d = {hi:.3f}')
    print('  MC 라벨 잡음(밑돌면 뽑기를 외운 것)  '
          + (f'd = {lo:.3f}' if lo is not None else '(생략됨)'))

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(res, f, indent=2)
        print(f'\n[saved] {args.out}')


if __name__ == '__main__':
    main()

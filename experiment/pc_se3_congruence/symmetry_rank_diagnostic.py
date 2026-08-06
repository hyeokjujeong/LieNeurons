"""Fast structural diagnostic for symmetry, kNN ties, and tensor pooling.

This script does not train.  It compares four randomly initialized/fixed
pipelines on the same exact-symmetry cloud:

  global-plueck   rank-ordered kNN Pluecker lift + global vector mean
  global-learned  VN-DGCNN direction field + global vector mean
  tensor-knn      kNN edge wrenches + second-moment pooling
  tensor-all      all edge wrenches + second-moment pooling (tie robust)

Read ``equiv_err`` together with ``rank``.  A full rank created by arbitrary
kNN tie breaking is not a valid success if equivariance is O(1).
"""
import argparse
import sys

sys.path.append('.')

import torch

torch.set_default_dtype(torch.float64)

from experiment.pc_se3_congruence.data_synth import (c2_clouds, sample_clouds,
                                                     symmetric_clouds,
                                                     tetra_orbit_clouds)
from experiment.pc_se3_congruence.encoders import (WrenchEdgeEncoder,
                                                   WrenchLearnableLiftEncoder,
                                                   WrenchPlueckerEncoder,
                                                   knn_indices)
from experiment.pc_se3_congruence.models import (ModelPC2K,
                                                   WrenchSecondMomentModel)
from experiment.pc_se3_congruence.se3_utils import (coadjoint, random_SE3,
                                                    scaled_err,
                                                    transform_cloud)


DATASETS = {
    'centro': symmetric_clouds,
    'c2': c2_clouds,
    'tetra': tetra_orbit_clouds,
    'iid': lambda n, npts, gen, **kw: sample_clouds(
        n, npts, gen, trans_scale=0.0),
}


def output_K(model, P):
    out = model(P)
    return out[1] if isinstance(out, tuple) else out


def build_models(k, hidden, seed):
    channels = (k, max(2 * k, 16), max(k, 8))

    torch.manual_seed(seed)
    global_plueck = ModelPC2K(WrenchPlueckerEncoder(k=k), channels=channels)

    torch.manual_seed(seed)
    learned_encoder = WrenchLearnableLiftEncoder(
        out_channels=k, hidden=hidden, k=k)
    global_learned = ModelPC2K(learned_encoder, channels=channels)

    tensor_knn = WrenchSecondMomentModel(
        WrenchEdgeEncoder(k=k, graph='knn'), weight_mode='uniform')
    tensor_all = WrenchSecondMomentModel(
        WrenchEdgeEncoder(k=k, graph='all'), weight_mode='uniform')
    return {
        'global-plueck': global_plueck,
        'global-learned': global_learned,
        'tensor-knn': tensor_knn,
        'tensor-all': tensor_all,
    }


def rank_stats(K):
    lam = torch.linalg.eigvalsh(K)
    scale = lam[:, -1:].clamp_min(torch.finfo(K.dtype).tiny)
    # Relative tolerance alone calls round-off in an almost-zero matrix
    # "full rank".  The absolute guard makes global-vector collapse visible.
    tol = torch.maximum(scale * 1e-9, torch.full_like(scale, 1e-12))
    rank = (lam > tol).sum(dim=1)
    return rank.double().mean().item(), lam[:, 0].median().item()


def diagnose(model, P, gen, n_transforms):
    model.eval()
    with torch.no_grad():
        K0 = output_K(model, P)
        equiv_err = 0.0
        equiv_abs = 0.0
        for _ in range(n_transforms):
            R, p = random_SE3(1.0, gen)
            R, p = R.to(P.device), p.to(P.device)
            Kt = output_K(model, transform_cloud(P, R, p))
            rho = coadjoint(R, p)
            expected = rho @ K0 @ rho.transpose(-1, -2)
            equiv_err = max(equiv_err, scaled_err(Kt, expected))
            equiv_abs = max(equiv_abs, (Kt - expected).norm().item())

        perm = torch.randperm(P.shape[1], generator=gen, device='cpu').to(P.device)
        Kp = output_K(model, P[:, perm])
        perm_err = scaled_err(Kp, K0)
        perm_abs = (Kp - K0).norm().item()
        rank, lam_min = rank_stats(K0)
    return {
        'equiv_err': equiv_err,
        'equiv_abs': equiv_abs,
        'perm_err': perm_err,
        'perm_abs': perm_abs,
        'rank': rank,
        'lam_min': lam_min,
        'K_norm': K0.norm(dim=(1, 2)).median().item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='tetra', choices=list(DATASETS))
    ap.add_argument('--eta', type=float, default=0.0)
    ap.add_argument('--n-samples', type=int, default=16)
    ap.add_argument('--n-points', type=int, default=48)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--hidden', type=int, default=8)
    ap.add_argument('--n-transforms', type=int, default=5)
    ap.add_argument('--data-seed', type=int, default=100)
    ap.add_argument('--model-seed', type=int, default=0)
    ap.add_argument('--device',
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    if args.dataset == 'tetra' and args.n_points % 12 != 0:
        args.n_points = max(12, args.n_points // 12 * 12)
        print(f'[tetra] n_points를 12의 배수 {args.n_points}로 보정')
    if args.k >= args.n_points:
        raise ValueError('--k must be smaller than --n-points')

    gen_data = torch.Generator().manual_seed(args.data_seed)
    make = DATASETS[args.dataset]
    P = (make(args.n_samples, args.n_points, gen_data, eta=args.eta)
         if args.dataset != 'iid'
         else make(args.n_samples, args.n_points, gen_data))
    P = P.to(args.device)

    # Same point labels before/after the rigid transform: changed entries
    # therefore expose floating tie-breaking in topk directly.
    gen_knn = torch.Generator().manual_seed(args.data_seed + 17)
    R, p = random_SE3(1.0, gen_knn)
    R, p = R.to(P.device), p.to(P.device)
    idx0 = knn_indices(P, args.k)
    idx1 = knn_indices(transform_cloud(P, R, p), args.k)
    changed = (idx0 != idx1).double().mean().item()

    models = build_models(args.k, args.hidden, args.model_seed)
    gen_eval = torch.Generator().manual_seed(args.data_seed + 23)
    results = {}
    for name, model in models.items():
        results[name] = diagnose(
            model.to(args.device), P, gen_eval, args.n_transforms)

    print(f'dataset={args.dataset} eta={args.eta} N={args.n_points} k={args.k}')
    print(f'rotation 후 kNN rank entry 변경률: {changed:.3e}')
    print('model              equiv_rel    equiv_abs    perm_rel     rank   lambda_min    ||K||')
    for name, r in results.items():
        print(f'{name:18s} {r["equiv_err"]:10.3e}  {r["equiv_abs"]:10.3e}  '
              f'{r["perm_err"]:10.3e}  {r["rank"]:7.2f}  '
              f'{r["lam_min"]:11.3e}  {r["K_norm"]:9.3e}')
    print('\n판정: rank=6만으로 성공이 아닙니다. equiv_err와 perm_err도 함께 '
          '수치 정밀도 수준이어야 합니다.')


if __name__ == '__main__':
    main()

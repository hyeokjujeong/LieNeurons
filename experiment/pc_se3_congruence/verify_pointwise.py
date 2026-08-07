"""Structural verification of the pointwise wrench -> stiffness pipeline.

Runs the checks that must hold BEFORE any training number is meaningful, and
prints the rank-channel comparison arm next to the new model so the two failure
modes stay separated:

  A. coadjoint equivariance   K(T.P) = Ad_T^{-T} K(P) Ad_T^{-1}, ||p|| = O(1)
  B. point permutation        K(P pi) = K(P)
  C. tie robustness           B on a cubic lattice, where "the c-th nearest
                              neighbour" is arbitrary.  The old rank-channel
                              backbone reads neighbour rank as a channel index
                              and fails here; the set-pooling encoder does not.
  D. rank preservation        rank K = 6 and lambda_min >> 0 on centro / C2 /
                              tetra clouds, where global vector pooling forces
                              every channel into Fix_H(rho) and collapses.
  E. graph health             candidate truncation fraction and degree -- the
                              only remaining way for top-k tie breaking to
                              reach the output.
  F. near-tie curve           B as a function of lattice jitter.

Run:
  python experiment/pc_se3_congruence/verify_pointwise.py
  python experiment/pc_se3_congruence/verify_pointwise.py --device cuda --full
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append('.')

import torch

torch.set_default_dtype(torch.float64)

from experiment.pc_se3_congruence.data_synth import (c2_clouds, lattice_clouds,
                                                     sample_clouds,
                                                     symmetric_clouds,
                                                     tetra_orbit_clouds)
from experiment.pc_se3_congruence.encoders import WrenchEdgeEncoder
from experiment.pc_se3_congruence.models import WrenchSecondMomentModel
from experiment.pc_se3_congruence.pointwise_models import \
    PointwiseStiffnessModel
from experiment.pc_se3_congruence.se3_utils import (coadjoint, random_SE3,
                                                    scaled_err,
                                                    transform_cloud)


def equivariance_err(model, P, gen, n_T=3, trans_scale=1.0):
    with torch.no_grad():
        K0 = model(P)
        err = 0.0
        for _ in range(n_T):
            R, p = random_SE3(trans_scale, gen)
            R, p = R.to(P.device), p.to(P.device)
            Kt = model(transform_cloud(P, R, p))
            rho = coadjoint(R, p).to(P.device)
            err = max(err, scaled_err(Kt, rho @ K0 @ rho.transpose(-1, -2)))
    return err


def permutation_err(model, P, gen, n_perm=3):
    with torch.no_grad():
        K0 = model(P)
        err = 0.0
        for _ in range(n_perm):
            perm = torch.randperm(P.shape[1], generator=gen).to(P.device)
            err = max(err, scaled_err(model(P[:, perm]), K0))
    return err


def spectrum(model, P):
    with torch.no_grad():
        lam = torch.linalg.eigvalsh(model(P))
    rank = (lam > lam[:, -1:].clamp_min(1e-300) * 1e-9).sum(1).double().mean()
    return {'rank': rank.item(), 'lam_min': lam[:, 0].median().item(),
            'lam_max': lam[:, -1].median().item()}


def make_pointwise(seed, device, **kw):
    torch.manual_seed(seed)
    return PointwiseStiffnessModel(**kw).to(device).eval()


def make_rank_channel(seed, device, candidate_k=16):
    """Comparison arm: the existing edge-tensor model with an LN backbone,
    whose channel index IS the neighbour rank."""
    torch.manual_seed(seed)
    enc = WrenchEdgeEncoder(graph='kernel', candidate_k=candidate_k)
    return WrenchSecondMomentModel(
        enc, weight_mode='learned',
        backbone_channels=(candidate_k, 16, 16, 8)).to(device).eval()


DATASETS = {
    'iid': lambda n, npts, gen: sample_clouds(n, npts, gen, trans_scale=1.0),
    'centro': lambda n, npts, gen: symmetric_clouds(n, npts, gen, eta=0.0),
    'c2': lambda n, npts, gen: c2_clouds(n, npts, gen, eta=0.0),
    'tetra': lambda n, npts, gen: tetra_orbit_clouds(n, npts, gen, eta=0.0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--n-samples', type=int, default=4)
    ap.add_argument('--n-points', type=int, default=48)
    ap.add_argument('--candidates', type=int, default=24)
    ap.add_argument('--radius-mode', default='global_scale',
                    choices=['global_scale', 'density_scaled', 'fixed',
                             'knn_adaptive', 'knn_shell'])
    ap.add_argument('--radius-alpha', type=float, default=0.75)
    ap.add_argument('--target-k', type=int, default=16,
                    help='density_scaled가 맞추려는 평균 degree')
    ap.add_argument('--lattice-side', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--full', action='store_true',
                    help='ablation 조합 전체를 A/B/D에 대해 반복')
    ap.add_argument('--out', default=None, help='결과 json 경로')
    args = ap.parse_args()

    device = torch.device(args.device)
    gen = torch.Generator().manual_seed(args.seed)
    base = dict(candidate_k=args.candidates, radius_mode=args.radius_mode,
                radius_alpha=args.radius_alpha, target_k=args.target_k)
    model = make_pointwise(args.seed, device, **base)
    results = {}

    # ---------------------------------------- A / B / D / E on each dataset
    print('=== A) equivariance, B) permutation, D) rank, E) graph ===')
    print(f'{"dataset":10s} {"equiv":>10s} {"perm":>10s} {"rank":>6s} '
          f'{"lam_min":>10s} {"deg":>6s} {"trunc":>7s}')
    for name, make in DATASETS.items():
        npts = args.n_points
        if name == 'tetra':
            npts = max(12, npts // 12 * 12)
        P = make(args.n_samples, npts, gen).to(device)
        row = {'equiv': equivariance_err(model, P, gen),
               'perm': permutation_err(model, P, gen),
               **spectrum(model, P), **model.last_graph_stats}
        results[name] = row
        print(f'{name:10s} {row["equiv"]:10.2e} {row["perm"]:10.2e} '
              f'{row["rank"]:6.2f} {row["lam_min"]:10.2e} '
              f'{row["graph_mean_degree"]:6.1f} '
              f'{row["graph_truncation_frac"]:7.3f}')

    # -------------------------------------------------- C) tie robustness
    print('\n=== C) exact-tie robustness (cubic lattice) ===')
    P = lattice_clouds(args.n_samples, args.lattice_side, gen,
                       trans_scale=1.0).to(device)
    rank_model = make_rank_channel(args.seed, device,
                                   min(args.candidates, P.shape[1] - 1))
    tie = {
        'pointwise_perm': permutation_err(model, P, gen),
        'pointwise_equiv': equivariance_err(model, P, gen),
        'rank_channel_perm': permutation_err(rank_model, P, gen),
        'rank_channel_equiv': equivariance_err(rank_model, P, gen),
    }
    results['lattice'] = {**tie, **model.last_graph_stats}
    print(f'  pointwise      perm {tie["pointwise_perm"]:.2e}   '
          f'equiv {tie["pointwise_equiv"]:.2e}')
    print(f'  rank-channel   perm {tie["rank_channel_perm"]:.2e}   '
          f'equiv {tie["rank_channel_equiv"]:.2e}')
    print('  (rank-channel perm error is the neighbour-rank channel index; '
          'it is not fixed by a smooth kernel)')

    # ------------------------------------------------- F) near-tie curve
    print('\n=== F) near-tie curve: permutation error vs lattice jitter ===')
    curve = {}
    for jitter in (0.0, 1e-8, 1e-4, 1e-2, 1e-1):
        Pj = lattice_clouds(args.n_samples, args.lattice_side, gen,
                            jitter=jitter, trans_scale=1.0).to(device)
        curve[jitter] = {'pointwise': permutation_err(model, Pj, gen),
                         'rank_channel': permutation_err(rank_model, Pj, gen)}
        print(f'  jitter {jitter:8.1e}   pointwise '
              f'{curve[jitter]["pointwise"]:.2e}   rank-channel '
              f'{curve[jitter]["rank_channel"]:.2e}')
    results['near_tie_curve'] = curve

    # ------------------------------------------------------- ablations
    if args.full:
        print('\n=== ablations (iid + c2) ===')
        variants = {
            'default': {},
            'separable-encoder-bracket': dict(bracket='separable'),
            'pairwise-encoder-bracket': dict(bracket='pairwise',
                                             bracket_channels=4),
            'no-backbone-bracket': dict(use_bracket_layers=False),
            'no-gate': dict(gate='none'),
            'full-gram-gate': dict(gate='full'),
            'no-global-context': dict(use_global_context=False),
            'message-passing': dict(message_passing=True),
            'attention-pool': dict(pool='attention'),
            'sum-pool': dict(pool='sum'),
            'knn-adaptive-radius': dict(radius_mode='knn_adaptive'),
            'knn-shell-radius': dict(radius_mode='knn_shell'),
            'density-scaled-radius': dict(radius_mode='density_scaled'),
            'uniform-beta': dict(beta_mode='uniform'),
            'force-invariant': dict(use_force_invariant=True),
        }
        abl = {}
        P_iid = sample_clouds(args.n_samples, args.n_points, gen,
                              trans_scale=1.0).to(device)
        P_c2 = c2_clouds(args.n_samples, args.n_points, gen, eta=0.0).to(device)
        P_lat = lattice_clouds(args.n_samples, args.lattice_side, gen,
                               trans_scale=1.0).to(device)
        print(f'{"variant":22s} {"equiv":>10s} {"perm(c2)":>10s} '
              f'{"perm(tie)":>10s} {"rank(c2)":>9s} {"params":>8s}')
        for name, kw in variants.items():
            m = make_pointwise(args.seed, device, **{**base, **kw})
            row = {'equiv': equivariance_err(m, P_iid, gen),
                   'perm_c2': permutation_err(m, P_c2, gen),
                   'perm_tie': permutation_err(m, P_lat, gen),
                   'rank_c2': spectrum(m, P_c2)['rank'],
                   'params': sum(q.numel() for q in m.parameters())}
            abl[name] = row
            print(f'{name:22s} {row["equiv"]:10.2e} {row["perm_c2"]:10.2e} '
                  f'{row["perm_tie"]:10.2e} {row["rank_c2"]:9.2f} '
                  f'{row["params"]:8d}')
        results['ablations'] = abl

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=float))
        print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()

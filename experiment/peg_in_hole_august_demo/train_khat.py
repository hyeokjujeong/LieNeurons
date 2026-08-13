"""K̂ 학습 — 8월 데모용 (README.md §4.1 계약 준수).

hole62/63/64 표면 PCD → 동일 body-frame diag(30,30,500,30,30,30) 라벨의
congruence 수송본을 회귀. force-invariant pointwise 인코더, 소형 capacity,
의도적 overfit. 목적: SE(3) equivariant adaptive compliant policy 검증.

repo 루트에서:
    python experiment/peg_in_hole_august_demo/train_khat.py \
        --data data/real_objects/holes626364_axis.pt --device cuda:0
"""

import argparse
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from experiment.pc_se3_congruence.pointwise_models import PointwiseStiffnessModel
from experiment.pc_se3_congruence.se3_utils import coadjoint, random_SE3
from experiment.pc_se3_congruence.spd_loss import affine_invariant_d

torch.set_default_dtype(torch.float64)


def scale_shape(chol_gt, K_pred):
    """AIRM d의 scale/shape 직교 분해 (peghole_training_report.md §5.2)."""
    X = torch.linalg.solve_triangular(chol_gt, K_pred, upper=False)
    A = torch.linalg.solve_triangular(chol_gt, X.transpose(-1, -2), upper=False)
    lam = torch.linalg.eigvalsh(0.5 * (A + A.transpose(-1, -2))).clamp_min(1e-12)
    ll = lam.log()
    mu = ll.mean(-1)
    d_scale = (6 ** 0.5) * mu.abs()
    d_shape = (ll - mu[:, None]).square().sum(-1).sqrt()
    return d_scale, d_shape


def equivariance_residual(model, P, gen, n_T=3, trans_scale=0.5):
    """max_T |K(T·P) - G K(P) G^T| — 아키텍처 보장의 수치 인증."""
    model.eval()
    with torch.no_grad():
        K0 = model(P)
        worst = 0.0
        for _ in range(n_T):
            R, p = random_SE3(trans_scale, gen)
            R, p = R.to(P.device), p.to(P.device)
            G = coadjoint(R, p)
            K1 = model(P @ R.T + p)
            worst = max(worst, float((K1 - G @ K0 @ G.T).abs().max()))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/real_objects/holes626364_axis.pt')
    ap.add_argument('--out', default='experiment/peg_in_hole_august_demo/'
                                     'khat_pointwise.pt')
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--grad-clip', type=float, default=5.0)
    ap.add_argument('--channels', type=int, nargs='+', default=[8, 16, 8])
    ap.add_argument('--factors', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    d = torch.load(args.data, weights_only=False)
    dev = args.device
    P_tr, K_tr = d['P_train'].to(dev), d['K_train'].to(dev)
    P_va, K_va = d['P_val'].to(dev), d['K_val'].to(dev)
    L_tr = torch.linalg.cholesky(K_tr)
    L_va = torch.linalg.cholesky(K_va)
    mesh_va = d['mesh_id_val']
    print(f'[data] train {P_tr.shape}  val {P_va.shape}  '
          f'meshes {d["meta"]["meshes"]}')

    model_kwargs = dict(channels=tuple(args.channels), factors=args.factors,
                        use_force_invariant=True)
    torch.manual_seed(args.seed)
    model = PointwiseStiffnessModel(**model_kwargs).double().to(dev)
    n_par = sum(q.numel() for q in model.parameters())
    print(f'[model] pointwise force-invariant, channels={args.channels} '
          f'factors={args.factors}  params {n_par}')

    gen_eq = torch.Generator().manual_seed(7)
    print(f'[equivariance, init] {equivariance_residual(model, P_va[:8], gen_eq):.2e}')

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    gp = torch.Generator().manual_seed(args.seed + 1)
    S = P_tr.shape[0]
    best = {'val_d': float('inf'), 'epoch': -1}

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(S, generator=gp)
        tot, nb = 0.0, 0
        for b in range(0, S, args.batch):
            i = perm[b:b + args.batch]
            dist, _ = affine_invariant_d(L_tr[i], model(P_tr[i]))
            loss = dist.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tot, nb = tot + loss.item(), nb + 1
        sched.step()

        model.eval()
        with torch.no_grad():
            K_pred = torch.cat([model(P_va[i:i + 128])
                                for i in range(0, P_va.shape[0], 128)])
            d_va, _ = affine_invariant_d(L_va, K_pred)
            d_sc, d_sh = scale_shape(L_va, K_pred)
        vd = d_va.mean().item()
        if vd < best['val_d']:
            best = {'val_d': vd, 'epoch': ep}
            torch.save({
                'state_dict': model.state_dict(),
                'model_kwargs': model_kwargs,
                'contract': {
                    'input': 'float64 [B, N=128, 3], canonical normalization '
                             '(center + unit max-radius; README §3.2)',
                    'output': 'float64 [B, 6, 6] SPD, ordering [m; f] '
                              '(rot 0:3, trans 3:6 — se3_utils.coadjoint 규약)',
                    'equivariance': 'K(T·P) = Ad_T^{-T} K(P) Ad_T^{-1}',
                },
                'train_meta': {'data': args.data, 'epoch': ep, 'val_d': vd,
                               'seed': args.seed, 'channels': args.channels,
                               'factors': args.factors,
                               'label_K_body_diag': d['meta'].get(
                                   'K_body_diag', [30., 30., 500., 30., 30., 30.])},
            }, args.out)
        if ep % 10 == 0 or ep == args.epochs - 1:
            per_mesh = '  '.join(
                f'h6{2 + m}:{d_va[mesh_va == m].mean().item():.3f}'
                for m in range(3))
            print(f'ep {ep:4d}  train_d {tot / nb:.4f}  val_d {vd:.4f}  '
                  f'scale {d_sc.mean():.3f}  shape {d_sh.mean():.3f}  '
                  f'[{per_mesh}]', flush=True)

    print(f'\n[best] val_d {best["val_d"]:.4f} @ ep {best["epoch"]}'
          f'  ->  {args.out}')
    print(f'[equivariance, final] '
          f'{equivariance_residual(model, P_va[:8], gen_eq):.2e}')


if __name__ == '__main__':
    main()

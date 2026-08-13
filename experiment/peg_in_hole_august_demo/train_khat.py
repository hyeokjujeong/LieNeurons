"""K̂ 학습 — 8월 데모용 (README.md §4.1 계약 준수).

hole62/63/64 표면 PCD → 동일 body-frame diag(30,30,30, 30,30,500) 라벨의
congruence 수송본을 회귀 (**[m; f] 순서** — 마지막 슬롯이 구멍 축 병진이다.
2026-08-10 리팩토링 83bb536 이전의 [f; m] 규약과 헷갈리지 말 것).
force-invariant pointwise 인코더, 소형 capacity, 의도적 overfit.
목적: SE(3) equivariant adaptive compliant policy 검증.

주의 — 축 병진 500 은 이 형상에서 물리적으로 실현 불가능하다.  기본
아키텍처(--pitch none)는 ~300 에서 포화한다.  이유와 해법은
experiment/peg_in_hole_august_demo/STIFFNESS_CEILING.md 참고 (--pitch head).

repo 루트에서:
    python data_gen/gen_holes_canonical.py            # 데이터셋 생성
    python experiment/peg_in_hole_august_demo/train_khat.py \
        --data data/real_objects/holes_canonical_axis.pt --pitch head
"""

import argparse
import copy
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from experiment.pc_se3_congruence.metrics import (WANDB_ENTITY, WANDB_PROJECT,
                                                  init_wandb)
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


def to_loss(K_pred, loss_dev):
    """Move a predicted K to wherever the 6x6 spectral algebra can run.

    On MPS that has to be the CPU: MPS has no float64 at all and no
    `aten::_linalg_eigh`.  It is a 6x6 problem so the copy costs nothing, and
    autograd flows back across the device boundary.  On CPU/CUDA loss_dev is
    the training device, so this is a no-op and the step never syncs.
    Order matters -- MPS cannot cast to float64 in place, so the DEVICE moves
    first and only then the dtype.
    """
    return K_pred.to(loss_dev).double()


def equivariance_residual(model, P, gen, n_T=3, trans_scale=0.5):
    """max_T |K(T·P) - G K(P) G^T| — 아키텍처 보장의 수치 인증.

    항상 CPU/float64 사본으로 잰다.  float32 로 재면 잔차가 ~1e-6 로 올라가는데,
    그건 아키텍처가 아니라 반올림을 재는 것이라 MODEL_CARD 의 ~1e-11 과 비교할
    수 없게 된다.  50k 파라미터라 사본 비용은 무시할 만하다.
    """
    # .cpu().double() 을 한 번에 .to(device, dtype) 으로 쓰면 MPS 가 dtype 캐스팅을
    # 장치 이동보다 먼저 시도해서 터진다.  장치부터 옮긴다.
    m64 = copy.deepcopy(model).cpu().double().eval()
    P = P.cpu().double()
    with torch.no_grad():
        K0 = m64(P)
        worst = 0.0
        for _ in range(n_T):
            R, p = random_SE3(trans_scale, gen)
            G = coadjoint(R, p)
            K1 = m64(P @ R.T + p)
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
    ap.add_argument('--device', default='cuda:0',
                    help='cpu | cuda:N | mps.  mps 는 float32 로 자동 전환된다')
    ap.add_argument('--dtype', default=None, choices=['float32', 'float64'],
                    help='백본 dtype. 기본은 mps면 float32, 그 외 float64. '
                         '손실의 6x6 선형대수는 어느 쪽이든 CPU/float64 다')
    ap.add_argument('--pitch', default='none', choices=['none', 'head', 'all'],
                    help=('두 번째 등변 생성자 N 을 쓸 층. none이면 K가 '
                          '"미는 접촉"으로 실현 가능한 원뿔에 갇혀 축 강성이 '
                          '~300에서 포화한다 (STIFFNESS_CEILING.md §8)'))
    ap.add_argument('--const-lr', action='store_true',
                    help=('cosine anneal 대신 상수 LR. 기본 스케줄은 '
                          'eta_min=0 이라 마지막 epoch에서 LR이 정확히 0이 되어 '
                          '곡선이 눕는 것을 수렴으로 오독하게 만든다'))
    ap.add_argument('--wandb-mode', default='online',
                    choices=['online', 'offline', 'disabled'])
    ap.add_argument('--wandb-project', default=WANDB_PROJECT)
    ap.add_argument('--wandb-entity', default=WANDB_ENTITY)
    args = ap.parse_args()

    dev = args.device
    is_mps = dev.startswith('mps')
    dt = getattr(torch, args.dtype) if args.dtype else (
        torch.float32 if is_mps else torch.float64)
    if is_mps and dt is torch.float64:
        raise SystemExit('MPS 에는 float64 가 없다 — --dtype float32 로 돌릴 것')

    # 손실은 언제나 float64 다.  MPS 에는 float64 도 eigvalsh 도 없으므로 그때만
    # CPU 로 뺀다 -- CPU/CUDA 에서는 학습 장치 그대로라 스텝마다 동기화가 없다.
    loss_dev = 'cpu' if is_mps else dev

    d = torch.load(args.data, weights_only=False)
    K_tr = d['K_train'].to(loss_dev).double()
    K_va = d['K_val'].to(loss_dev).double()
    L_tr, L_va = torch.linalg.cholesky(K_tr), torch.linalg.cholesky(K_va)
    P_tr = d['P_train'].to(device=dev, dtype=dt)
    P_va = d['P_val'].to(device=dev, dtype=dt)
    mesh_va = d['mesh_id_val'].to(loss_dev)
    print(f'[data] train {tuple(P_tr.shape)}  val {tuple(P_va.shape)}  '
          f'meshes {d["meta"]["meshes"]}')
    print(f'[device] backbone {dev}/{str(dt).split(".")[-1]}  '
          f'loss {loss_dev}/float64  등변성 인증 cpu/float64')

    model_kwargs = dict(channels=tuple(args.channels), factors=args.factors,
                        use_force_invariant=True, pitch=args.pitch)
    torch.manual_seed(args.seed)
    model = PointwiseStiffnessModel(**model_kwargs).to(device=dev, dtype=dt)
    n_par = sum(q.numel() for q in model.parameters())
    print(f'[model] pointwise force-invariant, channels={args.channels} '
          f'factors={args.factors}  params {n_par}')

    gen_eq = torch.Generator().manual_seed(7)
    eq0 = equivariance_residual(model, P_va[:8], gen_eq)
    print(f'[equivariance, init] {eq0:.2e}')

    tag = (f'khat-{"-".join(map(str, args.channels))}-f{args.factors}'
           f'-pitch{args.pitch}' + ('-constlr' if args.const_lr else ''))
    wb = init_wandb(tag, dict(vars(args), n_params=n_par, equivariance_init=eq0),
                    mode=args.wandb_mode, project=args.wandb_project,
                    entity=args.wandb_entity)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = (torch.optim.lr_scheduler.ConstantLR(opt, 1.0, total_iters=1)
             if args.const_lr else
             torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs))
    gp = torch.Generator().manual_seed(args.seed + 1)
    S = P_tr.shape[0]
    best = {'val_d': float('inf'), 'epoch': -1}

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(S, generator=gp)
        tot, nb = 0.0, 0
        for b in range(0, S, args.batch):
            i = perm[b:b + args.batch]
            dist, _ = affine_invariant_d(L_tr[i],
                                         to_loss(model(P_tr[i]), loss_dev))
            loss = dist.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tot, nb = tot + loss.item(), nb + 1
        sched.step()

        model.eval()
        with torch.no_grad():
            K_pred = torch.cat([to_loss(model(P_va[i:i + 128]), loss_dev)
                                for i in range(0, P_va.shape[0], 128)])
            d_va, _ = affine_invariant_d(L_va, K_pred)
            d_sc, d_sh = scale_shape(L_va, K_pred)
        vd = d_va.mean().item()
        if vd < best['val_d']:
            best = {'val_d': vd, 'epoch': ep}
            torch.save({
                # 어느 장치/정밀도로 학습했든 체크포인트는 cpu/float64 로 낸다
                # -- MODEL_CARD 의 입출력 계약이 float64 고정이기 때문이다.
                'state_dict': {k: v.detach().cpu().double()
                               for k, v in model.state_dict().items()},
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
                               'factors': args.factors, 'pitch': args.pitch,
                               'train_device': dev,
                               'train_dtype': str(dt).split('.')[-1],
                               'label_K_body_diag': d['meta'].get(
                                   'K_body_diag',
                                   [30., 30., 30., 30., 30., 500.])},
            }, args.out)
        per_mesh = {f'val/d_h6{2 + m}': d_va[mesh_va == m].mean().item()
                    for m in range(3)}
        if wb is not None:
            # 축 강성은 이 실험의 진짜 지표다 (라벨 500, 기본 아키텍처 상한 ~300
            # -- STIFFNESS_CEILING.md).  tr(K_mf) 는 pitch 구속의 불변량 지문:
            # pitch='none' 이면 정확히 0 에 붙어 있어야 한다 (같은 문서 정리 5).
            lam_ff = torch.linalg.eigvalsh(K_pred[:, 3:6, 3:6])[:, -1]
            wb.log(dict(per_mesh, **{
                'train/d': tot / nb, 'val/d': vd,
                'val/d_scale': d_sc.mean().item(),
                'val/d_shape': d_sh.mean().item(),
                'val/axial_stiffness': lam_ff.mean().item(),
                'val/tr_K_mf': torch.diagonal(
                    K_pred[:, 0:3, 3:6], dim1=-2, dim2=-1).sum(-1).abs()
                    .mean().item(),
                'lr': opt.param_groups[0]['lr'], 'best/val_d': best['val_d'],
            }), step=ep)
        if ep % 10 == 0 or ep == args.epochs - 1:
            pm = '  '.join(f'h6{2 + m}:{per_mesh[f"val/d_h6{2 + m}"]:.3f}'
                           for m in range(3))
            print(f'ep {ep:4d}  train_d {tot / nb:.4f}  val_d {vd:.4f}  '
                  f'scale {d_sc.mean():.3f}  shape {d_sh.mean():.3f}  '
                  f'[{pm}]', flush=True)

    eq1 = equivariance_residual(model, P_va[:8], gen_eq)
    print(f'\n[best] val_d {best["val_d"]:.4f} @ ep {best["epoch"]}'
          f'  ->  {args.out}')
    print(f'[equivariance, final] {eq1:.2e}')
    if wb is not None:
        wb.summary.update({'best/val_d': best['val_d'],
                           'best/epoch': best['epoch'],
                           'equivariance/final': eq1})
        wb.finish()


if __name__ == '__main__':
    main()

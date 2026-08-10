"""[SUPERSEDED gen-1 -> blockage_bench.py]
Training sanity check for the Experiment-B architecture.

    P --PlueckerEncoder(k=16)--> V --LNLinear+LNLieBracket x5--> Z
      --KleinHeadB (Y = QZ, K = YY^T/C)--> K_pred

Two targets (both exactly congruence-equivariant functions of P):
  analytic : contact-spring wrench Gram (data_synth.contact_spring_K)
  teacher  : frozen randomly-initialized ModelB of the same architecture

Loss (docs/exp.md T6 / section 7.6): affine-invariant SPD distance

    d(K_gt, K_pred) = || log( K_gt^{-1/2} K_pred K_gt^{-1/2} ) ||_F
                    = sqrt( sum_i log^2 lambda_i(A) ),
    A = L^{-1} K_pred L^{-T},  K_gt = L L^T (Cholesky),

using only eigenvalues of A (same spectrum as the sqrt-whitened matrix), so
autograd goes through torch.linalg.eigvalsh eigenvalues only — no 1/(l_i-l_j)
eigenvector terms.  The metric is congruence-invariant, so it composes with
the equivariant model into a translation-consistent objective.

Run:  python experiment/pc_se3_congruence/legacy/train.py [--quick]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append('.')

import torch

torch.set_default_dtype(torch.float64)

from experiment.pc_se3_congruence.data_synth import (contact_spring_K,
                                                     sample_clouds, spd_stats)
from experiment.pc_se3_congruence.encoders import PlueckerEncoder
from experiment.pc_se3_congruence.metrics import (WANDB_ENTITY, WANDB_PROJECT,
                                                  block_metrics, f_signal,
                                                  init_wandb)
from experiment.pc_se3_congruence.models import ModelB
from experiment.pc_se3_congruence.se3_utils import (coadjoint, random_SE3,
                                                    scaled_err,
                                                    transform_cloud)
# The loss moved to spd_loss.py so that the CURRENT entry points do not have to
# import it from this superseded module.  Re-exported here for older callers.
from experiment.pc_se3_congruence.spd_loss import (EIG_CLAMP,  # noqa: F401
                                                   affine_invariant_d)

CHANNELS = (16, 64, 128, 128, 64, 32)
K_ENCODER = 16


def build_model(seed, device):
    torch.manual_seed(seed)
    model = ModelB(PlueckerEncoder(k=K_ENCODER), channels=CHANNELS)
    return model.double().to(device)


# ------------------------------------------------------------- structural
def structural_checks(model, targets_fn, P, gen, n_T=5, trans_scale=1.0):
    """Equivariance of the trained model and invariance of the loss.

    equiv_err : max scaled err of K(T.P) vs Ad^{-T} K(P) Ad^{-1}
    loss_inv  : max relative change of mean d(K_gt, K_pred) under T
                (uses freshly generated targets for T.P, so it also exercises
                the target function's own equivariance).
    """
    model.eval()
    with torch.no_grad():
        K0 = model(P)
        d0, _ = affine_invariant_d(torch.linalg.cholesky(targets_fn(P)), K0)
        equiv, loss_inv = 0.0, 0.0
        for _ in range(n_T):
            R, p = random_SE3(trans_scale, gen)
            R, p = R.to(P.device), p.to(P.device)
            Pt = transform_cloud(P, R, p)
            Kt = model(Pt)
            rho = coadjoint(R, p)
            equiv = max(equiv, scaled_err(Kt, rho @ K0 @ rho.transpose(-1, -2)))
            dt, _ = affine_invariant_d(torch.linalg.cholesky(targets_fn(Pt)), Kt)
            loss_inv = max(loss_inv,
                           ((dt.mean() - d0.mean()).abs() / d0.mean()).item())
    model.train()
    return {'equiv_err': equiv, 'loss_invariance_err': loss_inv}


# --------------------------------------------------------------- training
def run(target, data, args, device, outdir):
    P_tr, P_va = data['P_tr'], data['P_va']

    if target == 'analytic':
        targets_fn = lambda P: contact_spring_K(P, k=args.k_gt,
                                                sigma_k=args.sigma_k)
    else:
        teacher = build_model(args.teacher_seed, device)
        for q in teacher.parameters():
            q.requires_grad_(False)
        teacher.eval()
        targets_fn = lambda P: torch.cat(
            [teacher(P[i:i + 512]) for i in range(0, P.shape[0], 512)])

    with torch.no_grad():
        K_tr, K_va = targets_fn(P_tr), targets_fn(P_va)
    gt_stats = {'train': spd_stats(K_tr), 'val': spd_stats(K_va)}
    L_tr, L_va = torch.linalg.cholesky(K_tr), torch.linalg.cholesky(K_va)

    model = build_model(args.model_seed, device)
    n_params = sum(q.numel() for q in model.parameters())
    wb = init_wandb(f'train-{target}', {**vars(args), 'target': target,
                                        'n_params': n_params,
                                        'channels': CHANNELS,
                                        'k_encoder': K_ENCODER},
                    mode=args.wandb_mode, project=args.wandb_project,
                    entity=args.wandb_entity)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    gen_T = torch.Generator().manual_seed(args.data_seed + 1)
    P_chk = P_va[:64]
    checks = {'init': structural_checks(model, targets_fn, P_chk, gen_T)}

    S = P_tr.shape[0]
    gen_perm = torch.Generator().manual_seed(args.data_seed + 2)
    hist = {'train_d': [], 'val_d': [], 'lr': [], 'lam_min': [], 'lam_max': [],
            'clamped': []}
    t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(S, generator=gen_perm)
        tot, nb, ncl = 0.0, 0, 0
        for b in range(0, S, args.batch):
            i = perm[b:b + args.batch]
            d, n_c = affine_invariant_d(L_tr[i], model(P_tr[i]))
            loss = d.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot, nb, ncl = tot + loss.item(), nb + 1, ncl + n_c
        sched.step()

        model.eval()
        with torch.no_grad():
            K_va_pred = model(P_va)
            d_va, _ = affine_invariant_d(L_va, K_va_pred)
        diag = block_metrics(K_va_pred, K_va)
        diag['f_signal'] = f_signal(model, P_va[:64], slice(3, 6))  # [v; w]
        model.train()
        hist['train_d'].append(tot / nb)
        hist['val_d'].append(d_va.mean().item())
        hist['lr'].append(sched.get_last_lr()[0])
        hist['lam_min'].append(diag['lam_min'])
        hist['lam_max'].append(diag['lam_max'])
        hist['clamped'].append(ncl)
        if wb:
            wb.log({'train_d': hist['train_d'][-1], 'val_d': hist['val_d'][-1],
                    'lr': hist['lr'][-1], 'clamped': ncl, **diag}, step=ep)
        if ep % max(1, args.epochs // 20) == 0 or ep == args.epochs - 1:
            print(f'[{target}] ep {ep:4d}  train d {hist["train_d"][-1]:.4f}  '
                  f'val d {hist["val_d"][-1]:.4f}  clamped {ncl}', flush=True)

    checks['final'] = structural_checks(model, targets_fn, P_chk, gen_T)
    runtime = time.time() - t0

    out = {
        'target': target, 'n_params': n_params, 'channels': CHANNELS,
        'k_encoder': K_ENCODER, 'epochs': args.epochs, 'batch': args.batch,
        'lr': args.lr, 'gt_stats': gt_stats, 'checks': checks,
        'runtime_s': runtime, 'history': hist,
        'first_epoch_d': hist['train_d'][0], 'final_train_d': hist['train_d'][-1],
        'final_val_d': hist['val_d'][-1],
    }
    (outdir / f'{target}_results.json').write_text(json.dumps(out, indent=2))
    torch.save(model.state_dict(), outdir / f'{target}_model.pt')
    if wb:
        # 스칼라 요약만 — 체크포인트/json은 로컬 유지 (wandb 업로드 금지)
        wb.summary.update({
            'final_train_d': hist['train_d'][-1],
            'final_val_d': hist['val_d'][-1],
            'first_epoch_d': hist['train_d'][0],
            'runtime_s': runtime,
            'equiv_err_init': checks['init']['equiv_err'],
            'equiv_err_final': checks['final']['equiv_err'],
            'loss_inv_err_final': checks['final']['loss_invariance_err'],
            'gt_lam_min_train': gt_stats['train']['lam_min'],
            'gt_cond_median_train': gt_stats['train']['cond_median'],
        })
        wb.finish()
    print(f'[{target}] done in {runtime:.1f}s  '
          f'train d {hist["train_d"][0]:.3f} -> {hist["train_d"][-1]:.4f}  '
          f'equiv {checks["final"]["equiv_err"]:.2e}')
    return out


def plot(results, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 4))
    axes = [axes] if len(results) == 1 else list(axes)
    for ax, r in zip(axes, results):
        ep = range(1, len(r['history']['train_d']) + 1)
        ax.plot(ep, r['history']['train_d'], label='train')
        ax.plot(ep, r['history']['val_d'], label='val', ls='--')
        ax.set_yscale('log')
        ax.set_xlabel('epoch')
        ax.set_ylabel(r'$\|\log(K_{gt}^{-1/2} K_{pred} K_{gt}^{-1/2})\|_F$')
        ax.set_title(f"target = {r['target']}")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / 'loss_curves.png', dpi=150)
    print('wrote', outdir / 'loss_curves.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', default='both',
                    choices=['analytic', 'teacher', 'both'])
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--n-train', type=int, default=4096)
    ap.add_argument('--n-val', type=int, default=512)
    ap.add_argument('--n-points', type=int, default=128)
    ap.add_argument('--trans-scale', type=float, default=1.0)
    ap.add_argument('--k-gt', type=int, default=12)
    ap.add_argument('--sigma-k', type=float, default=0.5)
    ap.add_argument('--data-seed', type=int, default=100)
    ap.add_argument('--model-seed', type=int, default=0)
    ap.add_argument('--teacher-seed', type=int, default=7)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--outdir',
                    default='experiment/pc_se3_congruence/train_results')
    ap.add_argument('--wandb-mode', default='online',
                    choices=['online', 'offline', 'disabled'])
    ap.add_argument('--wandb-project', default=WANDB_PROJECT)
    ap.add_argument('--wandb-entity', default=WANDB_ENTITY)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.epochs, args.n_train, args.n_val = 8, 512, 128

    device = torch.device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gen = torch.Generator().manual_seed(args.data_seed)
    P_all = sample_clouds(args.n_train + args.n_val, args.n_points, gen,
                          trans_scale=args.trans_scale).to(device)
    data = {'P_tr': P_all[:args.n_train], 'P_va': P_all[args.n_train:]}

    targets = ['analytic', 'teacher'] if args.target == 'both' else [args.target]
    results = [run(t, data, args, device, outdir) for t in targets]
    plot(results, outdir)


if __name__ == '__main__':
    main()

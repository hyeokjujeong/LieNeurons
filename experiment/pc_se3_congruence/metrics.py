"""Shared diagnostics + wandb helper for the pc_se3_congruence experiments.

wandb policy: scalars/summaries only.  init_wandb() disables code saving and
we never call wandb.save / wandb.watch / Artifact — model checkpoints and
result json stay local (train_results/), so nothing heavy is uploaded.
"""
import os

import torch

WANDB_ENTITY = 'adjoint_equivariant_network'
WANDB_PROJECT = 'pc-se3-congruence'


def block_metrics(K_pred, K_gt, eps=1e-30):
    """Per-block relative Frobenius errors + rank/eigenvalue stats.

    ff/fm/mm 블록별 오차는 실패를 국소화하는 핵심 진단이다
    (bracket_blockage_analysis.md — ff 오차만 높으면 blockage/인코더 소멸,
    전 블록이 높으면 용량·표본 문제).
    """
    out = {}
    blocks = {'ff': (slice(0, 3), slice(0, 3)),
              'fm': (slice(0, 3), slice(3, 6)),
              'mm': (slice(3, 6), slice(3, 6))}
    for name, (r, c) in blocks.items():
        num = (K_pred[:, r, c] - K_gt[:, r, c]).norm(dim=(1, 2))
        den = K_gt[:, r, c].norm(dim=(1, 2)) + eps
        out[f'err_rel_{name}'] = (num / den).mean().item()
    lam = torch.linalg.eigvalsh(K_pred)
    out['rank_pred'] = (lam > lam[:, -1:].clamp_min(eps) * 1e-9) \
        .sum(1).double().mean().item()
    out['lam_min'] = lam[:, 0].median().item()
    out['lam_max'] = lam[:, -1].median().item()
    return out


def airm_scale_shape(chol_gt, K_pred, eig_clamp=1e-12):
    """Split the AIRM distance into an isotropic-scale part and a shape part.

    With log(lambda_i) = mu + delta_i and mu the mean over the six generalized
    eigenvalues (so sum delta_i = 0), the split is exactly orthogonal:

        d^2 = 6 mu^2 + sum delta_i^2 = d_scale^2 + d_shape^2,
        mu  = (log det K_pred - log det K_gt) / 6.

    Why it matters here: the model carries exactly ONE global scale handle (the
    learned scalar e^g in the head).  A large d_scale therefore means that one
    scalar has not converged, not that the model lacks expressivity; only
    d_shape is evidence of model-class mismatch.

    The Pythagorean identity is PER SAMPLE (verified to 2e-15); the values
    logged here are batch means, so they do not recompose into ``val_d`` --
    read them as the average magnitude of each component.  ``scale_ratio`` is the
    geometric-mean eigenvalue ratio e^mu -- above 1 the prediction is too
    stiff.  It is also the honest version of the exp(d/sqrt6) reading, which
    assumes the whole error is isotropic.
    """
    X = torch.linalg.solve_triangular(chol_gt, K_pred, upper=False)
    A = torch.linalg.solve_triangular(chol_gt, X.transpose(-1, -2), upper=False)
    A = 0.5 * (A + A.transpose(-1, -2))
    ll = torch.linalg.eigvalsh(A).clamp_min(eig_clamp).log()
    mu = ll.mean(-1)
    n = ll.shape[-1]
    d_scale = (n ** 0.5) * mu.abs()
    d_shape = (ll - mu.unsqueeze(-1)).square().sum(-1).sqrt()
    return {'d_scale': d_scale.mean().item(),
            'd_shape': d_shape.mean().item(),
            # signed, in log units: >0 means systematically over-stiff
            'scale_bias': mu.mean().item(),
            'scale_ratio': mu.mean().exp().item()}


def group_metrics(K_pred, K_gt, d, group, names, eps=1e-30):
    """Same diagnostics as block_metrics, split by an integer group label.

    peg-and-hole의 stage(free/search/insert)처럼 타깃의 물리적 성격이 그룹마다
    다른 데이터에서, 총합 val_d 하나는 어느 그룹이 실패했는지 감춘다.
    """
    out = {}
    for gi, name in enumerate(names):
        m = group == gi
        n = int(m.sum().item())
        out[f'{name}/n'] = n
        if n == 0:
            continue
        out[f'{name}/val_d'] = d[m].mean().item()
        for blk, (r, c) in {'ff': (slice(0, 3), slice(0, 3)),
                            'fm': (slice(0, 3), slice(3, 6)),
                            'mm': (slice(3, 6), slice(3, 6))}.items():
            num = (K_pred[m][:, r, c] - K_gt[m][:, r, c]).norm(dim=(1, 2))
            den = K_gt[m][:, r, c].norm(dim=(1, 2)) + eps
            out[f'{name}/err_rel_{blk}'] = (num / den).mean().item()
    return out


def f_signal(model, P, direction_slots):
    """Encoder direction-channel norm — ff-계보의 유일한 입력 신호.

    direction_slots: covector [f; m]이면 slice(0, 3), twist [v; w]면 slice(3, 6).
    """
    with torch.no_grad():
        W = model.encoder(P)
    # Works for both globally pooled [B,C,6,1] encoders and late-pooling
    # [B,C,6,N/E] encoders.  Norm is over the physical 3-vector slot; all
    # channel/point/edge instances are then averaged.
    return W[:, :, direction_slots, :].norm(dim=2).mean().item()


def init_wandb(name, config, mode='online',
               project=WANDB_PROJECT, entity=WANDB_ENTITY):
    """Scalar-only wandb run.  대용량 업로드(코드·체크포인트·아티팩트) 차단.

    실패(미로그인/권한/오프라인)해도 학습은 계속되도록 None을 돌려준다.
    """
    if mode == 'disabled':
        return None
    os.environ.setdefault('WANDB_DISABLE_CODE', 'true')   # 코드 스냅샷 업로드 금지
    try:
        import wandb
        return wandb.init(entity=entity, project=project, name=name,
                          config=config, mode=mode, save_code=False,
                          reinit=True)
    except Exception as e:
        print(f'[wandb 비활성화: {e}]')
        return None

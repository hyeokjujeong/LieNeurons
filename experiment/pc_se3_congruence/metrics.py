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


def f_signal(model, P, direction_slots):
    """Encoder direction-channel norm — ff-계보의 유일한 입력 신호.

    direction_slots: covector [f; m]이면 slice(0, 3), twist [v; w]면 slice(3, 6).
    """
    with torch.no_grad():
        W = model.encoder(P)
    return W[:, :, direction_slots, 0].norm(dim=-1).mean().item()


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

"""Blockage failure-scenario benchmark with wandb monitoring.

Datasets (bracket_blockage_analysis.md §6):
  centro : centro-symmetric clouds {±a_i} + eta*noise — rank(K) <= 3 at eta=0,
           analytic loss floor d >= sqrt(3)*|log EIG_CLAMP|  (§6.2)
  c2     : single 180-deg symmetry axis — rank(K_ff) <= 1 at eta=0  (§6.5a)
  tetra  : A4 tetrahedral orbits (no -I) — f-channels tie-noise limited (§6.5a)
  iid    : no symmetry at all; sweep --n-points to see the statistical decay
           |f_c| ~ N^{-5/6} vs O(1) target ff-blocks  (§6.5c)
  fiber  : eval-only — construct same-{f_c} cloud pairs with different target
           ff-blocks; gateless models are provably constant on the pair (§6.5b)

Method (입력 표현 선택 — 수학적으로 Q-conjugate 동치, 코드 경로만 다름):
  vector   : twist(adjoint) 입력 — PlueckerEncoder + twist backbone + Klein head
  covector : wrench(coadjoint) 입력 — WrenchPlueckerEncoder + covector backbone

Per-epoch metrics (wandb + stdout): train/val AIRM distance, per-block relative
errors (ff/fm/mm — localizes the failure), prediction rank, encoder direction-
channel signal |f_c|, eigenvalue clamp count.

Examples:
  python experiment/pc_se3_congruence/blockage_bench.py --dataset centro --eta 0.0
  python experiment/pc_se3_congruence/blockage_bench.py --dataset iid --n-points 512
  # Does a learnable VN lift remove tetrahedral kNN tie instability?
  python experiment/pc_se3_congruence/blockage_bench.py --dataset tetra \
      --encoder learnable --method covector
  # Tie-robust second-order model and matching all-pairs target
  python experiment/pc_se3_congruence/blockage_bench.py --dataset c2 \
      --encoder tensor --method covector --tensor-graph all --target-graph all
  # Local compact-kernel second moment (bounded edge storage, robust boundary)
  python experiment/pc_se3_congruence/blockage_bench.py --dataset c2 \
      --encoder tensor --method covector --tensor-graph kernel \
      --target-graph kernel --kernel-candidates 32
  # Pointwise pipeline: set pooling -> point축 유지 LN backbone -> late Gram
  python experiment/pc_se3_congruence/blockage_bench.py --dataset c2 \
      --encoder pointwise --method covector --target-graph kernel
  # 같은 클래스의 고정 난수 teacher를 타깃으로 한 realizability 검증
  python experiment/pc_se3_congruence/blockage_bench.py --dataset centro \
      --encoder pointwise --method covector --target-graph teacher
  python experiment/pc_se3_congruence/blockage_bench.py --suite            # 표준 그리드
  python experiment/pc_se3_congruence/blockage_bench.py --dataset fiber   # 학습 없음
  ... --wandb-mode disabled  (wandb 없이 stdout만)
"""
import argparse
import sys

sys.path.append('.')

import torch

torch.set_default_dtype(torch.float64)

from experiment.pc_se3_congruence.data_synth import (c2_clouds,
                                                     contact_spring_all_pairs_K,
                                                     contact_spring_K,
                                                     contact_spring_kernel_K,
                                                     sample_clouds,
                                                     symmetric_clouds,
                                                     tetra_orbit_clouds)
from experiment.pc_se3_congruence.encoders import (BracketPlueckerEncoder,
                                                   LearnableLiftEncoder,
                                                   PlueckerEncoder,
                                                   WrenchEdgeEncoder,
                                                   WrenchLearnableLiftEncoder,
                                                   WrenchPlueckerEncoder)
from experiment.pc_se3_congruence.metrics import (WANDB_ENTITY,
                                                     WANDB_PROJECT,
                                                     block_metrics, f_signal,
                                                     init_wandb)
from experiment.pc_se3_congruence.models import (DualBackbone, GateBackbone,
                                                   ModelB, ModelPC2K,
                                                   WrenchSecondMomentModel)
from experiment.pc_se3_congruence.pointwise_models import \
    PointwiseStiffnessModel
from experiment.pc_se3_congruence.train import EIG_CLAMP, affine_invariant_d

DATASETS = {
    'centro': symmetric_clouds,
    'c2': c2_clouds,
    'tetra': tetra_orbit_clouds,
    'iid': lambda n, npts, gen, **kw: sample_clouds(n, npts, gen, trans_scale=1.0),
}


# ----------------------------------------------------------------- training
def build_pointwise_model(args, seed):
    """encoder=pointwise: tie-safe graph -> set pooling -> pointwise LN blocks
    -> late second moment.  The neighbour axis is reduced by learned invariant
    set aggregation (never by neighbour rank) and the POINT axis survives to
    the Gram, which is what preserves the factor gauge on symmetric clouds."""
    torch.manual_seed(seed)
    return PointwiseStiffnessModel(
        channels=tuple(args.pw_channels), factors=args.pw_factors,
        candidate_k=args.pw_candidates, radius_mode=args.pw_radius_mode,
        radius_alpha=args.pw_radius_alpha, radius_value=args.pw_radius,
        support_k=args.pw_support_k, target_k=args.pw_target_k,
        tie_eps=args.pw_tie_eps, n_rbf=args.pw_rbf, pool=args.pw_pool,
        bracket=args.pw_bracket, bracket_channels=args.pw_bracket_channels,
        use_bracket_layers=not args.pw_no_bracket_layers,
        gate=args.pw_gate, use_global_context=not args.pw_no_global_context,
        message_passing=args.pw_message_passing,
        msg_channels=args.pw_msg_channels, hidden=args.pw_hidden,
        n_proj=args.pw_proj, normalize=args.pw_normalize,
        beta_mode=args.pw_beta, use_force_invariant=args.pw_force_invariant)


def build_model(args, seed=None):
    method, seed = args.method, args.model_seed if seed is None else seed
    k, channels = args.k_enc, tuple(args.channels)
    nonlinear, encoder = args.nonlinear, args.encoder
    lift_hidden, tensor_graph = args.lift_hidden, args.tensor_graph
    tensor_weight, tensor_hidden = args.tensor_weight, args.tensor_hidden
    sigma_k, kernel_candidates = args.sigma_k, args.kernel_candidates
    tensor_backbone = args.tensor_backbone
    tensor_backbone_channels = tuple(args.tensor_backbone_channels)
    if encoder == 'pointwise':
        return build_pointwise_model(args, seed)
    torch.manual_seed(seed)
    if encoder == 'tensor':
        assert method == 'covector', 'encoder=tensor는 covector method로 실행'
        enc = WrenchEdgeEncoder(k=k, graph=tensor_graph,
                                candidate_k=kernel_candidates)
        backbone_channels = None
        if tensor_backbone == 'covector':
            if tensor_graph == 'all':
                in_channels = 1
            elif tensor_graph == 'knn':
                in_channels = k
            else:
                in_channels = kernel_candidates
            backbone_channels = (in_channels,) + tuple(tensor_backbone_channels)
        model = WrenchSecondMomentModel(
            enc, weight_mode=tensor_weight, sigma=sigma_k,
            hidden=tensor_hidden, backbone_channels=backbone_channels)
    elif encoder == 'learnable':
        if method == 'covector':
            enc = WrenchLearnableLiftEncoder(
                out_channels=channels[0], hidden=lift_hidden, k=k)
            model = ModelPC2K(enc, channels=channels)
        else:
            enc = LearnableLiftEncoder(
                out_channels=channels[0], hidden=lift_hidden, k=k)
            model = ModelB(enc, channels=channels)
    elif encoder == 'bracket':
        assert method == 'covector', 'encoder=bracket은 covector method로 실행'
        enc = BracketPlueckerEncoder(k=k)
        if channels[0] != enc.out_channels:
            print(f'[encoder=bracket] channels[0] {channels[0]} -> {enc.out_channels}')
            channels = (enc.out_channels,) + tuple(channels[1:])
        model = ModelPC2K(enc, channels=channels)
    elif method == 'covector':
        model = ModelPC2K(WrenchPlueckerEncoder(k=k), channels=channels)
    else:
        model = ModelB(PlueckerEncoder(k=k), channels=channels)
    if encoder != 'tensor' and nonlinear == 'gate':
        model.backbone = GateBackbone(channels)
    elif encoder != 'tensor' and nonlinear == 'dual':
        model.backbone = DualBackbone(channels, method=method)
    return model


def forward_K(method, model, P):
    out = model(P)
    return out[1] if isinstance(out, tuple) else out


def equiv_check(args, model, P, n_T=3):
    from experiment.pc_se3_congruence.se3_utils import (coadjoint, random_SE3,
                                                        scaled_err,
                                                        transform_cloud)
    gen = torch.Generator().manual_seed(args.data_seed + 7)
    model.eval()
    err = 0.0
    with torch.no_grad():
        K0 = forward_K(args.method, model, P)
        for _ in range(n_T):
            R, pt = random_SE3(1.0, gen)
            R, pt = R.to(P.device), pt.to(P.device)
            Kt = forward_K(args.method, model, transform_cloud(P, R, pt))
            rho = coadjoint(R, pt)
            err = max(err, scaled_err(Kt, rho @ K0 @ rho.transpose(-1, -2)))
    model.train()
    return err


def run_training(args, wb):
    gen = torch.Generator().manual_seed(args.data_seed)
    make = DATASETS[args.dataset]
    n_total = args.n_train + args.n_val
    P = make(n_total, args.n_points, gen, eta=args.eta) \
        if args.dataset != 'iid' else make(n_total, args.n_points, gen)
    P = P.to(args.device)
    if args.target_graph == 'all':
        target_fn = lambda x: contact_spring_all_pairs_K(
            x, sigma_k=args.sigma_k)
    elif args.target_graph == 'kernel':
        target_fn = lambda x: contact_spring_kernel_K(
            x, candidate_k=args.kernel_candidates, sigma_k=args.sigma_k)
    elif args.target_graph == 'teacher':
        # Realizability control: the target is a frozen, randomly initialized
        # model OF THE SAME CLASS.  It separates optimisability from
        # expressivity -- a large residual here cannot be blamed on the target
        # lying outside the model class.
        if args.encoder != 'pointwise':
            raise ValueError("--target-graph teacher는 --encoder pointwise 전용")
        teacher = build_pointwise_model(args, args.teacher_seed).to(args.device)
        for q in teacher.parameters():
            q.requires_grad_(False)
        teacher.eval()
        target_fn = lambda x: teacher(x)
    else:
        target_fn = lambda x: contact_spring_K(
            x, k=args.k_gt, sigma_k=args.sigma_k)
    # All-pairs tensors are deliberately evaluated in chunks: a full recipe
    # otherwise materializes [4096,128,128,...] intermediates at once.
    with torch.no_grad():
        K_gt = torch.cat([
            target_fn(P[i:i + args.target_batch])
            for i in range(0, P.shape[0], args.target_batch)
        ])
    lam_gt = torch.linalg.eigvalsh(K_gt)
    if lam_gt[:, 0].min() <= 0:
        raise ValueError(
            f'target is not SPD (min eigenvalue {lam_gt[:, 0].min():.3e}); '
            'the AIRM distance is undefined there')
    print(f'[target={args.target_graph}] lam_min {lam_gt[:, 0].min():.3e}  '
          f'cond median {(lam_gt[:, -1] / lam_gt[:, 0]).median():.3e}')
    L_gt = torch.linalg.cholesky(K_gt)
    P_tr, P_va = P[:args.n_train], P[args.n_train:]
    L_tr, L_va = L_gt[:args.n_train], L_gt[args.n_train:]
    K_va = K_gt[args.n_train:]

    model = build_model(args).to(args.device)
    n_params = sum(q.numel() for q in model.parameters())
    print(f'[model={args.encoder}] trainable parameters: {n_params}')
    if wb: wb.summary['n_params'] = n_params
    trainable = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.Adam(trainable, lr=args.lr) if trainable else None
    epochs = args.epochs if opt is not None else 1
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=args.lr * 0.01)
        if opt is not None else None)
    if opt is None:
        print(f'[encoder={args.encoder}, weight={args.tensor_weight}] '
              '학습 파라미터가 없어 1회 구조 평가만 수행')

    eq0 = equiv_check(args, model, P_va[:32])
    print(f'학습 전 equivariance: {eq0:.2e}')
    if wb: wb.summary['equiv_err_init'] = eq0

    if (args.dataset == 'centro' and args.eta == 0.0
            and args.encoder != 'tensor'):
        floor = (3 ** 0.5) * abs(torch.log(torch.tensor(EIG_CLAMP)).item())
        print(f'[centro eta=0] 해석적 하한: d >= {floor:.2f}')
        if wb: wb.summary['analytic_floor_d'] = floor

    S = P_tr.shape[0]
    gp = torch.Generator().manual_seed(args.data_seed + 1)
    for ep in range(epochs):
        perm = torch.randperm(S, generator=gp)
        tot, nb, ncl = 0.0, 0, 0
        for b in range(0, S, args.batch):
            i = perm[b:b + args.batch]
            d, n_c = affine_invariant_d(L_tr[i], forward_K(args.method, model, P_tr[i]))
            loss = d.mean()
            if opt is not None:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot, nb, ncl = tot + loss.item(), nb + 1, ncl + n_c
        if sched is not None:
            sched.step()

        model.eval()
        with torch.no_grad():
            K_pred = forward_K(args.method, model, P_va)
            d_va, _ = affine_invariant_d(L_va, K_pred)
        # Snapshot before f_signal below runs another forward on a subset and
        # overwrites the stats with that subset's graph.
        graph_stats = dict(getattr(model, 'last_graph_stats', {}))
        log = {'train_d': tot / nb, 'val_d': d_va.mean().item(),
               'clamped': ncl,
               'lr': sched.get_last_lr()[0] if sched is not None else 0.0,
               'f_signal': f_signal(model, P_va[:64],
                                    slice(0, 3) if args.method == 'covector'
                                    else slice(3, 6)),   # covector:[f;m] / vector:[v;w]
               **block_metrics(K_pred, K_va),
               # graph_truncation_frac > 0 이면 support 안의 이웃이 top-k에서
               # 잘려나갔다는 뜻 — set-equivariance 보장이 깨진다.
               **graph_stats}
        model.train()
        if wb: wb.log(log, step=ep)
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            extra = (f'  deg {log["graph_mean_degree"]:.1f}'
                     f'  trunc {log["graph_truncation_frac"]:.3f}'
                     if 'graph_mean_degree' in log else '')
            print(f'ep {ep:4d}  train d {log["train_d"]:8.3f}  '
                  f'val d {log["val_d"]:8.3f}  ff {log["err_rel_ff"]:.3f}  '
                  f'mm {log["err_rel_mm"]:.3f}  rank {log["rank_pred"]:.1f}  '
                  f'|f_c| {log["f_signal"]:.4f}{extra}', flush=True)
    eq1 = equiv_check(args, model, P_va[:32])
    print(f'학습 후 equivariance: {eq1:.2e}')
    if wb: wb.summary['equiv_err_final'] = eq1
    return log


# --------------------------------------------- fiber-collision demo (eval)
def run_fiber(args, wb):
    """Fiber collision (§6.5b): 같은 f-요약, 다른 타깃을 갖는 cloud 쌍에서
    모델 ff-예측이 동일함을 보인다.

    구현 노트: 일반(비대칭) fiber 쌍을 null-space + Newton으로 구성하려 했으나
    kNN 순위 채널의 조밀한 불연속(미세 교란에도 순위 스왑 다발) 때문에 수치
    구성이 막힌다 — 이 fragility 자체가 순위-채널 인코더의 병리다.  여기서는
    fiber의 정확한 인스턴스인 "서로 다른 두 중심대칭 cloud"를 쓴다: 둘 다
    f_c = 0 (동일한 f-요약)이지만 타깃 ff-블록은 O(1)로 다르다."""
    gen = torch.Generator().manual_seed(args.data_seed)
    P0 = symmetric_clouds(1, args.n_points, gen, eta=0.0).to(args.device)
    P1 = symmetric_clouds(1, args.n_points, gen, eta=0.0).to(args.device)
    enc = WrenchPlueckerEncoder(k=args.k_enc)

    def fc(P):
        return enc(P)[:, :, 0:3, 0]

    model = build_model(args).to(args.device).eval()
    with torch.no_grad():
        Kp0 = forward_K(args.method, model, P0)
        Kp1 = forward_K(args.method, model, P1)
    ff0 = contact_spring_K(P0, k=args.k_gt, sigma_k=args.sigma_k)[:, 0:3, 0:3]
    ff1 = contact_spring_K(P1, k=args.k_gt, sigma_k=args.sigma_k)[:, 0:3, 0:3]
    res = {
        'fc_norm_P0': fc(P0).norm().item(),
        'fc_norm_P1': fc(P1).norm().item(),
        'fc_gap': (fc(P0) - fc(P1)).norm().item(),
        'target_ff_gap_rel': ((ff0 - ff1).norm() / ff0.norm()).item(),
        'model_ff_gap': (Kp0[:, 0:3, 0:3] - Kp1[:, 0:3, 0:3]).norm().item(),
        'model_ff_norm': Kp0[:, 0:3, 0:3].norm().item(),
    }
    print('[fiber] 같은 f-요약(f_c=0), 다른 타깃의 cloud 쌍:')
    for k, v in res.items():
        print(f'  {k:18s} = {v:.3e}')
    print('  => fc_gap ~ 0, target_ff_gap O(1), model_ff_gap ~ 0'
          ' 이면 §6.5b 확인 (모델은 ff에서 두 cloud 구별 불가)')
    if wb: wb.summary.update(res)
    return res


# --------------------------------------------------------------------- main
RECIPES = {
    'toy': {},
    # train_report.md §5.1의 학습 하이퍼파라미터 (method와는 직교하는 축)
    'full': dict(channels=[16, 64, 128, 128, 64, 32],
                 k_enc=16, k_gt=12, sigma_k=0.5, lr=1e-3, batch=64,
                 epochs=150, n_train=4096, n_val=512, n_points=128),
}


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--recipe', default='toy', choices=list(RECIPES))
    pre_args, _ = pre.parse_known_args()

    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument('--dataset', default='centro',
                    choices=list(DATASETS) + ['fiber'])
    ap.add_argument('--encoder', default='plueck',
                    choices=['plueck', 'learnable', 'bracket', 'tensor',
                             'pointwise'],
                    help=('plueck: 기존 global mean / learnable: VN lift 뒤 global mean / '
                          'bracket: pooling-전 bracket / tensor: edge를 유지해 ww^T late pooling / '
                          'pointwise: set pooling + point축 유지 LN backbone + late second moment'))
    ap.add_argument('--nonlinear', default='bracket',
                    choices=['bracket', 'gate', 'dual'],
                    help='백본 비선형성: bracket(기존) / gate(Gram gate로 교체) / dual(병렬 두 브랜치+bracket 병합)')
    ap.add_argument('--method', default='vector', choices=['vector', 'covector'],
                    help='vector: twist(adjoint) 입력 + Klein head / covector: wrench(coadjoint) 입력')
    ap.add_argument('--eta', type=float, default=0.0)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--n-train', type=int, default=512)
    ap.add_argument('--n-val', type=int, default=128)
    ap.add_argument('--n-points', type=int, default=64)
    ap.add_argument('--k-enc', type=int, default=8)
    ap.add_argument('--k-gt', type=int, default=8)
    ap.add_argument('--sigma-k', type=float, default=0.5)
    ap.add_argument('--target-graph', default='knn',
                    choices=['knn', 'kernel', 'all', 'teacher'],
                    help=('GT contact spring graph; kernel은 local compact window, '
                          'all은 exact-tie 대조군, teacher는 같은 클래스의 고정 '
                          '난수 모델(realizability 검증, pointwise 전용)'))
    ap.add_argument('--teacher-seed', type=int, default=7,
                    help='--target-graph teacher의 고정 teacher 모델 seed')
    ap.add_argument('--target-batch', type=int, default=64,
                    help='GT 생성 chunk 크기 (all-pairs 메모리 제어)')
    ap.add_argument('--lift-hidden', type=int, default=8,
                    help='learnable VN lift의 hidden channel 수')
    ap.add_argument('--tensor-graph', default='all',
                    choices=['knn', 'kernel', 'all'],
                    help=('tensor edge graph; kernel은 bounded local 후보와 '
                          'smooth zero-boundary window 사용'))
    ap.add_argument('--kernel-candidates', type=int, default=None,
                    help=('kernel graph의 최대 local 후보 수; 기본값은 4*k-enc. '
                          'N-1보다 크면 자동으로 줄임'))
    ap.add_argument('--tensor-weight', default='learned',
                    choices=['learned', 'analytic', 'uniform'],
                    help='2차 pooling의 invariant radial weight')
    ap.add_argument('--tensor-hidden', type=int, default=32,
                    help='learned tensor radial MLP hidden channel 수')
    ap.add_argument('--tensor-backbone', default='none',
                    choices=['none', 'covector'],
                    help=('none: radial second moment 직접 pooling; covector: '
                          'pooling 전에 LNLinear+covector-bracket stack 적용'))
    ap.add_argument('--tensor-backbone-channels', type=int, nargs='+',
                    default=[32, 32, 16],
                    help='tensor covector backbone의 hidden/output channel 수')
    ap.add_argument('--channels', type=int, nargs='+', default=[8, 32, 32, 16])

    # ------------------------------------------------ encoder=pointwise 전용
    # 축 규약: N=point, k=neighbor(집합), C=latent channel, H=factor, K=6x6 강성
    pw = ap.add_argument_group('pointwise pipeline')
    pw.add_argument('--pw-channels', type=int, nargs='+', default=[8, 16, 32, 16],
                    help='[C_0(set pooling), C_1, ...]; 권장 8-16-32-16')
    pw.add_argument('--pw-factors', type=int, default=8, help='factor 채널 H')
    pw.add_argument('--pw-candidates', type=int, default=32,
                    help='point당 후보 이웃 수 k (support보다 넉넉해야 한다)')
    pw.add_argument('--pw-radius-mode', default='global_scale',
                    choices=['global_scale', 'density_scaled', 'fixed',
                             'knn_adaptive', 'knn_shell'],
                    help=('support 반경 정의. 앞의 셋은 smooth·invariant(tie-safe), '
                          'knn_*는 anchor별 k번째 거리에 의존하는 비교군'))
    pw.add_argument('--pw-radius-alpha', type=float, default=0.75,
                    help='global_scale/density_scaled의 반경 계수')
    pw.add_argument('--pw-radius', type=float, default=None,
                    help="radius-mode=fixed의 물리 반경")
    pw.add_argument('--pw-support-k', type=int, default=8,
                    help='knn_adaptive / knn_shell의 support 이웃 수')
    pw.add_argument('--pw-target-k', type=int, default=16,
                    help='density_scaled가 맞추려는 평균 degree')
    pw.add_argument('--pw-tie-eps', type=float, default=0.0,
                    help='knn_shell에서 동일 shell을 포함시키는 거리 여유')
    pw.add_argument('--pw-rbf', type=int, default=8,
                    help='edge invariant의 radial basis 개수')
    pw.add_argument('--pw-pool', default='basis_mean',
                    choices=['basis', 'basis_mean', 'attention', 'sum', 'mean'],
                    help=('neighbor 집합 -> channel 축약 방식. basis는 파라미터 없는 '
                          '고정 거리 shell (뒤따르는 LNLinear가 span 안의 임의 radial '
                          'kernel을 복원한다)'))
    pw.add_argument('--pw-bracket', default='none',
                    choices=['none', 'separable', 'pairwise'],
                    help=('pooling 단계의 bracket 채널. separable은 O(k), '
                          'pairwise는 O(k^2)이지만 국소 대칭에서도 살아남는다'))
    pw.add_argument('--pw-bracket-channels', type=int, default=None,
                    help='bracket 채널 수 (기본: C_0와 동일)')
    pw.add_argument('--pw-no-bracket-layers', action='store_true',
                    help='backbone에서 covector bracket 제거 (ablation)')
    pw.add_argument('--pw-gate', default='projected',
                    choices=['none', 'projected', 'full'],
                    help='Klein-form gate; projected는 O(P), full은 O(C^2) Gram')
    pw.add_argument('--pw-no-global-context', action='store_true',
                    help='gate/head에서 global invariant scalar context 제거')
    pw.add_argument('--pw-message-passing', action='store_true',
                    help='block마다 invariant-weighted message passing 추가')
    pw.add_argument('--pw-msg-channels', type=int, default=8,
                    help='message passing 채널 수 (메모리는 B*C*6*N*k)')
    pw.add_argument('--pw-hidden', type=int, default=32,
                    help='scalar MLP hidden 폭')
    pw.add_argument('--pw-proj', type=int, default=8,
                    help='gate/head invariant projection 개수 P')
    pw.add_argument('--pw-normalize', default='nh',
                    choices=['nh', 'beta', 'one'],
                    help='second moment 정규화 Z(P)')
    pw.add_argument('--pw-beta', default='learned',
                    choices=['learned', 'uniform'],
                    help='factor별 positive weight beta_ih')
    pw.add_argument('--pw-force-invariant', action='store_true',
                    help='gate/head 불변량에 f_c . f_d 계열 추가')
    ap.add_argument('--data-seed', type=int, default=100)
    ap.add_argument('--model-seed', type=int, default=0)
    ap.add_argument('--device',
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--wandb-mode', default='online',
                    choices=['online', 'offline', 'disabled'])
    ap.add_argument('--wandb-project', default=WANDB_PROJECT)
    ap.add_argument('--wandb-entity', default=WANDB_ENTITY)
    ap.add_argument('--suite', action='store_true',
                    help='표준 그리드 실행 (각각 별도 wandb run)')
    ap.add_argument('--quick', action='store_true')
    ap.set_defaults(**RECIPES[pre_args.recipe])
    args = ap.parse_args()
    if args.kernel_candidates is None:
        args.kernel_candidates = 4 * args.k_enc
    if args.kernel_candidates < 2:
        ap.error('--kernel-candidates는 2 이상이어야 합니다')
    if args.quick:
        args.epochs, args.n_train, args.n_val = 10, 128, 64

    if args.suite:
        grid = ([('centro', {'eta': e}) for e in (0.0, 0.02, 0.1, 0.5)]
                + [('c2', {'eta': 0.0}), ('tetra', {'eta': 0.0})]
                + [('iid', {'n_points': n}) for n in (32, 128, 512)])
        for sc, over in grid:
            sub = argparse.Namespace(**{**vars(args), 'dataset': sc,
                                        'suite': False, **over})
            print(f'\n===== suite: {sc} {over} =====')
            one(sub)
        return
    one(args)


def one(args):
    if args.encoder == 'tensor':
        if args.method != 'covector':
            raise ValueError('--encoder tensor에는 --method covector가 필요합니다')
        if args.nonlinear != 'bracket':
            raise ValueError('tensor 모델의 LN ablation은 covector bracket만 지원합니다')
    if args.encoder == 'pointwise':
        # 파이프라인 전체가 se(3)* 안에서만 동작한다 (Q가 등장하지 않는다).
        if args.method != 'covector':
            raise ValueError('--encoder pointwise에는 --method covector가 필요합니다')
        if args.nonlinear != 'bracket':
            raise ValueError('pointwise 백본의 비선형성은 --pw-bracket / --pw-gate로 지정합니다')
    if args.target_graph == 'teacher' and args.encoder != 'pointwise':
        raise ValueError('--target-graph teacher는 --encoder pointwise 전용입니다')
    if args.dataset == 'tetra' and args.n_points % 12 != 0:
        adj = max(12, args.n_points // 12 * 12)
        print(f'[tetra] n_points {args.n_points} -> {adj} (12의 배수 보정)')
        args.n_points = adj
    tag = (f'{args.dataset}-{args.method}-{args.nonlinear}-{args.recipe}'
           + (f'-{args.encoder}' if args.encoder != 'plueck' else '')
           + (f'-{args.tensor_graph}-{args.tensor_weight}'
              if args.encoder == 'tensor' else '')
           + (f'-backbone-{args.tensor_backbone}'
              if args.encoder == 'tensor' else '')
           + (f'-{args.pw_radius_mode}-{args.pw_pool}-br{args.pw_bracket}'
              f'-gate{args.pw_gate}' + ('-mp' if args.pw_message_passing else '')
              if args.encoder == 'pointwise' else '')
           + (f'-target-{args.target_graph}'
              if args.target_graph != 'knn' else '')
           + (f'-eta{args.eta}' if args.dataset != 'iid' else
              f'-N{args.n_points}'))
    wb = init_wandb(tag, vars(args), mode=args.wandb_mode,
                    project=args.wandb_project, entity=args.wandb_entity)
    try:
        if args.dataset == 'fiber':
            run_fiber(args, wb)
        else:
            run_training(args, wb)
    finally:
        if wb: wb.finish()


if __name__ == '__main__':
    main()

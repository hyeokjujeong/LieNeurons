"""[CURRENT ENTRY POINT]
Blockage failure-scenario benchmark with wandb monitoring.

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
  # 디스크 데이터셋 + 저장 라벨.  경로 하나로 peg-and-hole 샤드 데이터셋과
  # 직접 준비한 (cloud, K) 파일을 모두 받는다 (형식은 경로를 보고 판별).
  python experiment/pc_se3_congruence/blockage_bench.py \
      --data-path data/peg_hole/v2 --n-points 512 \
      --encoder pointwise --method covector --target-graph stored
  python experiment/pc_se3_congruence/blockage_bench.py \
      --data-path mydata.npz \
      --encoder pointwise --method covector --target-graph stored
  ... --wandb-mode disabled  (wandb 없이 stdout만)
"""
import argparse
import sys

sys.path.append('.')

import torch

torch.set_default_dtype(torch.float64)

from data_loader.pc_stiffness_data_loader import load_split
from experiment.pc_se3_congruence.data_synth import (c2_clouds,
                                                     contact_spring_all_pairs_K,
                                                     object_clouds,
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
                                                     airm_scale_shape,
                                                     block_metrics, f_signal,
                                                     group_metrics, init_wandb)
from experiment.pc_se3_congruence.peg_hole_synth import STAGES
from experiment.pc_se3_congruence.models import (DualBackbone, GateBackbone,
                                                   ModelB, ModelPC2K,
                                                   WrenchSecondMomentModel)
from experiment.pc_se3_congruence.pointwise_models import (
    PointwiseStiffnessModel, bounded_invariant, force_pair, klein_pair)
from experiment.pc_se3_congruence.spd_loss import (EIG_CLAMP,
                                                     affine_invariant_d)

DATASETS = {
    'centro': symmetric_clouds,
    'c2': c2_clouds,
    'tetra': tetra_orbit_clouds,
    'iid': lambda n, npts, gen, **kw: sample_clouds(n, npts, gen, trans_scale=1.0),
    'objects': object_clouds,
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


def _merge_graph_stats(chunks):
    """Combine per-chunk graph stats into one set of numbers.

    Degrees and truncation are size-weighted means (they are per-point rates);
    the max/required capacities are maxima, because a single chunk that needed
    a larger candidate budget is the one that would break set-equivariance.
    """
    if not chunks:
        return {}
    tot = sum(n for _, n in chunks)
    out = {}
    for key in chunks[0][0]:
        vals = [(s[key], n) for s, n in chunks]
        out[key] = (max(v for v, _ in vals)
                    if key in ('graph_max_degree', 'graph_required_candidate_k',
                               'graph_candidate_k')
                    else sum(v * n for v, n in vals) / tot)
    return out


def eval_forward(args, model, P, chunk):
    """Validation forward in chunks.

    The graph builder materializes a [B, N, N] distance matrix, so an
    unchunked val pass is O(n_val * N^2) -- at n_val=2048, N=1024 that is a
    16 GiB allocation, well past a 24 GB card, even though the model itself is
    tiny.  Chunking here does not change any number, only peak memory.
    """
    preds, stats = [], []
    for i in range(0, P.shape[0], chunk):
        preds.append(forward_K(args.method, model, P[i:i + chunk]))
        s = getattr(model, 'last_graph_stats', None)
        if s:
            stats.append((dict(s), preds[-1].shape[0]))
    return torch.cat(preds), _merge_graph_stats(stats)


def head_diagnostics(model, P, chunk=64, n=256):
    """Is the head's invariant scalar pathway actually alive?

    K = (e^g / NH) sum beta_ih z_ih z_ih^T, and beta is the ONLY per-scene
    handle on the magnitude (e^g is global).  beta is driven by invariants of
    the latent covectors, so if those invariants are identically zero the whole
    magnitude pathway degenerates to a constant and d_scale can never beat a
    constant predictor -- silently, because nothing else looks wrong.

    That is exactly what the Klein pairing does here: the encoder anchors every
    neighbour wrench at the same point, so m_c = p x f_c holds through every
    layer (LNLinear and the covector bracket both preserve it), and then
    <X_c, X_c'> = f_c.(p x f_c') + (p x f_c).f_c' = 0 identically -- the two
    triple products are the same determinant with two rows swapped.  Every
    latent feature is a ZERO-PITCH wrench and the Klein form measures pitch.

    beta_scene_std == 0 is the signature.  force_pair (f_a . f_b, enabled by
    --pw-force-invariant) does not vanish on zero-pitch features.
    """
    h = getattr(model, 'head', None)
    if h is None or getattr(h, 'weight_mode', None) != 'learned':
        return {}
    bs, iv = [], []
    with torch.no_grad():
        for i in range(0, min(n, P.shape[0]), chunk):
            X = model.features(P[i:i + chunk])
            bs.append(h.weights(X).mean(dim=(1, 2)))
            u, v = h.proj_u(X), h.proj_v(X)
            s = [bounded_invariant(klein_pair(u, v))]
            if getattr(h, 'use_force', False):
                s.append(bounded_invariant(force_pair(u, v)))
            iv.append(torch.cat(s, 1).abs().mean(dim=(1, 2)))
    bs, iv = torch.cat(bs), torch.cat(iv)
    return {'beta_scene_std': bs.std().item(), 'beta_mean': bs.mean().item(),
            'head_inv_abs': iv.mean().item()}


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


def load_reference_levels(args):
    """The two brackets every val_d has to be read between (peghole_baseline.py).

    baseline_d : best CONSTANT predictor under AIRM.  val_d above it means the
                 model learned nothing about the cloud.
    mc_noise_d : distance between two subsample draws of the SAME scene, i.e.
                 the resolution of the label itself.  val_d below it means one
                 particular draw was memorised.

    Only meaningful against the stored labels -- the teacher target is a frozen
    network, for which neither number says anything, so they are not attached.
    """
    if not args.baseline_json or args.target_graph != 'stored':
        return {}
    import json
    with open(args.baseline_json) as f:
        b = json.load(f)
    out = {'baseline_d': b['frechet_train_fit_on_val']['all']['mean']}
    if 'mc_noise' in b:
        out['mc_noise_d'] = b['mc_noise']['all']['mean']
    print(f'[reference] baseline d {out["baseline_d"]:.3f}'
          + (f'   MC label noise d {out["mc_noise_d"]:.3f}'
             if 'mc_noise_d' in out else '')
          + f'   ({args.baseline_json})')
    return out


def run_training(args, wb):
    K_stored, stage_va = None, None
    if args.data_path:
        # On-disk dataset.  load_split picks the format from the path, so
        # peg-and-hole and bring-your-own take the same branch here.
        kw = dict(seed=args.data_seed, n_points=args.n_points,
                  val_frac=args.val_frac, lambda_body=args.lambda_body,
                  relabel=not args.no_relabel, device=args.device)
        # 0 이하는 "있는 만큼 전부" — 데이터셋 크기를 미리 모를 때 쓴다.
        n_tr = args.n_train if args.n_train > 0 else None
        n_va = args.n_val if args.n_val > 0 else None
        P_tr_, K_tr_, _ = load_split(args.data_path, 'train', n=n_tr, **kw)
        # info 는 val 에서만 쓴다: stage id 가 어느 구간이 실패하는지 국소화하며
        # (free 는 접촉이 아예 없고 insert 는 접촉 지배), 뭉뚱그린 val_d 는 이를
        # 가린다.  이 정보가 없는 형식이면 그냥 비어 있다.
        P_va_, K_va_, info_va = load_split(args.data_path, 'val', n=n_va, **kw)
        if 'stage' in info_va:
            stage_va = info_va['stage'].to(args.device)
        P = torch.cat([P_tr_, P_va_]).to(args.device)
        K_stored = torch.cat([K_tr_, K_va_]).to(args.device)
        # 아래 학습 루프는 args.n_train/n_val 로 이 텐서를 다시 자른다.  "있는
        # 만큼 전부"(0) 를 그대로 두면 train 슬라이스가 비어 배치가 0개가 된다.
        args.n_train, args.n_val = P_tr_.shape[0], P_va_.shape[0]
        print(f'[data] {args.data_path}  N={P.shape[1]}  '
              f'train/val={args.n_train}/{args.n_val}')
    else:
        gen = torch.Generator().manual_seed(args.data_seed)
        make = DATASETS[args.dataset]
        n_total = args.n_train + args.n_val
        if args.dataset == 'objects':
            P = make(n_total, args.n_points, gen, shape=args.shape)
        elif args.dataset != 'iid':
            P = make(n_total, args.n_points, gen, eta=args.eta)
        else:
            P = make(n_total, args.n_points, gen)
        P = P.to(args.device)
    if args.target_graph == 'stored':
        target_fn = None
    elif args.target_graph == 'all':
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
    if target_fn is None:
        K_gt = K_stored
    else:
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

    ref = load_reference_levels(args)
    if wb:
        wb.summary.update(ref)
        try:            # best val over the run, without a manual pass over logs
            wb.define_metric('val_d', summary='min')
        except Exception:
            pass
    best = {'val_d': float('inf'), 'epoch': -1}

    S = P_tr.shape[0]
    gp = torch.Generator().manual_seed(args.data_seed + 1)
    for ep in range(epochs):
        perm = torch.randperm(S, generator=gp)
        tot, nb, ncl, gtot = 0.0, 0, 0, 0.0
        for b in range(0, S, args.batch):
            i = perm[b:b + args.batch]
            d, n_c = affine_invariant_d(L_tr[i], forward_K(args.method, model, P_tr[i]))
            loss = d.mean()
            if opt is not None:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                # pre-clip norm: the direct evidence for "is this an
                # optimisation problem?" when the teacher phase does not reach 0
                gtot += float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip))
                opt.step()
            tot, nb, ncl = tot + loss.item(), nb + 1, ncl + n_c
        if sched is not None:
            sched.step()

        model.eval()
        with torch.no_grad():
            # Stats come back from eval_forward rather than being read off the
            # model afterwards: f_signal below runs another forward on a subset
            # and would overwrite last_graph_stats with that subset's graph.
            K_pred, graph_stats = eval_forward(args, model, P_va,
                                               args.eval_batch)
            d_va, _ = affine_invariant_d(L_va, K_pred)
        log = {'train_d': tot / nb, 'val_d': d_va.mean().item(),
               'clamped': ncl, 'grad_norm': gtot / nb,
               'gap': d_va.mean().item() - tot / nb,
               'lr': sched.get_last_lr()[0] if sched is not None else 0.0,
               **airm_scale_shape(L_va, K_pred),
               **head_diagnostics(model, P_va),
               **ref,
               'f_signal': f_signal(model, P_va[:64],
                                    slice(3, 6) if args.method == 'covector'
                                    else slice(0, 3)),   # covector:[m;f] / vector:[w;v]
               **block_metrics(K_pred, K_va),
               # graph_truncation_frac > 0 이면 support 안의 이웃이 top-k에서
               # 잘려나갔다는 뜻 — set-equivariance 보장이 깨진다.
               **graph_stats,
               **(group_metrics(K_pred, K_va, d_va, stage_va, STAGES)
                  if stage_va is not None else {})}
        # The headline number of the whole experiment: an absolute val_d says
        # nothing, the ratio to the best constant predictor does.
        if 'baseline_d' in ref:
            log['val_d_rel'] = log['val_d'] / ref['baseline_d']
        if log['val_d'] < best['val_d']:
            best = {'val_d': log['val_d'], 'epoch': ep}
            if args.ckpt_out:
                torch.save({'state_dict': model.state_dict(), 'epoch': ep,
                            'val_d': log['val_d'], 'args': vars(args)},
                           args.ckpt_out)
        model.train()
        if wb: wb.log(log, step=ep)
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            extra = (f'  deg {log["graph_mean_degree"]:.1f}'
                     f'  trunc {log["graph_truncation_frac"]:.3f}'
                     if 'graph_mean_degree' in log else '')
            rel = (f'  rel {log["val_d_rel"]:.3f}' if 'val_d_rel' in log
                   else '')
            print(f'ep {ep:4d}  train d {log["train_d"]:8.3f}  '
                  f'val d {log["val_d"]:8.3f}{rel}  '
                  f'scale {log["d_scale"]:6.3f}  shape {log["d_shape"]:6.3f}  '
                  f'mm {log["err_rel_mm"]:.3f}  '
                  f'ff {log["err_rel_ff"]:.3f}  rank {log["rank_pred"]:.1f}  '
                  f'clamp {log["clamped"]:5d}  |g| {log["grad_norm"]:.2e}  '
                  + (f'βσ {log["beta_scene_std"]:.2e}  '
                     f'inv {log["head_inv_abs"]:.2e}  '
                     if 'beta_scene_std' in log else '')
                  + f'|f_c| {log["f_signal"]:.4f}{extra}', flush=True)
            if stage_va is not None:
                print('            stage  ' + '  '.join(
                    f'{s}: d {log[f"{s}/val_d"]:6.3f} mm {log[f"{s}/err_rel_mm"]:.3f}'
                    f' ff {log[f"{s}/err_rel_ff"]:.3f}'
                    for s in STAGES if log.get(f'{s}/n')), flush=True)
    eq1 = equiv_check(args, model, P_va[:32])
    print(f'학습 후 equivariance: {eq1:.2e}')
    rel_best = (f'  (기준선 대비 {best["val_d"] / ref["baseline_d"]:.3f})'
                if 'baseline_d' in ref else '')
    print(f'최저 val d {best["val_d"]:.4f} @ ep {best["epoch"]}{rel_best}'
          + (f'  -> {args.ckpt_out}' if args.ckpt_out else ''))
    if wb:
        wb.summary['equiv_err_final'] = eq1
        wb.summary['best_val_d'] = best['val_d']
        wb.summary['best_epoch'] = best['epoch']
        if 'baseline_d' in ref:
            wb.summary['best_val_d_rel'] = best['val_d'] / ref['baseline_d']
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

    ap = argparse.ArgumentParser(
        parents=[pre],
        description=(
            '등변 강성 학습 벤치. 하나의 드라이버가 여러 실험을 겸하므로 '
            '플래그가 많다 — 아래 그룹 중 자기 실험에 해당하는 것만 보면 된다.'),
        epilog=(
            '자주 쓰는 조합:\n'
            '  디스크 데이터셋 (권장 진입점: train.py)\n'
            '    --data-path <경로> --n-points 1024\n'
            '    --encoder pointwise --method covector --target-graph stored\n'
            '  합성 대칭 클라우드 스위트\n'
            '    --suite --encoder pointwise --method covector\n'),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ph = ap.add_argument_group('디스크 데이터셋 (--data-path)')
    ds = ap.add_argument_group('합성 데이터셋과 타깃')
    md = ap.add_argument_group('모델 구성')
    ts = ap.add_argument_group('tensor 인코더 전용 (--encoder tensor)')
    tr = ap.add_argument_group('학습과 실행')
    wb = ap.add_argument_group('wandb')
    ap.add_argument('--dataset', default='centro',
                    choices=list(DATASETS) + ['fiber'],
                    help=('그 자리에서 생성하는 합성 cloud. --data-path 를 주면 '
                          '무시되고 디스크에서 읽는다'))
    ph.add_argument('--data-path',
                    help=('디스크 데이터셋 경로. meta.json 이 있으면 peg-and-hole '
                          '샤드 데이터셋이고, 아니면 points [S,N,3] 와 K [S,6,6] '
                          '이 든 .npz/.pt 파일(또는 train/val 이 든 디렉터리)로 '
                          '읽는다. 형식과 K 의 블록 순서는 '
                          'data_loader/pc_stiffness_data_loader.py 참조. '
                          '--target-graph stored 와 함께 쓴다'))
    ph.add_argument('--val-frac', type=float, default=0.1,
                    help=('경로가 파일 하나일 때 val 로 뗄 비율 (디렉터리나 '
                          'peg-hole 데이터셋이면 무시된다)'))
    ph.add_argument('--no-relabel', action='store_true',
                    help=('서브샘플에서 라벨을 재계산하지 않고 저장 라벨(full '
                          'cloud 기준)을 그대로 쓴다. 회귀가 ill-posed해지므로 '
                          '비교 목적으로만 사용 (peg-hole 전용)'))
    ph.add_argument('--lambda-body', type=float, default=None,
                    help=('peg-hole 타깃 K = K_contact + lambda*K_body의 '
                          'lambda (기본: meta.json의 캘리브레이션 값)'))
    md.add_argument('--encoder', default='plueck',
                    choices=['plueck', 'learnable', 'bracket', 'tensor',
                             'pointwise'],
                    help=('plueck: 기존 global mean / learnable: VN lift 뒤 global mean / '
                          'bracket: pooling-전 bracket / tensor: edge를 유지해 ww^T late pooling / '
                          'pointwise: set pooling + point축 유지 LN backbone + late second moment'))
    md.add_argument('--nonlinear', default='bracket',
                    choices=['bracket', 'gate', 'dual'],
                    help='백본 비선형성: bracket(기존) / gate(Gram gate로 교체) / dual(병렬 두 브랜치+bracket 병합)')
    md.add_argument('--method', default='vector', choices=['vector', 'covector'],
                    help='vector: twist(adjoint) 입력 + Klein head / covector: wrench(coadjoint) 입력')
    ds.add_argument('--eta', type=float, default=0.0)
    tr.add_argument('--epochs', type=int, default=150)
    tr.add_argument('--batch', type=int, default=32)
    tr.add_argument('--lr', type=float, default=3e-4)
    tr.add_argument('--grad-clip', type=float, default=1.0)
    ds.add_argument('--n-train', type=int, default=512)
    ds.add_argument('--n-val', type=int, default=128)
    ds.add_argument('--n-points', type=int, default=None,
                    help=('cloud 해상도. 생성 데이터셋은 만들 점 개수(기본 64), '
                          '디스크 데이터셋은 서브샘플 목표(기본: 저장된 그대로). '
                          '서브샘플은 라벨을 재계산할 수 있는 형식에서만 된다'))
    md.add_argument('--k-enc', type=int, default=8)
    ds.add_argument('--k-gt', type=int, default=8)
    ds.add_argument('--sigma-k', type=float, default=0.5)
    ds.add_argument('--target-graph', default='knn',
                    choices=['knn', 'kernel', 'all', 'teacher', 'stored'],
                    help=('GT contact spring graph; kernel은 local compact window, '
                          'all은 exact-tie 대조군, teacher는 같은 클래스의 고정 '
                          '난수 모델(realizability 검증, pointwise 전용), '
                          'stored는 peghole 데이터셋의 저장 라벨 '
                          'K_contact + lambda*K_body (peghole 전용)'))
    ds.add_argument('--teacher-seed', type=int, default=7,
                    help='--target-graph teacher의 고정 teacher 모델 seed')
    ds.add_argument('--target-batch', type=int, default=64,
                    help='GT 생성 chunk 크기 (all-pairs 메모리 제어)')
    tr.add_argument('--eval-batch', type=int, default=64,
                    help=('검증 forward chunk 크기. 그래프가 [B,N,N] 거리행렬을 '
                          '만들므로 n_val 전체를 한 번에 돌리면 N=1024, '
                          'n_val=2048에서 16 GiB를 요구한다 (숫자는 불변, '
                          '최대 메모리만 달라진다)'))
    md.add_argument('--lift-hidden', type=int, default=8,
                    help='learnable VN lift의 hidden channel 수')
    ts.add_argument('--tensor-graph', default='all',
                    choices=['knn', 'kernel', 'all'],
                    help=('tensor edge graph; kernel은 bounded local 후보와 '
                          'smooth zero-boundary window 사용'))
    ts.add_argument('--kernel-candidates', type=int, default=None,
                    help=('kernel graph의 최대 local 후보 수; 기본값은 4*k-enc. '
                          'N-1보다 크면 자동으로 줄임'))
    ts.add_argument('--tensor-weight', default='learned',
                    choices=['learned', 'analytic', 'uniform'],
                    help='2차 pooling의 invariant radial weight')
    ts.add_argument('--tensor-hidden', type=int, default=32,
                    help='learned tensor radial MLP hidden channel 수')
    ts.add_argument('--tensor-backbone', default='none',
                    choices=['none', 'covector'],
                    help=('none: radial second moment 직접 pooling; covector: '
                          'pooling 전에 LNLinear+covector-bracket stack 적용'))
    ts.add_argument('--tensor-backbone-channels', type=int, nargs='+',
                    default=[32, 32, 16],
                    help='tensor covector backbone의 hidden/output channel 수')
    ds.add_argument('--shape', default='mixed',
                    help='objects 데이터셋의 형태: mixed/box/cylinder/mug/lbracket/bowl')
    md.add_argument('--channels', type=int, nargs='+', default=[8, 32, 32, 16])

    # ------------------------------------------------ encoder=pointwise 전용
    # 축 규약: N=point, k=neighbor(집합), C=latent channel, H=factor, K=6x6 강성
    pw = ap.add_argument_group('pointwise 파이프라인 전용 (--encoder pointwise)')
    pw.add_argument('--pw-channels', type=int, nargs='+', default=[8, 16, 32, 16],
                    help='[C_0(set pooling), C_1, ...]; 권장 8-16-32-16')
    pw.add_argument('--pw-factors', type=int, default=8, help='factor 채널 H')
    pw.add_argument('--pw-candidates', type=int, default=64,
                    help=('point당 후보 이웃 수 k — 모델 파라미터가 아니라 메모리 '
                          '예산이며 support를 덮기만 하면 된다. 부족하면 로그의 '
                          'graph_required_candidate_k가 필요한 값을 알려준다'))
    pw.add_argument('--pw-radius-mode', default='degree_matched',
                    choices=['degree_matched', 'global_scale',
                             'density_scaled', 'fixed', 'knn_adaptive',
                             'knn_shell'],
                    help=('support 반경 정의. degree_matched(기본)는 평균 degree를 '
                          'target_k로 고정하는 닫힌 형태 분위수 반경 — 부피/표면/곡선 '
                          '어느 분포에서도 동일하게 동작한다. global_scale은 N에 따라 '
                          'degree가 발산하고 density_scaled는 내재차원 3을 가정한다. '
                          'knn_*는 anchor별 k번째 거리에 의존하는 비교군'))
    pw.add_argument('--pw-radius-alpha', type=float, default=None,
                    help=('반경 계수 (기본: 모드별 기본값 — degree_matched는 1.0, '
                          '나머지는 0.75)'))
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
    tr.add_argument('--baseline-json',
                    help=('peghole_baseline.py가 낸 json. Frechet 기준선과 라벨 '
                          'MC 잡음을 읽어 val_d_rel(기준선 대비 비율)을 '
                          '지표로 만든다. --target-graph stored에만 적용'))
    tr.add_argument('--ckpt-out',
                    help='val_d 최저점의 state_dict 저장 경로 (없으면 저장 안 함)')
    ds.add_argument('--data-seed', type=int, default=100)
    md.add_argument('--model-seed', type=int, default=0)
    tr.add_argument('--device',
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    wb.add_argument('--wandb-mode', default='online',
                    choices=['online', 'offline', 'disabled'])
    wb.add_argument('--wandb-project', default=WANDB_PROJECT)
    wb.add_argument('--wandb-entity', default=WANDB_ENTITY)
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
    if args.target_graph == 'stored' and not args.data_path:
        raise ValueError('--target-graph stored는 --data-path 가 필요합니다 '
                         '(라벨을 디스크에서 읽는 경로)')
    if args.data_path and args.target_graph in ('knn', 'all'):
        print('[data] 경고: 저장 라벨 대신 on-the-fly '
              f'{args.target_graph} 타깃을 사용합니다 (대조군 용도)')
    if args.n_points is None:
        # 생성 데이터셋은 크기를 정해 줘야 하고, 디스크 데이터셋은 저장된
        # 해상도를 그대로 쓴다 (None 이 곧 "건드리지 않음").
        if not args.data_path:
            args.n_points = 64
    elif args.dataset == 'tetra' and args.n_points % 12 != 0:
        adj = max(12, args.n_points // 12 * 12)
        print(f'[tetra] n_points {args.n_points} -> {adj} (12의 배수 보정)')
        args.n_points = adj
    tag = ((f'{args.dataset}-{args.shape}' if args.dataset == 'objects'
            else args.dataset)
           + f'-{args.method}-{args.nonlinear}-{args.recipe}'
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
           + (f'-N{args.n_points or "full"}'
              if args.data_path else
              f'-eta{args.eta}' if args.dataset != 'iid' else
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

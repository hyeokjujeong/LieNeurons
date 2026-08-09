#!/usr/bin/env bash
# peg-and-hole 학습 3단계.  기본값이 곧 본학습 설정이라 인자 없이 그냥 돌리면 된다.
#
#   conda activate lieneurons && cd <repo 루트>
#   bash experiment/pc_se3_congruence/run_peghole_experiments.sh
#
# 5-6 시간짜리다.  ssh 가 끊겨도 살아남게 하려면
#   nohup bash experiment/pc_se3_congruence/run_peghole_experiments.sh \
#       > peghole_main.log 2>&1 &
#   tail -f peghole_main.log
#
# 단계 (인자로 일부만 선택 가능: ... run_peghole_experiments.sh teacher stored)
#   baseline  학습 없음. Frechet 평균 기준선(train 적합 -> val 평가)과 라벨의
#             MC 분산.  이후 두 단계의 val_d 는 반드시 이 두 값과 함께 읽는다.
#             서브샘플 재라벨 캐시도 여기서 데워진다.
#   teacher   REALIZABILITY.  타깃이 같은 클래스의 고정 난수 모델이므로 정답
#             가중치가 존재한다.  여기서 0 근처로 가지 않으면 최적화 문제이고,
#             그러면 stored 결과는 해석할 수 없다.
#   stored    데이터셋의 저장 라벨 K_contact + lambda*K_body.
#   compare   stored 타깃으로 두 팔을 GPU 두 장에 나눠 동시에 돌린다.
#             ARM_A(기본 없음) vs ARM_B(기본 --pw-force-invariant).
#             크기 변조 경로(beta)가 살아나는지 보는 짧은 대조 실험이다.
#
# 그래프 플래그는 일부러 주지 않는다 — 기본값 degree_matched 가 이 데이터에서
# degree ~16, truncation 0 을 준다 (구조 지표가 깨지면 나머지는 볼 필요 없음).
#
# 환경변수
#   ROOT NPTS NTRAIN NVAL EPOCHS BATCH LR CH FAC SEED
#   DEV        기본 학습 장치
#   DEV2       PARALLEL=1 일 때 stored 를 올릴 두 번째 장치
#   PARALLEL   1 이면 teacher(DEV) 와 stored(DEV2) 를 동시에 실행
#   WANDB_MODE online | offline | disabled
#   OUTDIR     로그와 json 출력 경로
#   BASELINE_JSON  기준선 json 경로 (기본: 이번 실행의 baseline 단계 출력).
#                  학습 단계가 이걸 읽어 val_d_rel(기준선 대비 비율)을 기록한다
#   QUICK      1 이면 smoke 설정으로 덮어쓴다 (10 epoch, 소량 표본)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ROOT="${ROOT:-data/peg_hole/v2}"
NPTS="${NPTS:-1024}"
NTRAIN="${NTRAIN:-20480}"
NVAL="${NVAL:-2048}"
EPOCHS="${EPOCHS:-150}"
BATCH="${BATCH:-32}"
# 검증/타깃 생성 chunk.  그래프가 [B,N,N] 거리행렬을 만들기 때문에 이 값이
# 최대 GPU 메모리를 지배한다 (N=1024, chunk=64 -> 537 MB).
EVALBATCH="${EVALBATCH:-64}"
LR="${LR:-1e-3}"
CH="${CH:-16 32 64 32}"
FAC="${FAC:-16}"
SEED="${SEED:-100}"
DEV="${DEV:-cuda:0}"
DEV2="${DEV2:-cuda:1}"
PARALLEL="${PARALLEL:-1}"
WANDB_MODE="${WANDB_MODE:-online}"
OUTDIR="${OUTDIR:-experiment/pc_se3_congruence/train_results/peghole}"
QUICK="${QUICK:-0}"
MC_N="${MC_N:-512}"

if [[ "${QUICK}" == "1" ]]; then
    NPTS="${NPTS}"; NTRAIN=64; NVAL=32; EPOCHS=2; BATCH=8; MC_N=32
    echo "[quick] smoke 설정: NTRAIN=${NTRAIN} NVAL=${NVAL} EPOCHS=${EPOCHS}"
fi

PHASES=("$@")
if [[ ${#PHASES[@]} -eq 0 ]]; then
    PHASES=(baseline teacher stored)
fi
for p in "${PHASES[@]}"; do
    case "${p}" in
        baseline|teacher|stored|compare) ;;
        *) echo "알 수 없는 단계 '${p}' (baseline|teacher|stored|compare)" >&2
           exit 2 ;;
    esac
done

# compare 단계의 팔들.  'skip' 으로 두면 그 팔은 돌리지 않는다.
#   A  대조군 — 현행 그대로
#   B  처치군 — Klein 대신 죽지 않는 불변량(f_a . f_b)을 gate/beta 에 공급
#   C  제거군 — gate 와 beta 를 통째로 뺀다.  B 의 진단이 맞다면 gate 출력은
#      채널별 상수이고 그건 바로 뒤의 LNLinear 가, beta 상수는 e^g 가 흡수하므로
#      C 는 A 와 거의 같아야 한다.  달라지면 진단이 틀린 것이다.
ARM_A="${ARM_A:-}"
ARM_B="${ARM_B:---pw-force-invariant}"
ARM_C="${ARM_C:---pw-gate none --pw-beta uniform}"

# ------------------------------------------------------- 사전 점검 (빨리 실패)
# 5시간 뒤가 아니라 지금 실패하는 편이 낫다.
echo "===== 사전 점검 ====="
echo "  conda env : ${CONDA_DEFAULT_ENV:-<없음>}"
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "  [중단] conda 환경이 활성화되지 않았다.  conda activate lieneurons" >&2
    exit 1
fi

python - "${DEV}" "${DEV2}" "${PARALLEL}" <<'PY'
import sys
import torch
dev, dev2, parallel = sys.argv[1], sys.argv[2], sys.argv[3] == '1'
print(f'  torch     : {torch.__version__}  cuda {torch.version.cuda}')
for d in [dev] + ([dev2] if parallel else []):
    if not d.startswith('cuda'):
        continue
    if not torch.cuda.is_available():
        raise SystemExit('  [중단] CUDA 를 쓸 수 없다')
    i = int(d.split(':')[1]) if ':' in d else 0
    if i >= torch.cuda.device_count():
        raise SystemExit(f'  [중단] {d} 없음 (보이는 GPU {torch.cuda.device_count()}장)')
    free, total = torch.cuda.mem_get_info(i)
    print(f'  {d}      : {torch.cuda.get_device_name(i)}  '
          f'여유 {free/2**30:.1f} / {total/2**30:.1f} GiB')
    if free / 2**30 < 6:
        raise SystemExit(f'  [중단] {d} 여유 메모리가 부족하다')
PY

python - "${NTRAIN}" "${NPTS}" "${EPOCHS}" "${PARALLEL}" <<'PY'
import sys
ntr, npts, ep = (int(x) for x in sys.argv[1:4])
parallel = sys.argv[4] == '1'
# 실측 기준점: N=1024, train 20480, batch 32, RTX 4090 에서 134 s/epoch.
# 비용은 O(N^2) 거리행렬이 지배하고 표본 수에 선형이다.
# 주의: torch 는 기본이 FASTEST_FIRST 라 cuda:0 이 nvidia-smi 의 0 이 아닐 수
# 있다.  실측에서 RTX 3090 은 4090 의 2.34 배 느렸다 (fp64 처리량 비율과 일치).
sec = 134 * (ntr / 20480) * (npts / 1024) ** 2 * ep
print(f'  예상 소요  : 빠른 카드 {sec/3600:.1f} 시간 / 느린 카드 '
      f'{sec*2.34/3600:.1f} 시간 (단계당)')
print(f'  {"병렬 -> 벽시계는 느린 쪽" if parallel else "순차 -> 합계"}: '
      f'약 {(sec*2.34 if parallel else sec*(1+2.34))/3600:.1f} 시간')
print('  (N=1024/train 20480/batch 32, RTX 4090 에서 134 s/epoch 실측 기준)')
PY

mkdir -p "${OUTDIR}"
STAMP="$(date +%Y%m%d-%H%M%S)"
TAG="N${NPTS}-tr${NTRAIN}-ep${EPOCHS}-${STAMP}"
# baseline 단계가 여기에 쓰고, 학습 단계가 여기서 읽어 val_d_rel 을 만든다.
# 이전 실행의 json 을 재사용하려면 BASELINE_JSON 을 직접 지정한다.
BASELINE_JSON="${BASELINE_JSON:-${OUTDIR}/baseline-${TAG}.json}"

echo "===== peg-and-hole 실험 ${TAG} ====="
echo "  root=${ROOT}  N=${NPTS}  train/val=${NTRAIN}/${NVAL}"
echo "  epochs=${EPOCHS} batch=${BATCH} lr=${LR} channels=[${CH}] factors=${FAC}"
echo "  device=${DEV}$([[ "${PARALLEL}" == "1" ]] && echo " + ${DEV2} (parallel)")"
echo "  phases=${PHASES[*]}  wandb=${WANDB_MODE}  out=${OUTDIR}"

python - "${ROOT}" <<'PY'
import json, sys
m = json.load(open(sys.argv[1] + '/meta.json'))
print(f"[dataset] version {m['version']}  stored N {m['n_points']}  "
      f"lambda {m['cfg']['lambda_body']}  "
      f"splits { {s: sum(x['n'] for x in e['shards']) for s, e in m['splits'].items()} }")
if m['version'] != 2:
    raise SystemExit('version 2 데이터셋이 아닙니다 -- 재생성이 필요합니다')
PY

has_phase() { for p in "${PHASES[@]}"; do [[ "${p}" == "$1" ]] && return 0; done; return 1; }

run_baseline() {
    echo -e "\n===== phase 0/2: 기준선 + 라벨 MC 분산 (학습 없음) ====="
    python experiment/pc_se3_congruence/peghole_baseline.py \
        --peghole-root "${ROOT}" \
        --n-points "${NPTS}" \
        --n-train "${NTRAIN}" \
        --n-val "${NVAL}" \
        --data-seed "${SEED}" \
        --mc-n "${MC_N}" \
        --device "${DEV}" \
        --out "${BASELINE_JSON}"
}

# blockage_bench 공통 인자.  --recipe 는 주지 않는다: full 레시피는 이 데이터에
# 무관한 합성-클라우드 하이퍼파라미터를 함께 덮어쓴다.
# bench <target> <device> <name> [blockage_bench 추가 플래그...]
bench() {
    local target="$1" dev="$2" name="$3"; shift 3
    # 기준선 json은 stored 타깃에만 의미가 있다 (teacher 타깃은 고정 난수 모델).
    # blockage_bench 쪽에서도 같은 조건으로 무시하므로 여기서는 존재만 확인한다.
    local ref=()
    [[ -f "${BASELINE_JSON}" ]] && ref=(--baseline-json "${BASELINE_JSON}")
    python experiment/pc_se3_congruence/blockage_bench.py \
        "${ref[@]}" \
        --ckpt-out "${OUTDIR}/${name}-${TAG}.pt" \
        --dataset peghole \
        --peghole-root "${ROOT}" \
        --peghole-n-points "${NPTS}" \
        --encoder pointwise \
        --method covector \
        --target-graph "${target}" \
        --n-train "${NTRAIN}" \
        --n-val "${NVAL}" \
        --epochs "${EPOCHS}" \
        --batch "${BATCH}" \
        --eval-batch "${EVALBATCH}" \
        --target-batch "${EVALBATCH}" \
        --lr "${LR}" \
        --pw-channels ${CH} \
        --pw-factors "${FAC}" \
        --data-seed "${SEED}" \
        --device "${dev}" \
        --wandb-mode "${WANDB_MODE}" \
        "$@"
}

# 재라벨 캐시를 미리 채운다.  두 학습을 병렬로 띄우면 둘 다 같은 캐시를 계산해
# 중복 작업이 되므로 (쓰기 자체는 tmp+rename 이라 안전) 먼저 한 번 데운다.
warm_cache() {
    echo -e "\n[cache] 서브샘플 재라벨 캐시 준비 (n${NPTS}_seed${SEED})"
    python - "${ROOT}" "${NPTS}" "${NTRAIN}" "${NVAL}" "${SEED}" "${DEV}" <<'PY'
import sys
sys.path.append('.')
from data_loader.peg_hole_data_loader import load_peg_hole_split
root, npts, ntr, nva, seed, dev = sys.argv[1:7]
for split, n in (('train', int(ntr)), ('val', int(nva))):
    P, K = load_peg_hole_split(root, split, n=n, n_points=int(npts),
                               seed=int(seed), device=dev)
    print(f'  {split}: P {tuple(P.shape)}  K {tuple(K.shape)}', flush=True)
PY
}

if has_phase baseline; then
    run_baseline
else
    if [[ ! -f "${BASELINE_JSON}" ]]; then
        echo -e "\n[경고] 기준선 json 이 없다 (${BASELINE_JSON}) — val_d_rel 을"
        echo "       기록하지 않는다. 이전 실행 것을 쓰려면 BASELINE_JSON 을 지정할 것."
    fi
    if has_phase compare \
       || { has_phase teacher && has_phase stored && [[ "${PARALLEL}" == "1" ]]; }
    then
        warm_cache
    fi
fi

if has_phase compare; then
    # 모든 팔은 데이터·seed·epoch·표본수가 같고 플래그만 다르다.
    # 배치: 첫 팔을 느린 카드(DEV2)에 하나만 걸고, 나머지를 빠른 카드(DEV)에
    # 순차로 쌓는다.  DEV 가 2.34 배 빠르므로 두 줄의 길이가 얼추 맞는다.
    names=(); flags=()
    for t in A B C; do
        eval "v=\${ARM_${t}}"
        [[ "${v}" == "skip" ]] && continue
        names+=("cmp${t}"); flags+=("${v}")
    done
    [[ ${#names[@]} -eq 0 ]] && { echo "compare: 돌릴 팔이 없다" >&2; exit 2; }

    run_arm() {   # run_arm <index>
        local i="$1" a=()
        read -r -a a <<< "${flags[$i]}"
        bench stored "$2" "${names[$i]}" ${a[@]+"${a[@]}"} \
            > "${OUTDIR}/${names[$i]}-${TAG}.log" 2>&1
    }

    echo -e "\n===== compare (${#names[@]} arms) ====="
    for i in "${!names[@]}"; do
        echo "  ${names[$i]} : ${flags[$i]:-<플래그 없음>}"
    done

    rcs=0
    if [[ "${PARALLEL}" == "1" && ${#names[@]} -gt 1 && "${DEV}" != "${DEV2}" ]]
    then
        echo "  배치: ${names[0]}@${DEV2}   나머지@${DEV} (순차)"
        run_arm 0 "${DEV2}" & slow_pid=$!
        for i in "${!names[@]}"; do
            [[ ${i} -eq 0 ]] && continue
            run_arm "${i}" "${DEV}" || rcs=1
        done
        wait "${slow_pid}" || rcs=1
    else
        echo "  배치: 전부 ${DEV} 순차"
        for i in "${!names[@]}"; do run_arm "${i}" "${DEV}" || rcs=1; done
    fi

    for n in "${names[@]}"; do
        echo -e "\n----- ${n} -----"
        grep -E "^ep |^최저|parameters|equivar" \
             "${OUTDIR}/${n}-${TAG}.log" | tail -n 6 \
            || tail -n 5 "${OUTDIR}/${n}-${TAG}.log"
    done
    cat <<'EOF'

[판정]
  1) βσ (beta_scene_std) 가 0 을 벗어났는가        -> 크기 변조 경로 부활 여부
  2) scale (d_scale) 이 1.677 아래로 갔는가         -> 상수 예측자를 크기에서 이김
                                                      (150 epoch 본 실험은 1.817)
  3) cmpC 가 cmpA 와 거의 같은가                    -> 진단 확인.  gate 출력이
     상수라면 뒤의 LNLinear 가 흡수하므로 제거해도 같아야 한다.  크게 다르면
     "gate 가 죽어 있다"는 진단이 틀린 것이다.
EOF
    [[ ${rcs} -eq 0 ]] || exit 1
    echo -e "\n===== 완료: ${TAG} ====="
    echo "  결과: ${OUTDIR}"
    exit 0
fi

if [[ "${PARALLEL}" == "1" ]] && has_phase teacher && has_phase stored; then
    T_LOG="${OUTDIR}/teacher-${TAG}.log"
    S_LOG="${OUTDIR}/stored-${TAG}.log"
    echo -e "\n===== phase 1&2 병렬: teacher@${DEV}, stored@${DEV2} ====="
    echo "  로그: ${T_LOG}"
    echo "        ${S_LOG}"
    bench teacher "${DEV}"  teacher > "${T_LOG}" 2>&1 &
    t_pid=$!
    bench stored  "${DEV2}" stored  > "${S_LOG}" 2>&1 &
    s_pid=$!
    t_rc=0; s_rc=0
    wait "${t_pid}" || t_rc=$?
    wait "${s_pid}" || s_rc=$?
    echo -e "\n----- teacher (rc=${t_rc}) 마지막 20줄 -----"; tail -n 20 "${T_LOG}"
    echo -e "\n----- stored  (rc=${s_rc}) 마지막 20줄 -----"; tail -n 20 "${S_LOG}"
    # teacher 가 0 으로 가지 않았다면 stored 는 해석하지 않는다 -- 병렬 실행이
    # 그 판단 순서를 없애지 않도록 여기서 명시적으로 상기시킨다.
    echo -e "\n[읽는 순서] teacher 최종 val d 를 먼저 확인할 것. 0 근처가 아니면"
    echo "            최적화 문제이므로 stored 숫자는 보류한다."
    [[ ${t_rc} -eq 0 && ${s_rc} -eq 0 ]] || exit 1
else
    if has_phase teacher; then
        echo -e "\n===== phase 1/2: realizability (frozen teacher) ====="
        bench teacher "${DEV}" teacher 2>&1 | tee "${OUTDIR}/teacher-${TAG}.log"
    fi
    if has_phase stored; then
        echo -e "\n===== phase 2/2: 저장 라벨 (stored) ====="
        bench stored "${DEV}" stored 2>&1 | tee "${OUTDIR}/stored-${TAG}.log"
    fi
fi

echo -e "\n===== 완료: ${TAG} ====="
echo "  결과: ${OUTDIR}"

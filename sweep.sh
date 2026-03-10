#!/usr/bin/env bash
set -euo pipefail

# 只让一个进程跑 sweep（如果你不是 slurm 环境，这段也不会误伤）
if [[ -n "${SLURM_PROCID-}" && "${SLURM_PROCID}" != "0" ]]; then
  exit 0
fi

WORLD_SIZE=4
SEED=42
CHECK=1

HIDDEN="${HIDDEN:-4096}"
CAPACITY="${CAPACITY:-32768}"
OUT_CSV="${OUT_CSV:-sweep_results.csv}"

EXPERTS_TOTAL_LIST=(32 64)
TOPK_LIST=(2 4 8)
BATCH_LIST=(2 4 8 16)
SEQ_LIST=(2048 4096)

# 只在文件不存在时写表头
if [[ ! -f "${OUT_CSV}" ]]; then
  echo "world_size,experts_total,e_local,topk,batch,seq,hidden,capacity,seed,baseline_ms,custom_ms,speedup,status" > "${OUT_CSV}"
fi

for EXPERTS_TOTAL in "${EXPERTS_TOTAL_LIST[@]}"; do
  if (( EXPERTS_TOTAL % WORLD_SIZE != 0 )); then
    echo "[SKIP] EXPERTS_TOTAL=${EXPERTS_TOTAL} not divisible by WORLD_SIZE=${WORLD_SIZE}"
    continue
  fi

  E_LOCAL=$((EXPERTS_TOTAL / WORLD_SIZE))
  echo "=== EXPERTS_TOTAL=${EXPERTS_TOTAL} -> E_LOCAL=${E_LOCAL} (WORLD_SIZE=${WORLD_SIZE}) ==="

  for TOPK in "${TOPK_LIST[@]}"; do
    for BATCH in "${BATCH_LIST[@]}"; do
      for SEQ in "${SEQ_LIST[@]}"; do
        MASTER_ADDR=127.0.0.1
        MASTER_PORT=$((29500 + (RANDOM % 1000)))

        echo "[RUN] W=${WORLD_SIZE} E_TOTAL=${EXPERTS_TOTAL} E_LOCAL=${E_LOCAL} TOPK=${TOPK} BATCH=${BATCH} SEQ=${SEQ} HIDDEN=${HIDDEN} CAP=${CAPACITY} SEED=${SEED} MASTER_PORT=${MASTER_PORT}"

        LOG=$(env \
          WORLD_SIZE="${WORLD_SIZE}" SEED="${SEED}" CHECK="${CHECK}" \
          HIDDEN="${HIDDEN}" CAPACITY="${CAPACITY}" \
          TOPK="${TOPK}" BATCH="${BATCH}" SEQ="${SEQ}" E_LOCAL="${E_LOCAL}" \
          MASTER_ADDR="${MASTER_ADDR}" MASTER_PORT="${MASTER_PORT}" \
          python benchmark_gather.py 2>&1)

        echo "${LOG}"

        RESULT_LINE=$(echo "${LOG}" | grep '^RESULT,' | tail -n 1 || true)

        if [[ -n "${RESULT_LINE}" ]]; then
          echo "${RESULT_LINE#RESULT,}" >> "${OUT_CSV}"
        else
          echo "${WORLD_SIZE},${EXPERTS_TOTAL},${E_LOCAL},${TOPK},${BATCH},${SEQ},${HIDDEN},${CAPACITY},${SEED},,,,missing_result" >> "${OUT_CSV}"
        fi
      done
    done
  done
done
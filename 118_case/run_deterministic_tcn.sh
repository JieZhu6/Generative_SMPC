#!/bin/bash
#SBATCH --job-name=det-tcn
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=/data/home/scvk217/run/118_case/118_case/slurm_logs/det-tcn-%j.out

# Train the deterministic non-causal TCN on one Slurm-assigned GPU.
set -euo pipefail

PROJECT_DIR=/data/home/scvk217/run/118_case/118_case
ENV_DIR=/data/home/scvk217/run/conda_envs/smpc_py312
DATA_DIR="$PROJECT_DIR/Data_generation/data/e2e118_N5000_S20_T16"
DECS_PATH="$PROJECT_DIR/Neural_network/decs_pgm_fixedpv.pt"
OUTPUT_DIR="$PROJECT_DIR/output/remote5090/deterministic_tcn"
LOG_DIR="$PROJECT_DIR/slurm_logs"

# B32 matches the tuned deterministic default; override through sbatch --export.
BATCH_SIZE=${BATCH_SIZE:-32}

module load miniforge/25.3.0-3
source /data/apps/miniforge/25.3.0-3/etc/profile.d/conda.sh
conda activate "$ENV_DIR"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

test -d "$DATA_DIR"
test -f "$DECS_PATH"

# The gpu partition assigns eight CPU cores per requested GPU.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8

echo "method=deterministic-TCN"
echo "job_id=${SLURM_JOB_ID:-none}"
echo "node=$(hostname)"
echo "start=$(date --iso-8601=seconds)"
echo "batch_size=$BATCH_SIZE"
echo "python=$(which python)"
python --version
nvidia-smi

# Record GPU memory and utilization every 30 seconds for later batch tuning.
GPU_MONITOR_LOG="$LOG_DIR/gpu-det-tcn-${SLURM_JOB_ID}.csv"
nvidia-smi \
    --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw \
    --format=csv \
    --loop=30 > "$GPU_MONITOR_LOG" &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable on the assigned compute node")

print(f"torch={torch.__version__}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"capability={torch.cuda.get_device_capability(0)}")
PY

python -u Neural_network/train_deterministic_tcn.py \
    --data "$DATA_DIR" \
    --decs "$DECS_PATH" \
    --output "$OUTPUT_DIR/deterministic_tcn.pt" \
    --epochs 500 \
    --batch-size "$BATCH_SIZE" \
    --hidden-channels 128 \
    --kernel-size 3 \
    --dilations 1 2 4 8 \
    --ramp-fraction 0.25 \
    --learning-rate 2e-4 \
    --lambda-objective 4e-3 \
    --lambda-fea 5e-1 \
    --lambda-max-violation 0.0 \
    --objective-cost-scale 0.0 \
    --mean-cvar-alpha 5e-2 \
    --cvar-tail-fraction 5e-2 \
    --feasibility-tolerance 1e-4 \
    --patience 100 \
    --min-delta 0.0 \
    --max-grad-norm 1.0 \
    --num-workers 0 \
    --device cuda \
    --seed 2026 \
    2>&1 | tee "$LOG_DIR/train-det-tcn-${SLURM_JOB_ID}.log"

echo "finish=$(date --iso-8601=seconds)"

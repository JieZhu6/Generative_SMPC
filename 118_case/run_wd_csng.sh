#!/bin/bash
#SBATCH --job-name=wd-csng
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --output=/data/home/scvk217/run/118_case/118_case/slurm_logs/wd-csng-%j.out

# Train WD-CSNG on one Slurm-assigned GPU.
set -euo pipefail

PROJECT_DIR=/data/home/scvk217/run/118_case/118_case
ENV_DIR=/data/home/scvk217/run/conda_envs/smpc_py312
DATA_DIR="$PROJECT_DIR/Data_generation/data/e2e118_N5000_S20_T16"
DECS_PATH="$PROJECT_DIR/Neural_network/decs_pgm_fixedpv.pt"
OUTPUT_DIR="$PROJECT_DIR/output/remote5090/wd_csng"
LOG_DIR="$PROJECT_DIR/slurm_logs"

# Defaults to the requested 12/1 configuration. Override with sbatch --export.
BATCH_SIZE=${BATCH_SIZE:-12}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}

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

echo "method=WD-CSNG"
echo "job_id=${SLURM_JOB_ID:-none}"
echo "node=$(hostname)"
echo "start=$(date --iso-8601=seconds)"
echo "batch_size=$BATCH_SIZE"
echo "gradient_accumulation_steps=$GRAD_ACCUM_STEPS"
echo "python=$(which python)"
python --version
nvidia-smi

# Record GPU memory and utilization every 30 seconds for batch-size evaluation.
GPU_MONITOR_LOG="$LOG_DIR/gpu-wd-csng-${SLURM_JOB_ID}.csv"
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

python -u Neural_network/train_generator_wd_csng.py \
    --data "$DATA_DIR" \
    --decs "$DECS_PATH" \
    --output "$OUTPUT_DIR/generator_tcn_wd_csng.pt" \
    --stage1-output "$OUTPUT_DIR/generator_tcn_wd_csng_stage1.pt" \
    --epochs 700 \
    --stage1-epochs 80 \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACCUM_STEPS" \
    --candidates 50 \
    --hidden-channels 64 \
    --latent-dim 16 \
    --latent-embedding-dim 32 \
    --kernel-size 3 \
    --dilations 1 2 4 8 \
    --ramp-fraction 0.25 \
    --learning-rate 2.5e-4 \
    --stage2-learning-rate 5e-5 \
    --lr-scheduler plateau \
    --lr-decay-factor 0.5 \
    --stage1-lr-decay-patience 4 \
    --stage2-lr-decay-patience 25 \
    --lr-min-delta 1e-5 \
    --stage1-min-learning-rate 2.5e-5 \
    --stage2-min-learning-rate 1e-5 \
    --lambda-fea 2e-2 \
    --lambda-eco 3e-3 \
    --economic-warmup-epochs 60 \
    --mean-cvar-alpha 5e-2 \
    --cvar-tail-fraction 5e-2 \
    --economic-cost-scale 0.0 \
    --tau-feas 2.0 \
    --gamma-eco 5.0 \
    --alpha-eco 5e-2 \
    --economic-top-k 1 \
    --feasibility-tolerance 1e-4 \
    --patience 50 \
    --min-delta 0.0 \
    --economic-min-delta 100.0 \
    --max-grad-norm 1.0 \
    --num-workers 0 \
    --device cuda \
    --seed 2026 \
    2>&1 | tee "$LOG_DIR/train-wd-csng-${SLURM_JOB_ID}.log"

echo "finish=$(date --iso-8601=seconds)"

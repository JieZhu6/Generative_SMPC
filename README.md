# End-to-End Stochastic Model Predictive Dispatch for ACOPF Using Generative Models

Official training and evaluation code for the manuscript:

> **End-to-End Stochastic Model Predictive Dispatch for ACOPF Using Generative Models**  
> Jie Zhu and Yinliang Xu  
> Tsinghua Shenzhen International Graduate School, Tsinghua University

This repository implements a conditional stochastic neural generator (CSNG) for multi-period, multi-scenario stochastic model predictive control (SMPC) under nonlinear AC power-flow constraints. Given an uncertainty instance, CSNG generates multiple dispatch trajectories, validates them with the original AC power-flow model, and selects the lowest-cost feasible candidate.

The code includes the complete data-generation, DECS training, CSNG training, ablation, benchmark, and evaluation pipelines for the IEEE 14-bus and IEEE 118-bus systems.

## Method overview

The implementation combines four main components:

1. **Scenario pooling and a non-causal TCN** encode the multi-scenario load forecast over the prediction horizon.
2. **Conditional stochastic generation** uses trajectory-level latent variables to produce multiple dispatch candidates for each SMPC instance.
3. **Constraint-aware projection** exactly enforces box and ramping constraints while retaining useful gradients near active bounds.
4. **Differentiable equality completion (DECS)** reconstructs AC power-flow states during self-supervised training. At inference, all candidates are checked using the original AC power-flow equations.

The self-supervised objective combines feasibility, diversity, and economic terms, avoiding the need to solve and label a large collection of full SMPC problems.

## Paper-to-code map

| Paper component | Main implementation |
|---|---|
| SMPC scenario generation and pooling | `*/Data_generation/generate_smpc_dataset.py` |
| Fixed-PV AC power-flow data for DECS | `*/Data_generation/generate_decs_dataset.py` |
| Differentiable equality-completion surrogate | `*/Neural_network/decs.py`, `*/Neural_network/train_decs.py` |
| Non-causal temporal backbone | `*/Neural_network/noncausal_tcn.py` |
| CSNG, constraint-aware projection, and self-supervised loss | `*/Neural_network/train_generator.py` |
| Deterministic D-NN baseline | `*/Neural_network/train_deterministic_tcn.py` |
| Sigmoid-output S-CSNG ablation | `*/Neural_network/train_generator_s_csng.py` |
| No-diversity WD-CSNG ablation | `*/Neural_network/train_generator_wd_csng.py` |
| IPOPT benchmark | `*/evaluate_ipopt.py` |
| Exact batch AC-PF screening | `*/evaluate_generator_tcn_pgm_batch.py`, `*/pgm_batch_power_flow.py` |

Here, `*` denotes either `14_case` or `118_case`.

## Repository structure

```text
Generative_SMPC/
|-- 14_case/
|   |-- Data_generation/       # IEEE 14-bus data and AC-PF labels
|   |-- Neural_network/        # DECS, CSNG, baselines, and checkpoints
|   |-- output/                # Evaluation results and summaries
|   |-- evaluate_*.py          # IPOPT and neural-model evaluation
|   `-- requirements_pgm_py312.txt
|-- 118_case/
|   |-- Data_generation/       # IEEE 118-bus data and AC-PF labels
|   |-- Neural_network/        # DECS, CSNG, baselines, and checkpoints
|   |-- output/                # Evaluation and sensitivity results
|   |-- run_*.sh               # Slurm training scripts used on the GPU server
|   |-- evaluate_*.py
|   `-- requirements_pgm_py312.txt
|-- .gitattributes             # Git LFS rules for large arrays
`-- README.md
```

More detailed Chinese documentation is available in:

- [`14_case/Data_generation/README.md`](14_case/Data_generation/README.md)
- [`118_case/Data_generation/README.md`](118_case/Data_generation/README.md)

## Installation

### 1. Clone the repository and download LFS data

Several large training arrays are stored with Git LFS.

```bash
git lfs install
git clone https://github.com/JieZhu6/Generative_SMPC.git
cd Generative_SMPC
git lfs pull
```

Without `git lfs pull`, the large `.npy` files will remain small pointer files and training will fail.

### 2. Create the Python environment

Python 3.12 is recommended. The checked requirements use PyTorch 2.5.1 with CUDA 12.1.

```bash
python -m venv .venv
```

Activate the environment on Linux:

```bash
source .venv/bin/activate
```

or on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r 118_case/requirements_pgm_py312.txt
```

The main dependencies are NumPy, SciPy, pandas, PyTorch, pandapower, Power Grid Model, Pyomo, and Matplotlib. The IPOPT comparison additionally requires an IPOPT executable accessible to Pyomo.

Most scripts accept `--device auto`, `--device cpu`, or `--device cuda`. CPU execution is suitable for small checks, while full CSNG training - especially for the IEEE 118-bus case - is intended for a CUDA GPU.

## Supplied data and checkpoints

The repository already includes the datasets, trained checkpoints, and evaluation summaries used by the current experiments. The main settings are:

| Setting | IEEE 14-bus | IEEE 118-bus |
|---|---:|---:|
| SMPC instances | 5,000 | 5,000 |
| Uncertainty scenarios | 20 | 20 |
| Prediction horizon | 16 | 16 |
| Train/validation/test split | 8:1:1 | 8:1:1 |
| DECS training points | 10,000 | 50,000 |
| Evaluation candidates per instance | 50 | 100 |
| SMPC dataset | `e2e14_N5000_S20_T16` | `e2e118_N5000_S20_T16` |
| DECS dataset | `e2e14_decs_pgm_fixedpv_N10000` | `e2e118_decs_pandapower_N50000` |

The exact random seeds, load ranges, dimensions, solver settings, and split indices are recorded in each dataset's `metadata.json`. Treat those metadata files as the source of truth when reproducing a saved checkpoint.

## Quick evaluation with the supplied models

Run commands from the selected case directory.

### IEEE 14-bus

```bash
cd 14_case

python evaluate_deterministic_tcn.py
python evaluate_generator_tcn_pgm_batch.py --candidates 50
python evaluate_generator_s_csng_pgm_batch.py --candidates 50
python evaluate_generator_wd_csng_pgm_batch.py --candidates 50
python evaluate_ipopt.py
```

### IEEE 118-bus

```bash
cd 118_case

python evaluate_deterministic_tcn.py --max-instances 200
python evaluate_generator_tcn_pgm_batch.py --candidates 100 --max-instances 200
python evaluate_generator_s_csng_pgm_batch.py --candidates 100 --max-instances 200
python evaluate_generator_wd_csng_pgm_batch.py --candidates 100 --max-instances 200
python evaluate_ipopt.py --max-instances 200
```

Evaluation produces CSV solution tables, NPZ arrays, JSON summaries, and independent pandapower/Power Grid Model consistency checks under each case's `output/` directory.

## Reproduce the training pipeline

The commands below assume that the current directory is `14_case` or `118_case`.

### 1. Generate SMPC datasets

For IEEE 14-bus:

```bash
python Data_generation/generate_smpc_dataset.py \
  --n-instances 5000 --n-scenarios 20 --horizon 16 \
  --output Data_generation/data/e2e14_N5000_S20_T16
```

For IEEE 118-bus:

```bash
python Data_generation/generate_smpc_dataset.py \
  --n-instances 5000 --n-scenarios 20 --horizon 16 \
  --output Data_generation/data/e2e118_N5000_S20_T16
```

Each dataset contains current loads, full future scenario tensors, min/mean/max pooled features, normalization parameters, metadata, and fixed train/validation/test indices.

### 2. Generate DECS labels

For IEEE 14-bus, native Power Grid Model batches are used:

```bash
python Data_generation/generate_decs_dataset.py \
  --base-data Data_generation/data/e2e14_N5000_S20_T16 \
  --n-samples 10000 --batch-size 1000 \
  --load-range-extension 0.02 --free-range-extension 0.05 \
  --output Data_generation/data/e2e14_decs_pgm_fixedpv_N10000
```

For IEEE 118-bus, the supplied dataset was generated with the pandapower backend:

```bash
python Data_generation/generate_decs_dataset.py \
  --base-data Data_generation/data/e2e118_N5000_S20_T16 \
  --solver pandapower --n-samples 50000 --batch-size 1000 \
  --load-range-extension 0.02 --free-range-extension 0.02 \
  --output Data_generation/data/e2e118_decs_pandapower_N50000
```

### 3. Train DECS

IEEE 14-bus:

```bash
python Neural_network/train_decs.py \
  --data Data_generation/data/e2e14_decs_pgm_fixedpv_N10000 \
  --hidden-dims 128 128 --lambda-physics 1.0 \
  --output Neural_network/decs_pgm_fixedpv.pt
```

IEEE 118-bus:

```bash
python Neural_network/train_decs.py \
  --data Data_generation/data/e2e118_decs_pandapower_N50000 \
  --hidden-dims 256 256 --epochs 700 --batch-size 1024 \
  --learning-rate 1e-3 --lambda-physics 3 \
  --output Neural_network/decs_pgm_fixedpv.pt
```

DECS is trained with supervised state reconstruction and an AC power-balance residual. The best validation checkpoint is frozen during subsequent surrogate training.

### 4. Train CSNG and the baselines

The default paths in the training scripts point to the standard dataset and DECS checkpoint for the selected case:

```bash
# Proposed model
python Neural_network/train_generator.py

# Deterministic D-NN baseline
python Neural_network/train_deterministic_tcn.py

# Sigmoid-output ablation
python Neural_network/train_generator_s_csng.py

# No-diversity ablation
python Neural_network/train_generator_wd_csng.py
```

CSNG training uses two stages. Stage 1 shapes a feasible and non-collapsed candidate distribution; Stage 2 activates the economic loss. The best checkpoints from both stages are saved separately.

For the IEEE 118-bus GPU experiments, the `run_*.sh` files provide the full Slurm commands used for long-running training. Before submitting them, update `PROJECT_DIR` and `ENV_DIR` to match your cluster:

```bash
cd 118_case
sbatch run_deterministic_tcn.sh
sbatch run_generator_csng.sh
sbatch run_s_csng.sh
sbatch run_wd_csng.sh
```

## Paper results

The manuscript reports the following test-set performance. Times depend on hardware and available AC power-flow parallelism.

### IEEE 14-bus system

| Method | Objective | Optimality gap | Feasibility | Time/instance |
|---|---:|---:|---:|---:|
| IPOPT | 35,031 | - | 100% | 19.86 s |
| D-NN | 35,140 | 0.31% | 46.2% | 0.01 s |
| S-CSNG | 35,658 | 1.79% | 100% | 0.19 s |
| WD-CSNG | 35,362 | 0.94% | 99.8% | 0.19 s |
| **CSNG** | **35,310** | **0.80%** | **100%** | **0.19 s** |

### IEEE 118-bus system

| Method | Objective | Optimality gap | Feasibility | Time/instance |
|---|---:|---:|---:|---:|
| IPOPT | 1,541,150 | - | 100% | 398.02 s |
| D-NN | 1,556,493 | 1.00% | 83.0% | 0.01 s |
| S-CSNG | 1,588,919 | 3.10% | 100% | 14.27 s |
| WD-CSNG | 1,552,132 | 0.71% | 36.2% | 16.17 s |
| **CSNG** | **1,568,985** | **1.81%** | **100%** | **13.74 s** |

For the IEEE 118-bus sensitivity study, all test instances recover at least one feasible candidate once `K >= 5`; the manuscript uses `K = 100` as the main feasibility/economy/runtime trade-off.

## Reproducibility notes

- The default random seed is `2026`; dataset splits use saved indices.
- Reactive-power limit violations are evaluated as inequalities and do not trigger PV-to-PQ switching.
- DECS is used only to provide differentiable constraint feedback during offline training. Final inference feasibility is determined with the original AC power-flow equations.
- Generated candidates are screened in parallel, and the lowest-cost feasible candidate is selected.
- Checkpoint compatibility depends on the dataset dimensions and normalization parameters. Keep each checkpoint with the corresponding dataset metadata.
- Full training can be memory intensive because each instance expands across candidates, scenarios, and time periods. Reduce the micro-batch size or use gradient accumulation instead of changing the network or constraint set.

## Citation

If you use this repository, please cite the manuscript. Replace the entry below with the final journal metadata after publication.

```bibtex
@article{zhu_end_to_end_smpc,
  author  = {Jie Zhu and Yinliang Xu},
  title   = {End-to-End Stochastic Model Predictive Dispatch for ACOPF Using Generative Models},
  note    = {Manuscript},
}
```

## Contact

For questions about the paper or implementation, please open a GitHub issue or contact Yinliang Xu at `xu.yinliang@sz.tsinghua.edu.cn`.

"""生成 conditional generator 使用的无标签 SMPC 基础数据集。

本程序对应论文中条件随机生成器（conditional stochastic generator）的离线
训练数据构建阶段。每个 SMPC 实例 c 由以下部分组成：

- ξ_1：t=1 的确定性净负荷；
- {ξ_s}_{s∈N_s}：t=2,...,T 的 S 条未来净负荷场景；
- pool = [min_s ξ_s, mean_s ξ_s, max_s ξ_s]：场景维的经验池化，对应论文
  式 (8)。池化后的表示与 ξ_1 拼接成生成器条件输入。

最终保存的数组
--------------
current_load : (N, n_load_bus, 2)
    t=1 确定性节点净负荷，通道为 [P_net, Q_net]。
future_load : (N, S, T-1, n_load_bus, 2)
    t=2,...,T 的未来净负荷场景。
future_pool : (N, 3, T-1, n_load_bus, 2)
    沿场景维的 min/mean/max，直接对应论文式 (8)。

数据生成保持纯采样逻辑，不在循环中调用 AC-OPF/IPOPT。默认通过保守的
非对称负荷范围降低不可行样本的风险；最终约束可行性由训练/评估程序检验。
"""

import argparse
import gc
import json
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
from numpy.lib.format import open_memmap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Data_generation.case118_pglib import load_case118  # noqa: E402
from Data_generation.generate_smpc_scenarios import generate_bundle  # noqa: E402


def prepare_staged_output(final_output: Path) -> tuple[Path, Path]:
    """Create a unique sibling work directory without creating ``final_output``.

    Parameters
    ----------
    final_output : pathlib.Path
        Dataset directory that should become visible only after a successful run.

    Returns
    -------
    final_output, staged_output : tuple[pathlib.Path, pathlib.Path]
        Absolute final path and a unique temporary sibling used for all writes.
        An interrupted run may leave only the ``.incomplete-*`` directory, which
        never blocks a retry with the same final dataset name.
    """
    final_output = final_output.resolve()
    if final_output.exists():
        raise FileExistsError(f"output already exists: {final_output}")
    final_output.parent.mkdir(parents=True, exist_ok=True)
    staged_output = Path(tempfile.mkdtemp(
        prefix=f".{final_output.name}.incomplete-",
        dir=final_output.parent,
    ))
    return final_output, staged_output


def file_size_manifest(directory: Path) -> dict[str, int]:
    """Return relative file paths and sizes for a dataset directory.

    Parameters
    ----------
    directory : pathlib.Path
        Existing dataset directory to inspect recursively.

    Returns
    -------
    dict[str, int]
        Mapping from POSIX-style relative paths to file sizes in bytes.
    """
    return {
        path.relative_to(directory).as_posix(): path.stat().st_size
        for path in directory.rglob("*")
        if path.is_file()
    }


def publish_staged_output(staged_output: Path, final_output: Path) -> None:
    """Publish a completed dataset despite transient Windows directory locks.

    Parameters
    ----------
    staged_output : pathlib.Path
        Completed sibling directory whose files are fully closed.
    final_output : pathlib.Path
        New final dataset path; it must not already exist.

    Notes
    -----
    The preferred operation is an atomic sibling-directory rename. If Windows
    still denies that rename after all retries, the completed files are copied
    to a newly created final directory and checked against a size manifest.
    """
    if final_output.exists():
        raise FileExistsError(f"output appeared during generation: {final_output}")
    attempts = 10 if sys.platform == "win32" else 1
    last_error = None
    for attempt in range(attempts):
        try:
            staged_output.replace(final_output)
            return
        except PermissionError as error:
            last_error = error
            if final_output.exists():
                raise FileExistsError(
                    f"output appeared during generation: {final_output}"
                )
            if attempt + 1 < attempts:
                # Large memmap files can remain briefly locked by Windows or an
                # antivirus scanner even after their Python handles are closed.
                gc.collect()
                sleep(1.0)

    if sys.platform != "win32":
        raise last_error

    source_manifest = file_size_manifest(staged_output)
    try:
        shutil.copytree(staged_output, final_output)
        if file_size_manifest(final_output) != source_manifest:
            raise OSError("copied dataset does not match the staged file manifest")
    except Exception:
        # The final directory was created only by this failed copy attempt.
        if final_output.exists():
            shutil.rmtree(final_output)
        raise

    try:
        shutil.rmtree(staged_output)
    except OSError as error:
        warnings.warn(
            f"dataset was saved successfully, but temporary directory "
            f"{staged_output} could not be removed: {error}",
            RuntimeWarning,
            stacklevel=2,
        )


def pool_scenarios(future_load: np.ndarray) -> np.ndarray:
    """沿场景维逐元素计算论文式 (8) 的 min/mean/max。

    输入 ``future_load`` 的形状为 (S, T-1, n_load_bus, 2)。沿第 0 维（场景）
    分别取最小值、平均值、最大值，再沿新维度堆叠，得到形状
    (3, T-1, n_load_bus, 2)。该池化表示对场景顺序不敏感，与生成器输入
    的集合型场景建模一致。
    """
    return np.stack([
        future_load.min(axis=0),
        future_load.mean(axis=0),
        future_load.max(axis=0),
    ])


def make_paper_splits(n_instances: int, seed: int) -> dict[str, np.ndarray]:
    """Return the reproducible 8:1:1 instance split used by generator training.

    Parameters
    ----------
    n_instances : int
        Number of accepted scenario trees; at least ten.
    seed : int
        Permutation seed, conventionally the generation seed plus one.

    Returns
    -------
    dict[str, np.ndarray]
        Train, validation, and test indices whose union is ``range(n_instances)``.
    """
    if n_instances < 10:
        raise ValueError("at least ten instances are required for an 8:1:1 split")
    order = np.random.default_rng(seed).permutation(n_instances)
    n_train = int(0.8 * n_instances)
    n_validation = int(0.1 * n_instances)
    return {
        "train": order[:n_train],
        "validation": order[n_train:n_train + n_validation],
        "test": order[n_train + n_validation:],
    }


def bundle_load_arrays(bundle, load_buses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compress one full-bus scenario bundle to dataset P/Q load arrays.

    Parameters
    ----------
    bundle : ScenarioBundle
        Generated current and future full-bus load trajectories.
    load_buses : np.ndarray, shape (n_load_buses,)
        Zero-based buses with nonzero base demand.

    Returns
    -------
    current_load, future_load : tuple[np.ndarray, np.ndarray]
        Arrays shaped ``(n_load_buses,2)`` and ``(S,T-1,n_load_buses,2)``.
    """
    current_load = np.stack([
        bundle.current_pd[load_buses], bundle.current_qd[load_buses],
    ], axis=-1)
    future_load = np.stack([
        np.take(bundle.future_pd, load_buses, axis=-1),
        np.take(bundle.future_qd, load_buses, axis=-1),
    ], axis=-1)
    return current_load, future_load


def base_dataset_directory_name(n: int, s: int, t: int) -> str:
    """返回由规模标识的 SMPC 基础数据集默认目录名。"""
    return f"e2e118_N{n}_S{s}_T{t}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-instances", type=int, default=5000, metavar="N",
        help="场景树的数量",
    )
    parser.add_argument(
        "--n-scenarios", type=int, default=20, metavar="S",
        help="每个实例从 t=2 开始的未来场景数 S",
    )
    parser.add_argument(
        "--horizon", type=int, default=16, metavar="T",
        help="论文预测域总时段数 T；t=1 确定，t=2,...,T 为未来场景",
    )
    parser.add_argument(
        "--forecast-deviation", type=float, default=0.08, metavar="D",
        help="tanh预测误差相对日内曲线的最大幅度，取值范围[0,1)",
    )
    parser.add_argument(
        "--load-scale-min", type=float, default=0.75, metavar="LOWER",
        help="节点负荷相对PGLib基准值的保守非对称绝对下界",
    )
    parser.add_argument(
        "--load-scale-max", type=float, default=1.05, metavar="UPPER",
        help="节点负荷相对PGLib基准值的保守非对称绝对上界",
    )
    parser.add_argument(
        "--rho", type=float, default=0.95, metavar="RHO",
        help="未来预测误差的 AR(1) 时间相关系数",
    )
    parser.add_argument(
        "--seed", type=int, default=2026, metavar="SEED",
        help="主随机种子；第 n 个实例使用 SEED+1009*n",
    )
    parser.add_argument(
        "--output", type=Path, metavar="DIR", default=None,
        help=(
            "显式指定基础数据集最终目录；省略时自动写入 data/"
            "e2e118_N{N}_S{S}_T{T}"
        ),
    )
    args = parser.parse_args()

    if args.n_instances < 10 or args.n_scenarios < 1:
        raise ValueError("N must be at least ten and S must be positive")
    if args.horizon < 3:
        raise ValueError("T must be at least 3")
    if not 0.0 <= args.forecast_deviation < 1.0:
        raise ValueError("forecast deviation must satisfy 0 <= D < 1")
    if not 0.0 < args.load_scale_min < args.load_scale_max:
        raise ValueError("load scale limits must be positive and ordered")
    if not 0.0 <= args.rho < 1.0:
        raise ValueError("rho must satisfy 0 <= rho < 1")

    if args.output is None:
        name = base_dataset_directory_name(
            args.n_instances, args.n_scenarios, args.horizon,
        )
        args.output = Path(__file__).resolve().parent / "data" / name
    final_output, args.output = prepare_staged_output(args.output)

    case = load_case118()
    load_buses = case.load_buses
    n, s = args.n_instances, args.n_scenarios
    future_steps = args.horizon - 1

    shapes = {
        "current_load": (n, len(load_buses), 2),
        "future_load": (n, s, future_steps, len(load_buses), 2),
        "future_pool": (n, 3, future_steps, len(load_buses), 2),
    }
    data = {
        name: open_memmap(
            args.output / f"{name}.npy", mode="w+", dtype="float32", shape=shape,
        )
        for name, shape in shapes.items()
    }

    print("Base dataset configuration:")
    print(f"  System={case.name}, buses={case.n_bus}, generators={case.n_gen}")
    print(f"  N={n}, S={s}, T={args.horizon}")
    print(
        f"  load scale=[{args.load_scale_min:.3f},{args.load_scale_max:.3f}], "
        f"forecast deviation={args.forecast_deviation:.3f}, rho={args.rho:.3f}"
    )
    print("  generation=direct sampling (no IPOPT/OPF certification)")
    print(f"  Output after successful completion: {final_output}")

    source_seeds = np.empty(n, dtype="int64")
    start_time = perf_counter()
    progress_interval = max(1, n // 100)
    for index in range(n):
        sample_seed = args.seed + 1009 * index
        bundle = generate_bundle(
            sample_seed=sample_seed,
            horizon=args.horizon,
            n_scenarios=s,
            forecast_deviation=args.forecast_deviation,
            rho=args.rho,
            case=case,
            load_scale_min=args.load_scale_min,
            load_scale_max=args.load_scale_max,
        )
        current_load, future_load = bundle_load_arrays(bundle, load_buses)

        # 写入内存映射数组；注意 open_memmap 创建的是磁盘文件，循环内逐实例
        # 写入即可，无需在内存中保存完整数据集。
        data["current_load"][index] = current_load
        data["future_load"][index] = future_load
        data["future_pool"][index] = pool_scenarios(future_load)
        source_seeds[index] = sample_seed

        completed = index + 1
        if completed % progress_interval == 0 or completed == n:
            elapsed = perf_counter() - start_time
            remaining = elapsed / completed * (n - completed)
            print(
                f"\rGenerated: {completed}/{n} ({100.0 * completed / n:5.1f}%) "
                f"| elapsed {elapsed:.1f} s | ETA {remaining:.1f} s",
                end="", flush=True,
            )
    print()
    np.save(args.output / "source_seeds.npy", source_seeds)

    splits = make_paper_splits(n, args.seed + 1)
    train, validation, test = (
        splits["train"], splits["validation"], splits["test"],
    )
    split_dir = args.output / "split"
    split_dir.mkdir()
    np.save(split_dir / "train_indices.npy", train)
    np.save(split_dir / "validation_indices.npy", validation)
    np.save(split_dir / "test_indices.npy", test)

    # 归一化参数仅基于训练集计算，防止验证/测试信息泄漏。
    # min/max 归一化将各通道线性映射到 [0, 1]，使神经网络输入尺度一致。
    future_min = np.full((len(load_buses), 2), np.inf)
    future_max = np.full((len(load_buses), 2), -np.inf)
    for index in train:
        # 对每条场景序列取 (S, T-1) 上的全局最小/最大，得到按母线/通道的最值。
        future_min = np.minimum(
            future_min, data["future_load"][index].min(axis=(0, 1)),
        )
        future_max = np.maximum(
            future_max, data["future_load"][index].max(axis=(0, 1)),
        )
    np.savez(
        args.output / "normalization_parameters.npz",
        current_min=data["current_load"][train].min(axis=0),
        current_max=data["current_load"][train].max(axis=0),
        future_min=future_min,
        future_max=future_max,
    )

    # 元数据供 DECS 样本生成与 generator 训练校验网络和场景设置。
    metadata = {
        "dataset_type": "smpc_base",
        "case_name": case.name,
        "n_bus": case.n_bus,
        "n_gen": case.n_gen,
        "active_generators": (case.active_generators + 1).tolist(),
        "n_instances": n,
        "n_scenarios": s,
        "horizon": args.horizon,
        "forecast_deviation": args.forecast_deviation,
        "load_scale_min": args.load_scale_min,
        "load_scale_max": args.load_scale_max,
        "rho": args.rho,
        "seed": args.seed,
        "first_stage_ramp": False,
        "recourse_ramp": True,
        "load_buses": (load_buses + 1).tolist(),
        "load_channels": ["P_net", "Q_net"],
        "pooling_order": ["minimum", "mean", "maximum"],
        "split": {
            "ratio": [0.8, 0.1, 0.1],
            "seed": args.seed + 1,
            "counts": {name: len(indices) for name, indices in splits.items()},
        },
        "feasibility_strategy": {
            "method": "conservative asymmetric load-range sampling",
            "solver_called_during_generation": False,
            "formally_certified": False,
            "note": "Feasibility is evaluated by the downstream training/test pipeline.",
        },
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8",
    )

    for array in data.values():
        array.flush()
        # Windows 不允许在活动 memmap 句柄下重命名其父目录。
        array._mmap.close()
    data.clear()
    del array
    gc.collect()
    publish_staged_output(args.output, final_output)
    print(f"Saved SMPC base dataset to {final_output}")


if __name__ == "__main__":
    main()

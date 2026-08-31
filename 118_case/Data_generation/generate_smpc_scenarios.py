"""生成一个确定当前时刻和多场景未来时刻组成的 SMPC 输入样本。

本模块对应论文中端到端 SMPC 框架的“外生输入生成”部分：
- 条件生成器（conditional stochastic generator）的输入条件 c 由 t=1 的确定性
  信息 ξ_1 与未来场景集合 {ξ_s}_{s∈N_s} 的池化表示 pool 拼接而成，见论文
  式 (8)。

每天首先由 24 个整点系数插值得到 96 个 15 分钟负荷系数，再随机选择一个
连续预测窗口。论文时刻 ``t=1`` 只有一组已观测负荷；``t=2,...,T`` 的预测
误差从该共同状态分叉成多个 AR(1) 场景。当前暂不单列可再生能源：未来加入
可再生能源时，将其作为负节点负荷并入净负荷。

主要输出结构
------------
ScenarioBundle : dataclass
    包含 ``current_pd/qd`` (t=1 确定性负荷)、``future_pd/qd``
    (t=2,...,T 的 S 条场景) 以及 ``start_period`` (窗口在
    96 点日曲线中的起始位置)。所有功率量均为有名值（MW/Mvar）。
"""

from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Data_generation.case118_pglib import Case118, load_case118  # noqa: E402


# 24个整点负荷系数用于描述一天内的基础负荷变化：凌晨低谷、上午爬坡、
# 午间平台、傍晚高峰和夜间回落。它不是随机扰动，而是所有样本共享的
# 确定性日内均值曲线。各系数均相对于PGLib标准负荷取值。
HOURLY_LOAD_PROFILE_24 = np.array([
    0.92, 0.90, 0.89, 0.88, 0.89, 0.92,
    0.97, 1.02, 1.06, 1.08, 1.05, 1.02,
    1.00, 0.99, 0.98, 1.00, 1.04, 1.09,
    1.15, 1.12, 1.07, 1.02, 0.98, 0.95,
])


def _quarter_hour_profile() -> np.ndarray:
    """将24个整点负荷系数线性插值为96个15分钟负荷系数。

    输入参数
    --------
    无。该函数直接读取模块常量 ``HOURLY_LOAD_PROFILE_24``。

    返回值
    ------
    np.ndarray, shape (96,)
        一天96个15分钟区间的确定性基准负荷系数。第0个元素对应
        00:00--00:15，第95个元素对应23:45--24:00。
    """
    # 追加次日00:00的系数，使23:00--24:00也能平滑插值，并保证日曲线首尾连续。
    hourly_periodic = np.r_[HOURLY_LOAD_PROFILE_24, HOURLY_LOAD_PROFILE_24[0]]
    hourly_grid = np.arange(25, dtype=float)
    quarter_hour_grid = np.arange(96, dtype=float) / 4.0
    return np.interp(quarter_hour_grid, hourly_grid, hourly_periodic)


# REFERENCE_PROFILE_96[k]对应第k个15分钟区间，k=0表示00:00--00:15，
# k=95表示23:45--24:00。默认16时段预测域因此覆盖连续4小时。
REFERENCE_PROFILE_96 = _quarter_hour_profile()


@dataclass
class ScenarioBundle:
    """单个监督学习样本对应的完整 SMPC 外生输入。

    参数
    ----
    current_pd, current_qd : np.ndarray, shape (n_bus,)
        论文时刻 ``t=1`` 的确定性节点有功、无功净负荷，单位 MW、Mvar。
    future_pd, future_qd : np.ndarray,
        shape (n_scenarios, horizon-1, n_bus)
        论文时刻 ``t=2,...,T`` 的节点有功、无功净负荷场景。
    start_period : int
        ``t=1`` 在一天 96 点参考曲线中的 0 基索引。
    previous_pg : np.ndarray or None, shape (n_gen,)
        仅在求解 recourse 子问题时使用，表示紧邻该子问题之前的
        机组有功状态。完整 SMPC 的 t=1 输入为 ``None``。
    """

    current_pd: np.ndarray       # (bus,)：t=1确定性有功负荷，MW
    current_qd: np.ndarray       # (bus,)：t=1确定性无功负荷，Mvar
    future_pd: np.ndarray        # (scenario, horizon-1, bus)：t=2,...,T有功场景
    future_qd: np.ndarray        # (scenario, horizon-1, bus)：t=2,...,T无功场景
    start_period: int            # t=1在96点日曲线中的起始编号，范围0,...,96-horizon
    previous_pg: np.ndarray | None = None  # recourse 首时段的前一机组状态


def _select_daily_window(rng: np.random.Generator, horizon: int) -> tuple[int, np.ndarray]:
    """从 96 点日曲线中随机选择连续预测窗口。

    输入参数
    --------
    rng : np.random.Generator
        NumPy随机数生成器，用于抽取窗口起点。调用者使用固定随机种子
        初始化该对象后，可以复现完全相同的窗口位置。
    horizon : int
        SMPC预测域包含的决策时段数。每个时段为15分钟；默认值16对应
        连续4小时。该参数必须满足 ``2 <= horizon <= 96``。

    返回值
    ------
    start : int
        当前决策时段 ``t=1`` 在96点日曲线中的索引，取值范围为
        ``0,...,96-horizon``。
    profile : np.ndarray, shape (horizon,)
        从 t=1 开始的 ``horizon`` 个连续负荷系数。
    """
    if horizon > len(REFERENCE_PROFILE_96):
        raise ValueError("horizon cannot exceed the 96-period daily profile")

    # start最大为96-horizon，因此决策窗口[start, start+horizon)始终位于同一天内，
    # 不在数组尾部回绕。这使“连续16时段”的含义与原始96点序列完全一致。
    start = int(rng.integers(0, len(REFERENCE_PROFILE_96) - horizon + 1))
    return start, REFERENCE_PROFILE_96[start:start + horizon]


def _ar1_paths(
    rng: np.random.Generator,
    n_scenarios: int,
    n_steps: int,
    n_bus: int,
    rho: float,
    initial_global: float,
    initial_local: np.ndarray,
) -> np.ndarray:
    """生成从确定性当前状态出发、具有时间与空间相关性的预测误差。

    该函数实现论文中多场景 SMPC 的预测误差模型。假设相邻 15 分钟预测误差
    服从一阶自回归 AR(1) 过程：

        ε_{t+1} = ρ * ε_t + sqrt(1-ρ^2) * η_t,    η_t ~ N(0,1)

    其中 ``rho`` 控制时间平滑性，``innovation_scale`` 保证稳态方差为 1。
    空间相关性通过“系统级共同误差 + 节点级局部误差”的线性组合实现：

        error = 0.70 * global_error + 0.30 * local_error

    70% 的共同分量确保不同节点负荷呈现同步趋势；30% 的节点局部分量提供
    空间异质性，使各节点场景不会完全同步。

    输入参数
    --------
    rng : np.random.Generator
        NumPy随机数生成器，用于产生各场景、各时段的标准正态创新项。
    n_scenarios : int
        随机场景数量，默认数据生成流程中为50。
    n_steps : int
        需要生成随机误差的未来时段数。由于论文模型中的 ``t=1`` 已经
        确定，因此通常等于 ``horizon - 1``；默认预测域下为15。
    n_bus : int
        电力系统节点数。PGLib IEEE-118 算例中为 118。
    rho : float
        AR(1)时间相关系数，建议满足 ``0 <= rho < 1``。数值越接近1，
        相邻15分钟的预测误差越平滑；默认值0.82。
    initial_global : float
        ``t=1`` 时刻的系统级共同误差状态。它是一个标量，表示所有节点
        共同经历的系统总负荷偏差。
    initial_local : np.ndarray, shape (n_bus,)
        ``t=1`` 时刻各节点的局部误差状态。第 ``i`` 个元素表示节点
        ``i`` 相对于系统共同趋势的附加偏差。

    返回值
    ------
    np.ndarray, shape (n_scenarios, n_steps, n_bus)
        尚未乘以扰动上限的原始预测误差。第一个维度是场景，第二个维度
        对应论文中的 ``t=2,...,T``，第三个维度是节点。外层函数还会对
        该结果应用 ``tanh`` 并乘以 ``forecast_deviation``；外层函数再用
        独立的非对称绝对缩放边界裁剪最终负荷。
    """
    global_error = np.empty((n_scenarios, n_steps, 1))
    local_error = np.empty((n_scenarios, n_steps, n_bus))
    innovation_scale = np.sqrt(1.0 - rho**2)

    # t=2的误差以t=1的共同误差为条件，因此50个场景从同一当前状态向未来分叉。
    global_error[:, 0, 0] = (
        rho * initial_global
        + innovation_scale * rng.normal(size=n_scenarios)
    )
    local_error[:, 0] = (
        rho * initial_local
        + innovation_scale * rng.normal(size=(n_scenarios, n_bus))
    )

    # 每条场景独立沿自己的上一时段误差演化，避免用场景均值替代场景历史。
    for t in range(1, n_steps):
        global_error[:, t, 0] = (
            rho * global_error[:, t - 1, 0]
            + innovation_scale * rng.normal(size=n_scenarios)
        )
        local_error[:, t] = (
            rho * local_error[:, t - 1]
            + innovation_scale * rng.normal(size=(n_scenarios, n_bus))
        )

    # 70%的系统级误差使不同节点保持共同负荷趋势；30%的节点级误差提供空间差异。
    return 0.70 * global_error + 0.30 * local_error


def generate_bundle(
    sample_seed: int,
    horizon: int = 16,
    n_scenarios: int = 50,
    forecast_deviation: float = 0.10,
    rho: float = 0.82,
    case: Case118 | None = None,
    load_scale_min: float = 0.85,
    load_scale_max: float = 1.15,
) -> ScenarioBundle:
    """生成一个可复现的两阶段SMPC输入样本。

    输入参数
    --------
    sample_seed : int
        当前样本的独立随机种子，同时决定96点曲线窗口、当前负荷偏差、
        未来场景创新。相同种子和参数得到相同样本。
    horizon : int, default=16
        SMPC预测域长度。一个时段为15分钟，默认16个时段对应4小时；
        其中 ``t=1`` 确定，``t=2,...,T`` 具有随机场景。
    n_scenarios : int, default=50
        从 ``t=2`` 开始考虑的未来负荷场景数。
    forecast_deviation : float, default=0.10
        ``tanh`` 预测误差相对日内均值曲线的最大幅度。
    rho : float, default=0.82
        AR(1)预测误差的时间相关系数，控制相邻15分钟随机误差的连续性。
    case : Case118 or None, default=None
        PGLib IEEE-118 系统数据。传入已有对象可避免批量生成时重复构造网络
        参数；为 ``None`` 时由函数内部调用 ``load_case118``。
    load_scale_min, load_scale_max : float, defaults=(0.90,1.05)
        相对PGLib基准负荷的非对称绝对缩放边界。上界根据带运行裕度的
        历史可行性压力测设为1.05，下界收紧为0.90以避免过轻负荷工况。

    返回值
    ------
    ScenarioBundle
        包含以下字段：

        - ``current_pd``，shape ``(n_bus,)``：t=1确定性有功负荷，MW；
        - ``current_qd``，shape ``(n_bus,)``：t=1确定性无功负荷，Mvar；
        - ``future_pd``，shape ``(n_scenarios, horizon-1, n_bus)``：
          t=2,...,T的有功负荷场景，MW；
        - ``future_qd``，形状同上：未来无功负荷场景，Mvar；
        - ``start_period``：t=1在96点日曲线中的起始索引。
    """
    if horizon < 2 or n_scenarios < 1:
        raise ValueError("horizon must be at least 2 and n_scenarios must be positive")
    if not 0.0 <= forecast_deviation < 1.0:
        raise ValueError("forecast_deviation must lie in [0, 1)")
    if not 0.0 < load_scale_min < load_scale_max:
        raise ValueError("load_scale_min and load_scale_max must be positive and ordered")

    case = case or load_case118()
    rng = np.random.default_rng(sample_seed)
    start_period, profile = _select_daily_window(rng, horizon)

    # profile[0]对应论文中的 t=1，profile[1:]对应 t=2,...,T。

    # 当前时刻 (t=1) 只有一组已经观测到的负荷，不带场景下标。系统级误差和
    # 节点级误差的加权组合同时决定当前各节点相对于日内均值曲线的偏移。
    # scale=0.55 的高斯误差经 tanh 压缩后乘预测误差幅度；最终缩放再由
    # 独立的非对称上下界裁剪，避免把日内范围与预测不确定性混为一项参数。
    current_global = float(rng.normal(scale=0.55))
    current_local = rng.normal(scale=0.55, size=case.n_bus)
    current_error = forecast_deviation * np.tanh(
        0.70 * current_global + 0.30 * current_local,
    )
    current_scale = np.clip(
        profile[0] * (1.0 + current_error),
        load_scale_min,
        load_scale_max,
    )
    # PGLib 算例的 case.bus[:, 2] 为有功基准负荷，case.bus[:, 3] 为无功基准负荷。
    # 同一缩放系数同时作用于有功和无功，保持原始功率因数；零负荷节点仍为零。
    current_pd = case.bus[:, 2] * current_scale
    current_qd = case.bus[:, 3] * current_scale

    # 从 t=2 开始生成 S 条不同的预测轨迹。关键设计：所有场景在 t=1 共享
    # 同一 current_global/current_local，因此在 t=2 处从同一当前状态分叉；
    # 之后每条场景沿自己的 AR(1) 历史独立演化，避免用场景均值替代场景历史。
    # rho 控制相邻15分钟误差的相关性；tanh限制预测偏差，非对称clip限制
    # 系统绝对负荷水平。
    errors = forecast_deviation * np.tanh(_ar1_paths(
        rng, n_scenarios, horizon - 1, case.n_bus, rho,
        current_global, current_local,
    ))
    future_scale = np.clip(
        profile[1:][None, :, None] * (1.0 + errors),
        load_scale_min,
        load_scale_max,
    )

    # 同一个缩放因子同时作用于节点有功和无功负荷，从而保持PGLib基准算例
    # 中各节点的功率因数；零负荷节点经过缩放后仍保持为零。
    future_pd = case.bus[:, 2][None, None, :] * future_scale
    future_qd = case.bus[:, 3][None, None, :] * future_scale

    # 可再生能源在后续扩展中统一按负节点负荷处理：
    # P_net=P_load-P_renewable，Q_net=Q_load-Q_renewable。
    # 当前实验不考虑可再生能源，因此净负荷就是上述负荷。
    return ScenarioBundle(
        current_pd, current_qd, future_pd, future_qd, start_period,
    )


if __name__ == "__main__":
    bundle = generate_bundle(2026)
    print("96-period window start:", bundle.start_period)
    print("current load shape:", bundle.current_pd.shape)
    print("future scenario shape:", bundle.future_pd.shape)

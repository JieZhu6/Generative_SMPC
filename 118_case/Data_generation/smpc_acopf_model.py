"""PGLib IEEE 118 节点系统的两阶段场景型 SMPC--ACOPF 模型。

论文时刻 ``t=1`` 使用一组共享的确定性变量 ``pg1/qg1/vm1/va1``；论文时刻
``t=2,...,T`` 使用带场景索引的补救变量 ``pg/qg/vm/va``。Pyomo 集合 ``F``
采用 ``1,...,T-1``，因此代码中的 ``F=1`` 对应论文中的 ``t=2``。目标函数
由首时段成本与未来场景成本均值组成，所有场景同时出现在一个广义形式模型中，
并由一次 IPOPT 求解共同优化，而不是逐场景独立求解。

模型内部的功率变量采用标幺值；求解结果写入数据集前恢复为 MW/Mvar。
"""

import sys
from pathlib import Path

import numpy as np
import pyomo.environ as pyo
from scipy.optimize import root

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from Data_generation.case118_pglib import Case118, load_case118  # noqa: E402
from Data_generation.generate_smpc_scenarios import ScenarioBundle  # noqa: E402


IPOPT_PATH = Path("D:/anaconda/envs/py3.10/Library/bin/ipopt.exe")


def branch_states(case: Case118, vm: np.ndarray, va: np.ndarray) -> dict[str, np.ndarray]:
    """由节点电压计算每条支路两端的功率和电流幅值。

    参数
    ----
    case : Case118
        IEEE 118 节点网络数据，提供支路连接关系和两端复导纳。
    vm : np.ndarray, shape (n_bus,)
        节点电压幅值，单位 p.u.。
    va : np.ndarray, shape (n_bus,)
        节点电压相角，单位 rad。

    返回值
    ------
    dict[str, np.ndarray]
        ``pf/qf`` 和 ``pt/qt`` 分别为支路 from/to 端注入支路的有功、无功，
        单位 MW/Mvar；``if_mag/it_mag`` 为两端复电流幅值，单位 p.u.。每个
        数组的 shape 均为 ``(n_branch,)``。
    """
    voltage = vm * np.exp(1j * va)
    fbus = case.branch[:, 0].astype(int) - 1
    tbus = case.branch[:, 1].astype(int) - 1
    current_from = case.yff * voltage[fbus] + case.yft * voltage[tbus]
    current_to = case.ytf * voltage[fbus] + case.ytt * voltage[tbus]
    power_from = voltage[fbus] * np.conj(current_from) * case.base_mva
    power_to = voltage[tbus] * np.conj(current_to) * case.base_mva
    return {
        "pf": power_from.real,
        "qf": power_from.imag,
        "pt": power_to.real,
        "qt": power_to.imag,
        "if_mag": np.abs(current_from),
        "it_mag": np.abs(current_to),
    }


def nodal_injections(case: Case118, vm: np.ndarray, va: np.ndarray):
    """计算节点电压对应的 AC 网络净注入功率。

    参数
    ----
    case : Case118
        含节点导纳矩阵 ``Ybus`` 和功率基准的网络数据。
    vm, va : np.ndarray, shape (n_bus,)
        节点电压幅值（p.u.）和相角（rad）。

    返回值
    ------
    p_injection, q_injection : np.ndarray, shape (n_bus,)
        按“节点向网络注入为正”定义的有功、无功注入，单位 MW/Mvar。
    """
    voltage = vm * np.exp(1j * va)
    power = voltage * np.conj(case.ybus @ voltage) * case.base_mva
    return power.real, power.imag


def recover_first_stage(
    case: Case118,
    current_pd: np.ndarray,
    current_qd: np.ndarray,
    pg_pv: np.ndarray,
    vm_generator: np.ndarray,
    initial_vm: np.ndarray | None = None,
    initial_va: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """由神经网络控制量恢复首时段完整 AC 潮流状态。

    参数
    ----
    case : Case118
        IEEE 118 节点网络数据。
    current_pd, current_qd : np.ndarray, shape (n_bus,)
        ``t=1`` 的确定性节点有功、无功负荷，单位 MW/Mvar。
    pg_pv : np.ndarray, shape (n_pv,)
        PV 节点机组有功，按 ``case.pv_buses`` 顺序排列，单位 MW。同步调相机
        所在 PV 节点的该值为零；平衡节点有功由潮流方程恢复。
    vm_generator : np.ndarray, shape (n_gen,)
        平衡节点及各 PV 节点的给定电压幅值，按机组表顺序排列，单位 p.u.。
    initial_vm, initial_va : np.ndarray or None, shape (n_bus,)
        非线性潮流方程的初始电压幅值和相角。若为 ``None``，分别使用全 1
        幅值和全 0 相角；传入优化解可用于核对恢复结果。

    返回值
    ------
    dict[str, np.ndarray | float]
        包含平衡节点有功 ``pg_slack``、全部机组无功 ``qg``、PQ 节点电压
        ``vm_pq``、非平衡节点相角 ``va_nonref``、完整电压、支路两端功率和
        电流，以及潮流方程最大残差。

    说明
    ----
    未知量为全部非平衡节点相角和 PQ 节点电压幅值；方程为对应节点的
    有功与无功平衡方程。
    """
    ref, pv, pq = case.reference_bus, case.pv_buses, case.pq_buses
    gen_buses = case.generator_buses
    pg_by_bus = np.zeros(case.n_bus)
    pg_by_bus[pv] = pg_pv
    vm0 = np.ones(case.n_bus) if initial_vm is None else initial_vm.copy()
    va0 = np.zeros(case.n_bus) if initial_va is None else initial_va.copy()
    vm0[gen_buses] = vm_generator
    unknown0 = np.r_[va0[np.arange(case.n_bus) != ref], vm0[pq]]

    def residual(unknown):
        """组装潮流恢复方程残差；``unknown`` 为非平衡相角与 PQ 电压。"""
        va = np.zeros(case.n_bus)
        va[np.arange(case.n_bus) != ref] = unknown[: case.n_bus - 1]
        vm = vm0.copy()
        vm[pq] = unknown[case.n_bus - 1:]
        p_network, q_network = nodal_injections(case, vm, va)
        p_residual = pg_by_bus - current_pd - p_network
        q_residual = -current_qd - q_network
        return np.r_[p_residual[np.arange(case.n_bus) != ref], q_residual[pq]]

    result = root(residual, unknown0, method="hybr", tol=1e-10)
    if not result.success or np.max(np.abs(residual(result.x))) > 1e-6:
        raise RuntimeError(f"first-stage AC recovery failed: {result.message}")
    va = np.zeros(case.n_bus)
    va[np.arange(case.n_bus) != ref] = result.x[: case.n_bus - 1]
    vm = vm0.copy()
    vm[pq] = result.x[case.n_bus - 1:]
    p_network, q_network = nodal_injections(case, vm, va)
    pg_slack = current_pd[ref] + p_network[ref]
    qg = current_qd[gen_buses] + q_network[gen_buses]
    states = branch_states(case, vm, va)
    states.update({
        "pg_slack": np.asarray([pg_slack]),
        "qg": qg,
        "vm_pq": vm[pq],
        "va_nonref": va[np.arange(case.n_bus) != ref],
        "vm": vm,
        "va": va,
        "recovery_residual": float(np.max(np.abs(residual(result.x)))),
    })
    return states


class TwoStageSMPCAcopf:
    """可重复赋值求解的两阶段场景型 SMPC--ACOPF 广义形式模型。

    同一个对象只建立一次 Pyomo 模型，批量生成数据时通过可变参数替换负荷和
    初始机组状态。首时段决策不含场景索引，未来每个场景拥有独立补救变量，
    同时由首时段变量和跨时段爬坡约束耦合。
    """

    def __init__(
        self,
        horizon: int = 16,
        n_scenarios: int = 50,
        ramp_fraction: float = 0.25,
        ipopt_path: Path = IPOPT_PATH,
    ):
        """初始化网络、Pyomo 模型和 IPOPT 求解器。

        参数
        ----
        horizon : int, default=16
            预测域总时段数 ``T``；每个时段为 15 分钟，且必须不少于 2。
        n_scenarios : int, default=50
            ``t=2,...,T`` 共同考虑的随机场景数 ``|S|``。
        ramp_fraction : float, default=0.25
            单个 15 分钟时段允许的机组有功变化量，占机组有功调节范围
            ``Pmax-Pmin`` 的比例。
        ipopt_path : pathlib.Path, default=IPOPT_PATH
            IPOPT 可执行文件路径。
        """
        if horizon < 2 or n_scenarios < 1 or ramp_fraction <= 0.0:
            raise ValueError("invalid SMPC dimensions or ramp fraction")
        if not ipopt_path.is_file():
            raise FileNotFoundError(f"IPOPT executable not found: {ipopt_path}")
        self.case = load_case118()
        self.horizon = horizon
        self.n_scenarios = n_scenarios
        self.ramp_fraction = ramp_fraction
        self.model = self._build_model()
        self.solver = pyo.SolverFactory("ipopt", executable=str(ipopt_path))
        self.solver.options.update({
            "print_level": 0,
            "tol": 1e-8,
            "constr_viol_tol": 1e-8,
            "max_iter": 2500,
            "bound_push": 1e-8,
            "bound_frac": 1e-8,
            "honor_original_bounds": "yes",
        })

    def _build_model(self) -> pyo.ConcreteModel:
        """建立两阶段 SMPC--ACOPF 的 Pyomo 广义形式模型。

        返回值
        ------
        pyo.ConcreteModel
            含首时段共享变量、未来场景补救变量、AC 功率平衡、双端支路
            热稳限值、参考相角、支路相角差、机组出力/爬坡约束和期望成本。

        索引约定
        --------
        ``N/G/L/S`` 分别表示节点、机组、支路和场景；``F=1,...,T-1``
        表示论文中的未来时刻 ``t=2,...,T``。模型内功率均为 p.u.。
        """
        case = self.case
        m = pyo.ConcreteModel()

        # ------------------------------ 索引集合 ------------------------------
        # N、G、L、S 都采用 0 基索引，与 NumPy 网络数据保持一致。
        # F 故意从 1 开始：F=1 是首个随机未来时段，即论文时刻 t=2。
        m.N = pyo.RangeSet(0, case.n_bus - 1)
        m.G = pyo.RangeSet(0, case.n_gen - 1)
        m.L = pyo.RangeSet(0, len(case.branch) - 1)
        m.S = pyo.RangeSet(0, self.n_scenarios - 1)
        m.F = pyo.RangeSet(1, self.horizon - 1)

        # --------------------------- 样本相关可变参数 ---------------------------
        # current_* 无场景索引，对应确定的 t=1；future_* 带 s、t 索引，
        # 对应 t=2,...,T。previous_pg 仅用于独立 recourse 子问题的边界爬坡。
        # 可再生能源暂未单列；扩展时应将 P_load-P_renewable、
        # Q_load-Q_renewable 作为这里的节点净负荷参数。
        m.current_pd = pyo.Param(m.N, initialize=0.0, mutable=True)
        m.current_qd = pyo.Param(m.N, initialize=0.0, mutable=True)
        m.future_pd = pyo.Param(m.S, m.F, m.N, initialize=0.0, mutable=True)
        m.future_qd = pyo.Param(m.S, m.F, m.N, initialize=0.0, mutable=True)
        m.previous_pg = pyo.Param(m.G, initialize=0.0, mutable=True)

        # --------------------------- 变量边界与决策变量 -------------------------
        # P/Q 上下限来自 PGLib 机组表并除以 base_mva；节点电压边界来自 bus 表。
        # Pmin=Pmax=0 的同步调相机有功自动固定为零，但 qg 和电压仍可变化。
        pg_bounds = lambda _, g: (case.gen[g, 9] / case.base_mva, case.gen[g, 8] / case.base_mva)
        qg_bounds = lambda _, g: (case.gen[g, 4] / case.base_mva, case.gen[g, 3] / case.base_mva)
        vm_bounds = lambda _, i: (case.bus[i, 12], case.bus[i, 11])
        m.pg1 = pyo.Var(m.G, bounds=pg_bounds)
        m.qg1 = pyo.Var(m.G, bounds=qg_bounds)
        m.vm1 = pyo.Var(m.N, bounds=vm_bounds)
        m.va1 = pyo.Var(m.N, bounds=(-np.pi, np.pi))

        # 首时段变量 pg1/qg1/vm1/va1 不含场景索引，保证所有场景实施同一个
        # t=1 决策；未来补救变量则为每个 (s,t) 分别建立一组。
        m.pg = pyo.Var(m.S, m.F, m.G, bounds=lambda _, s, t, g: pg_bounds(_, g))
        m.qg = pyo.Var(m.S, m.F, m.G, bounds=lambda _, s, t, g: qg_bounds(_, g))
        m.vm = pyo.Var(m.S, m.F, m.N, bounds=lambda _, s, t, i: vm_bounds(_, i))
        m.va = pyo.Var(m.S, m.F, m.N, bounds=(-np.pi, np.pi))

        gens_at_bus = {
            i: [g for g in range(case.n_gen) if case.generator_buses[g] == i]
            for i in range(case.n_bus)
        }
        gmat, bmat = case.ybus.real, case.ybus.imag
        # Ybus 很稀疏；只遍历非零邻接可减少 Pyomo 表达式规模。
        neighbors = {
            i: np.flatnonzero(np.abs(case.ybus[i]) > 1e-14).tolist()
            for i in range(case.n_bus)
        }

        def balance(i, pg, qg, vm, va, pd, qd, reactive):
            """返回节点 ``i`` 的有功或无功 AC 平衡等式。

            ``pg/qg/vm/va/pd/qd`` 是按索引取值的局部访问函数，使同一公式
            可同时用于首时段变量和未来场景变量；``reactive=False`` 建立
            有功平衡，``True`` 建立无功平衡。当前没有独立可再生注入项，
            因而左端是“常规机组出力减净负荷”。
            """
            generation = sum((qg if reactive else pg)(g) for g in gens_at_bus[i])
            if reactive:
                injection = vm(i) * sum(
                    vm(j) * (gmat[i, j] * pyo.sin(va(i) - va(j))
                             - bmat[i, j] * pyo.cos(va(i) - va(j)))
                    for j in neighbors[i]
                )
                return generation - qd(i) == injection
            injection = vm(i) * sum(
                vm(j) * (gmat[i, j] * pyo.cos(va(i) - va(j))
                         + bmat[i, j] * pyo.sin(va(i) - va(j)))
                for j in neighbors[i]
            )
            return generation - pd(i) == injection

        # 约束 1：t=1 确定性节点有功平衡，每个节点仅建立一次、没有场景索引。
        m.p_balance1 = pyo.Constraint(
            m.N, rule=lambda model, i: balance(
                i, lambda g: model.pg1[g], lambda g: model.qg1[g],
                lambda j: model.vm1[j], lambda j: model.va1[j],
                lambda j: model.current_pd[j], lambda j: model.current_qd[j], False,
            ),
        )
        # 约束 2：t=1 确定性节点无功平衡，每个节点仅建立一次。
        m.q_balance1 = pyo.Constraint(
            m.N, rule=lambda model, i: balance(
                i, lambda g: model.pg1[g], lambda g: model.qg1[g],
                lambda j: model.vm1[j], lambda j: model.va1[j],
                lambda j: model.current_pd[j], lambda j: model.current_qd[j], True,
            ),
        )
        # 约束 3：t=2,...,T 各场景、各节点的 AC 有功平衡。
        m.p_balance = pyo.Constraint(
            m.S, m.F, m.N,
            rule=lambda model, s, t, i: balance(
                i, lambda g: model.pg[s, t, g], lambda g: model.qg[s, t, g],
                lambda j: model.vm[s, t, j], lambda j: model.va[s, t, j],
                lambda j: model.future_pd[s, t, j],
                lambda j: model.future_qd[s, t, j], False,
            ),
        )
        # 约束 4：t=2,...,T 各场景、各节点的 AC 无功平衡。
        m.q_balance = pyo.Constraint(
            m.S, m.F, m.N,
            rule=lambda model, s, t, i: balance(
                i, lambda g: model.pg[s, t, g], lambda g: model.qg[s, t, g],
                lambda j: model.vm[s, t, j], lambda j: model.va[s, t, j],
                lambda j: model.future_pd[s, t, j],
                lambda j: model.future_qd[s, t, j], True,
            ),
        )

        def flow(vm, va, ell, from_end, reactive):
            """返回支路 ``ell`` 指定端的有功或无功潮流表达式。

            ``vm/va`` 是节点变量访问函数；``from_end`` 选择 from 端或 to 端；
            ``reactive`` 选择无功或有功。公式直接使用 yff/yft/ytf/ytt，因此
            同时适用于普通线路和含非标准变比的变压器。返回值为 p.u.。
            """
            row = case.branch[ell]
            i, j = int(row[0]) - 1, int(row[1]) - 1
            if from_end:
                yii, yij = case.yff[ell], case.yft[ell]
            else:
                i, j = j, i
                yii, yij = case.ytt[ell], case.ytf[ell]
            delta = va(i) - va(j)
            if reactive:
                return -yii.imag * vm(i) ** 2 + vm(i) * vm(j) * (
                    yij.real * pyo.sin(delta) - yij.imag * pyo.cos(delta)
                )
            return yii.real * vm(i) ** 2 + vm(i) * vm(j) * (
                yij.real * pyo.cos(delta) + yij.imag * pyo.sin(delta)
            )

        # 支路潮流表达式：pf/qf 是 from 端，pt/qt 是 to 端。双端都计算是因为
        # 线路损耗使两端视在功率不同，热稳定约束必须分别检查。
        m.pf1 = pyo.Expression(m.L, rule=lambda model, ell: flow(lambda i: model.vm1[i], lambda i: model.va1[i], ell, True, False))
        m.qf1 = pyo.Expression(m.L, rule=lambda model, ell: flow(lambda i: model.vm1[i], lambda i: model.va1[i], ell, True, True))
        m.pt1 = pyo.Expression(m.L, rule=lambda model, ell: flow(lambda i: model.vm1[i], lambda i: model.va1[i], ell, False, False))
        m.qt1 = pyo.Expression(m.L, rule=lambda model, ell: flow(lambda i: model.vm1[i], lambda i: model.va1[i], ell, False, True))
        m.pf = pyo.Expression(m.S, m.F, m.L, rule=lambda model, s, t, ell: flow(lambda i: model.vm[s, t, i], lambda i: model.va[s, t, i], ell, True, False))
        m.qf = pyo.Expression(m.S, m.F, m.L, rule=lambda model, s, t, ell: flow(lambda i: model.vm[s, t, i], lambda i: model.va[s, t, i], ell, True, True))
        m.pt = pyo.Expression(m.S, m.F, m.L, rule=lambda model, s, t, ell: flow(lambda i: model.vm[s, t, i], lambda i: model.va[s, t, i], ell, False, False))
        m.qt = pyo.Expression(m.S, m.F, m.L, rule=lambda model, s, t, ell: flow(lambda i: model.vm[s, t, i], lambda i: model.va[s, t, i], ell, False, True))

        # 约束 5：支路双端热稳限值 P^2+Q^2 <= Smax^2。
        # thermal_*1 对应确定性 t=1；thermal_* 对应所有未来 (s,t)。
        m.thermal_f1 = pyo.Constraint(m.L, rule=lambda model, ell: model.pf1[ell] ** 2 + model.qf1[ell] ** 2 <= (case.branch[ell, 5] / case.base_mva) ** 2)
        m.thermal_t1 = pyo.Constraint(m.L, rule=lambda model, ell: model.pt1[ell] ** 2 + model.qt1[ell] ** 2 <= (case.branch[ell, 5] / case.base_mva) ** 2)
        m.thermal_f = pyo.Constraint(m.S, m.F, m.L, rule=lambda model, s, t, ell: model.pf[s, t, ell] ** 2 + model.qf[s, t, ell] ** 2 <= (case.branch[ell, 5] / case.base_mva) ** 2)
        m.thermal_t = pyo.Constraint(m.S, m.F, m.L, rule=lambda model, s, t, ell: model.pt[s, t, ell] ** 2 + model.qt[s, t, ell] ** 2 <= (case.branch[ell, 5] / case.base_mva) ** 2)
        # 约束 6：参考节点相角固定为 0，消除 AC 潮流相角的整体平移自由度。
        m.ref1 = pyo.Constraint(expr=m.va1[case.reference_bus] == 0.0)
        m.ref = pyo.Constraint(m.S, m.F, rule=lambda model, s, t: model.va[s, t, case.reference_bus] == 0.0)

        def angle_limits(va, ell, upper):
            """返回支路 ``ell`` 的相角差上限或下限约束。

            ``va`` 为节点相角访问函数；``upper=True`` 使用 branch 表的 angmax，
            否则使用 angmin。PGLib 中角度以 degree 存储，此处转为 rad。
            """
            row = case.branch[ell]
            delta = va(int(row[0]) - 1) - va(int(row[1]) - 1)
            return delta <= np.deg2rad(row[12]) if upper else delta >= np.deg2rad(row[11])

        # 约束 7：每条在线支路两端相角差的上下限，首时段和未来场景均施加。
        m.angle_low1 = pyo.Constraint(m.L, rule=lambda model, ell: angle_limits(lambda i: model.va1[i], ell, False))
        m.angle_high1 = pyo.Constraint(m.L, rule=lambda model, ell: angle_limits(lambda i: model.va1[i], ell, True))
        m.angle_low = pyo.Constraint(m.S, m.F, m.L, rule=lambda model, s, t, ell: angle_limits(lambda i: model.va[s, t, i], ell, False))
        m.angle_high = pyo.Constraint(m.S, m.F, m.L, rule=lambda model, s, t, ell: angle_limits(lambda i: model.va[s, t, i], ell, True))

        def ramp(g):
            """返回机组 ``g`` 单时段允许的有功变化量，单位 p.u.。"""
            return self.ramp_fraction * (case.gen[g, 8] - case.gen[g, 9]) / case.base_mva

        # 约束 8：独立 recourse 子问题的边界爬坡，完整 SMPC 中自动停用。
        # 仅在独立 recourse 子问题中激活，用于耦合已固定的 t=1 边界与 t=2。
        m.ramp_up1 = pyo.Constraint(m.G, rule=lambda model, g: model.pg1[g] - model.previous_pg[g] <= ramp(g))
        m.ramp_down1 = pyo.Constraint(m.G, rule=lambda model, g: model.previous_pg[g] - model.pg1[g] <= ramp(g))

        # 约束 9：从共享 t=1 决策到每个场景首个未来决策（论文 t=2）的
        # 上下爬坡限制。这组约束将所有未来场景与同一个首时段决策耦合。
        m.ramp_up2 = pyo.Constraint(m.S, m.G, rule=lambda model, s, g: model.pg[s, 1, g] - model.pg1[g] <= ramp(g))
        m.ramp_down2 = pyo.Constraint(m.S, m.G, rule=lambda model, s, g: model.pg1[g] - model.pg[s, 1, g] <= ramp(g))

        def future_ramp(model, s, t, g, up):
            """返回同一场景内相邻未来时段的单侧爬坡约束。

            ``model`` 为 Pyomo 模型；``s/t/g`` 为场景、未来时段和机组索引；
            ``up=True`` 建立上爬坡约束，否则建立下爬坡约束。``t=1`` 已由
            ramp_up2/ramp_down2 处理，因此在这里跳过。
            """
            if t == 1:
                return pyo.Constraint.Skip
            difference = model.pg[s, t, g] - model.pg[s, t - 1, g]
            return difference <= ramp(g) if up else -difference <= ramp(g)

        # 约束 10：各场景内部论文 t=2,...,T 的相邻时段上下爬坡限制。
        m.ramp_up = pyo.Constraint(m.S, m.F, m.G, rule=lambda model, s, t, g: future_ramp(model, s, t, g, True))
        m.ramp_down = pyo.Constraint(m.S, m.F, m.G, rule=lambda model, s, t, g: future_ramp(model, s, t, g, False))

        def cost(pg, g):
            """计算机组 ``g`` 在给定标幺有功 ``pg`` 下的单时段成本。"""
            power = case.base_mva * pg
            return case.gencost[g, 4] * power**2 + case.gencost[g, 5] * power + case.gencost[g, 6]

        # 目标函数：t=1 的确定性发电成本只计一次；t=2,...,T 的成本先对每个
        # 场景、时段和机组求和，再除以场景数，得到等概率场景下的期望成本。
        m.objective = pyo.Objective(expr=(
            sum(cost(m.pg1[g], g) for g in m.G)
            + sum(cost(m.pg[s, t, g], g) for s in m.S for t in m.F for g in m.G) / self.n_scenarios
        ))
        self._flat_start(m)
        return m

    def _flat_start(self, model=None) -> None:
        """将全部优化变量重置为平坦启动值。

        参数
        ----
        model : pyo.ConcreteModel or None, default=None
            需要初始化的模型；为 ``None`` 时使用当前对象的 ``self.model``。

        返回值
        ------
        None
            机组变量采用 PGLib 初始出力，电压幅值设为 1 p.u.、相角设为 0。
            该初始化只提供 IPOPT 起点，不改变变量边界或约束。
        """
        m = self.model if model is None else model
        case = self.case
        for g in range(case.n_gen):
            # 数据生成时会固定非平衡可调机组以采样不同边界。平坦启动不能
            # 覆盖固定值，否则 IPOPT 失败后的延拓求解会悄然改变采样标签。
            if not m.pg1[g].fixed:
                m.pg1[g].set_value(case.gen[g, 1] / case.base_mva)
            if not m.qg1[g].fixed:
                m.qg1[g].set_value(case.gen[g, 2] / case.base_mva)
        for i in range(case.n_bus):
            m.vm1[i].set_value(1.0)
            m.va1[i].set_value(0.0)
        for s in range(self.n_scenarios):
            for t in range(1, self.horizon):
                for g in range(case.n_gen):
                    m.pg[s, t, g].set_value(case.gen[g, 1] / case.base_mva)
                    m.qg[s, t, g].set_value(case.gen[g, 2] / case.base_mva)
                for i in range(case.n_bus):
                    m.vm[s, t, i].set_value(1.0)
                    m.va[s, t, i].set_value(0.0)

    def set_bundle(self, bundle: ScenarioBundle, blend: float = 1.0) -> None:
        """把一个样本的负荷和 recourse 边界状态写入 Pyomo 参数。

        参数
        ----
        bundle : ScenarioBundle
            当前负荷和后续多场景负荷。``previous_pg`` 非空时表示
            recourse 首时段之前的固定机组有功。
        blend : float, default=1.0
            场景离散程度的延拓系数。``0`` 将全部场景收缩到逐时场景均值，
            ``1`` 使用原始场景；中间值用于 IPOPT 失败后的逐步延拓求解。

        返回值
        ------
        None
            仅更新模型参数，不执行求解。
        """
        case, m = self.case, self.model
        expected = (self.n_scenarios, self.horizon - 1, case.n_bus)
        if bundle.future_pd.shape != expected or bundle.future_qd.shape != expected:
            raise ValueError(f"future loads must have shape {expected}")
        mean_pd = bundle.future_pd.mean(axis=0, keepdims=True)
        mean_qd = bundle.future_qd.mean(axis=0, keepdims=True)
        future_pd = mean_pd + blend * (bundle.future_pd - mean_pd)
        future_qd = mean_qd + blend * (bundle.future_qd - mean_qd)
        for i in range(case.n_bus):
            m.current_pd[i] = float(bundle.current_pd[i] / case.base_mva)
            m.current_qd[i] = float(bundle.current_qd[i] / case.base_mva)
        if bundle.previous_pg is None:
            m.ramp_up1.deactivate()
            m.ramp_down1.deactivate()
        else:
            if bundle.previous_pg.shape != (case.n_gen,):
                raise ValueError("previous_pg must have shape (n_gen,)")
            for g in range(case.n_gen):
                m.previous_pg[g] = float(bundle.previous_pg[g] / case.base_mva)
            m.ramp_up1.activate()
            m.ramp_down1.activate()
        for s in range(self.n_scenarios):
            for t in range(1, self.horizon):
                for i in range(case.n_bus):
                    m.future_pd[s, t, i] = float(future_pd[s, t - 1, i] / case.base_mva)
                    m.future_qd[s, t, i] = float(future_qd[s, t - 1, i] / case.base_mva)

    def _attempt(self, tee=False) -> bool:
        """调用 IPOPT 完成一次求解尝试。

        参数
        ----
        tee : bool, default=False
            是否把 IPOPT 迭代信息实时打印到终端。

        返回值
        ------
        bool
            终止条件包含 ``optimal`` 时返回 ``True`` 并载入解；否则返回
            ``False``，模型中不载入该次求解结果。
        """
        result = self.solver.solve(self.model, tee=tee, load_solutions=False)
        if "optimal" not in str(result.solver.termination_condition).lower():
            return False
        self.model.solutions.load_from(result)
        return True

    def _optimize(self, bundle: ScenarioBundle, tee=False) -> None:
        """求解当前样本，但不决定需要从解中保存哪些数据。

        将“数值求解”和“结果落盘所需的提取”分开后，数据生成程序可以只读取
        recourse 成本和首时段边界有功，而不构造或返回完整 AC 状态。完整提取
        接口仍保留，便于与原来的 extensive-form 数据进行核对。
        """
        self.set_bundle(bundle)
        try:
            solved = self._attempt(tee)
        except Exception:
            solved = False
        if not solved:
            self._flat_start()
            for blend in np.linspace(0.0, 1.0, 5):
                self.set_bundle(bundle, float(blend))
                try:
                    solved = self._attempt(tee)
                except Exception:
                    solved = False
                if not solved:
                    break
        if not solved:
            raise RuntimeError("IPOPT did not reach an optimal SMPC solution")

        # 延拓求解的最后一步必须对应原始场景。重新赋值不会改变优化变量，
        # 只是保证后续残差复核读取的参数与传入 bundle 完全一致。
        self.set_bundle(bundle)

    def solve(self, bundle: ScenarioBundle, tee=False, save_recourse=False) -> dict:
        """求解一个两阶段 SMPC 样本并提取完整监督学习数据。

        参数
        ----
        bundle : ScenarioBundle
            待优化的当前负荷、初始机组功率和未来负荷场景。
        tee : bool, default=False
            是否显示 IPOPT 迭代输出。
        save_recourse : bool, default=False
            是否在结果中保存所有未来场景的完整补救状态。即使为 ``False``，
            函数仍遍历未来解并验证功率平衡与支路热稳约束。

        返回值
        ------
        dict
            ``extract`` 生成的首时段控制量、恢复状态、数值校验指标，以及
            可选的未来补救状态。

        说明
        ----
        首次直接求解原场景；若失败，则从场景均值开始，通过五个 ``blend``
        水平逐步恢复完整场景离散度。该过程只是非线性求解延拓，不改变最终模型。
        """
        self._optimize(bundle, tee)
        return self.extract(bundle, save_recourse)

    def solve_minimal(self, bundle: ScenarioBundle, tee=False, return_first_pg=False) -> dict:
        """求解样本并只返回数据生成真正需要的标量和边界状态。

        参数
        ----
        bundle : ScenarioBundle
            当前时段、初始机组功率和后续单/多场景负荷。
        tee : bool, default=False
            是否显示 IPOPT 输出。
        return_first_pg : bool, default=False
            是否返回模型首时段的全部机组有功。生成 VFA 的一阶段边界标签时
            需要该量；逐场景求 recourse 标签时不需要。

        返回值
        ------
        dict
            始终包含 ``objective``、``max_balance_residual`` 和
            ``max_thermal_overload``。仅在 ``return_first_pg=True`` 时增加
            ``first_pg``。不会返回 Qg、电压、相角、支路潮流或 recourse 状态。
        """
        self._optimize(bundle, tee)
        return self.extract_minimal(bundle, return_first_pg)

    def extract_minimal(self, bundle: ScenarioBundle, return_first_pg=False) -> dict:
        """从当前解计算最小结果及必要的数值质量指标。

        AC 状态仍是优化问题的内部变量，检查功率平衡和热稳时必须临时读取；
        但这些数组在本函数返回前即被丢弃，不会进入训练数据集。
        """
        m, case = self.model, self.case

        def state_metrics(pg, qg, vm, va, pd, qd):
            """返回单个运行点的最大平衡残差和双端热稳越限。"""
            pnet, qnet = nodal_injections(case, vm, va)
            pgen, qgen = np.zeros(case.n_bus), np.zeros(case.n_bus)
            for g, bus in enumerate(case.generator_buses):
                pgen[bus] += pg[g]
                qgen[bus] += qg[g]
            balance = max(
                np.max(np.abs(pgen - pd - pnet)),
                np.max(np.abs(qgen - qd - qnet)),
            )
            branch = branch_states(case, vm, va)
            apparent = np.maximum(
                np.hypot(branch["pf"], branch["qf"]),
                np.hypot(branch["pt"], branch["qt"]),
            )
            overload = np.maximum(apparent - case.branch[:, 5], 0.0).max()
            return float(balance), float(overload)

        pg1 = np.array([pyo.value(m.pg1[g]) for g in m.G]) * case.base_mva
        qg1 = np.array([pyo.value(m.qg1[g]) for g in m.G]) * case.base_mva
        vm1 = np.array([pyo.value(m.vm1[i]) for i in m.N])
        va1 = np.array([pyo.value(m.va1[i]) for i in m.N])
        max_balance, max_overload = state_metrics(
            pg1, qg1, vm1, va1, bundle.current_pd, bundle.current_qd,
        )

        # 即使不保存未来状态，也逐时段复核 AC 平衡和线路热稳，避免把仅由
        # 求解器终止标志判断的低质量解写入 VFA 标签。
        for s in range(self.n_scenarios):
            for t in range(1, self.horizon):
                pg = np.array([pyo.value(m.pg[s, t, g]) for g in m.G]) * case.base_mva
                qg = np.array([pyo.value(m.qg[s, t, g]) for g in m.G]) * case.base_mva
                vm = np.array([pyo.value(m.vm[s, t, i]) for i in m.N])
                va = np.array([pyo.value(m.va[s, t, i]) for i in m.N])
                balance, overload = state_metrics(
                    pg, qg, vm, va,
                    bundle.future_pd[s, t - 1], bundle.future_qd[s, t - 1],
                )
                max_balance = max(max_balance, balance)
                max_overload = max(max_overload, overload)

        result = {
            "objective": float(pyo.value(m.objective)),
            "max_balance_residual": max_balance,
            "max_thermal_overload": max_overload,
        }
        if return_first_pg:
            result["first_pg"] = pg1
        return result

    def extract(self, bundle: ScenarioBundle, save_recourse=False) -> dict:
        """从已求解模型中提取控制标签、恢复状态和可行性指标。

        参数
        ----
        bundle : ScenarioBundle
            与当前模型参数对应的输入样本，用于潮流恢复和残差复核。
        save_recourse : bool, default=False
            若为 ``True``，额外返回 shape 为
            ``(n_scenarios, horizon-1, ...)`` 的未来机组、电压、支路功率和
            支路电流；若为 ``False``，仅保存首时段数据。

        返回值
        ------
        dict
            ``control_full`` 保留所有 PV 母线机组有功及 PV/平衡母线电压；
            ``control`` 只保留非平衡可调机组有功及 PV/平衡母线电压，作为
            论文自由变量。``recovered`` 保存首时段潮流恢复状态。
        """
        m, case = self.model, self.case
        pg1 = np.array([pyo.value(m.pg1[g]) for g in m.G]) * case.base_mva
        qg1 = np.array([pyo.value(m.qg1[g]) for g in m.G]) * case.base_mva
        vm1 = np.array([pyo.value(m.vm1[i]) for i in m.N])
        va1 = np.array([pyo.value(m.va1[i]) for i in m.N])
        first_branch = branch_states(case, vm1, va1)
        # 完整标签保留所有 PV 节点有功；紧凑标签只保留非平衡可调机组有功，
        # 再拼接 PV 与平衡母线电压，与论文中的自由变量 u 一致。
        pg_pv = np.array([pg1[np.flatnonzero(case.generator_buses == bus)[0]] for bus in case.pv_buses])
        vm_generator = vm1[case.generator_buses]
        recovered = recover_first_stage(
            case, bundle.current_pd, bundle.current_qd, pg_pv, vm_generator, vm1, va1,
        )
        control_full = np.r_[pg_pv, vm1[case.pv_buses], vm1[case.reference_bus]]
        control = np.r_[
            pg1[case.nonreference_active_generators],
            vm1[case.voltage_control_buses],
        ]
        # 使用监督标签重新解一次标准 AC 潮流，并与 OPF 首时段解比较。
        # recovery_difference 越小，说明所选控制输出足以恢复其余潮流状态。
        comparison = max(
            abs(float(recovered["pg_slack"][0]) - pg1[case.reference_generator]),
            np.max(np.abs(recovered["qg"] - qg1)),
            np.max(np.abs(recovered["vm"] - vm1)),
            np.max(np.abs(recovered["va"] - va1)),
        )
        def residual(pg, qg, vm, va, pd, qd):
            """返回一组状态的最大节点 P/Q 平衡残差，单位 MW/Mvar。

            参数依次为全部机组 P/Q、全部节点电压幅值/相角和节点 P/Q 负荷。
            """
            pnet, qnet = nodal_injections(case, vm, va)
            pgen, qgen = np.zeros(case.n_bus), np.zeros(case.n_bus)
            for g, bus in enumerate(case.generator_buses):
                pgen[bus] += pg[g]
                qgen[bus] += qg[g]
            return max(np.max(np.abs(pgen - pd - pnet)),
                       np.max(np.abs(qgen - qd - qnet)))

        # 校验指标覆盖首时段和所有未来场景：最大节点功率平衡残差，以及
        # 支路任一端超过 PGLib rateA 的最大视在功率（单位 MVA）。
        max_residual = residual(pg1, qg1, vm1, va1, bundle.current_pd, bundle.current_qd)
        rate = case.branch[:, 5]
        max_overload = float(np.maximum(
            np.maximum(np.hypot(first_branch["pf"], first_branch["qf"]),
                       np.hypot(first_branch["pt"], first_branch["qt"])) - rate,
            0.0,
        ).max())
        result = {
            "pg1": pg1, "qg1": qg1, "vm1": vm1, "va1": va1,
            "control_full": control_full, "control": control,
            "objective": float(pyo.value(m.objective)),
            "recovery_difference": float(comparison),
            "max_balance_residual": float(max_residual),
            "max_thermal_overload": max_overload,
            "recovered": recovered,
            "first_branch": first_branch,
        }
        # recourse 仅在显式请求时分配，以避免默认 7000×50×15 数据集占用
        # 大量磁盘；无论是否保存，下面仍逐场景复核可行性。
        recourse = None
        if save_recourse:
            shape_state = (self.n_scenarios, self.horizon - 1)
            recourse = {
                "pg": np.empty(shape_state + (case.n_gen,)),
                "qg": np.empty(shape_state + (case.n_gen,)),
                "vm": np.empty(shape_state + (case.n_bus,)),
                "va": np.empty(shape_state + (case.n_bus,)),
            }
            for key in ("pf", "qf", "pt", "qt", "if_mag", "it_mag"):
                recourse[key] = np.empty(shape_state + (len(case.branch),))
        for s in range(self.n_scenarios):
            for t in range(1, self.horizon):
                pg = np.array([pyo.value(m.pg[s, t, g]) for g in m.G]) * case.base_mva
                qg = np.array([pyo.value(m.qg[s, t, g]) for g in m.G]) * case.base_mva
                vm = np.array([pyo.value(m.vm[s, t, i]) for i in m.N])
                va = np.array([pyo.value(m.va[s, t, i]) for i in m.N])
                branch = branch_states(case, vm, va)
                max_residual = max(
                    max_residual,
                    residual(pg, qg, vm, va, bundle.future_pd[s, t - 1], bundle.future_qd[s, t - 1]),
                )
                max_overload = max(
                    max_overload,
                    float(np.maximum(
                        np.maximum(np.hypot(branch["pf"], branch["qf"]),
                                   np.hypot(branch["pt"], branch["qt"])) - rate,
                        0.0,
                    ).max()),
                )
                if recourse is not None:
                    for key, value in (("pg", pg), ("qg", qg), ("vm", vm), ("va", va)):
                        recourse[key][s, t - 1] = value
                    for key in ("pf", "qf", "pt", "qt", "if_mag", "it_mag"):
                        recourse[key][s, t - 1] = branch[key]
        result["max_balance_residual"] = float(max_residual)
        result["max_thermal_overload"] = float(max_overload)
        if recourse is not None:
            result["recourse"] = recourse
        return result


class FirstStageAcopf:
    """仅保留 t=1 AC 约束的可重复求解模型。

    该模型用于采样可行一阶段边界
    ``TwoStageSMPCAcopf`` 的首时段 AC 方程、设备限制和支路约束。
    """

    def __init__(self, ipopt_path: Path = IPOPT_PATH):
        """构建仅含 t=1 物理约束的 IPOPT 模型。"""
        self.base = TwoStageSMPCAcopf(
            horizon=2, n_scenarios=1, ipopt_path=ipopt_path,
        )
        self.case = self.base.case
        self.model = self.base.model
        self.solver = self.base.solver

        # 去掉 t=2 变量相关约束，只保留共享的 t=1 ACOPF 可行域。
        for name in (
            "p_balance", "q_balance", "thermal_f", "thermal_t", "ref",
            "angle_low", "angle_high", "ramp_up2", "ramp_down2",
            "ramp_up", "ramp_down",
        ):
            getattr(self.model, name).deactivate()
        self.model.objective.deactivate()
        self.model.first_stage_objective = pyo.Objective(
            expr=self._generation_cost(),
        )

    def _generation_cost(self):
        """返回 t=1 全部机组的二次发电成本表达式。"""
        case, m = self.case, self.model
        return sum(
            case.gencost[g, 4] * (case.base_mva * m.pg1[g]) ** 2
            + case.gencost[g, 5] * case.base_mva * m.pg1[g]
            + case.gencost[g, 6]
            for g in m.G
        )

    def set_load(self, current_pd: np.ndarray, current_qd: np.ndarray) -> None:
        """写入 t=1 节点负荷，并确保不启用 t=0 爬坡约束。"""
        zeros = np.zeros((1, 1, self.case.n_bus))
        bundle = ScenarioBundle(
            current_pd=current_pd,
            current_qd=current_qd,
            future_pd=zeros,
            future_qd=zeros,
            start_period=0,
        )
        self.base.set_bundle(bundle)

    def solve(self, current_pd: np.ndarray, current_qd: np.ndarray,
              tee: bool = False) -> dict:
        """求解 t=1 ACOPF，并返回一阶段边界与数值质量指标。"""
        self.set_load(current_pd, current_qd)
        if tee:
            self.solver.options["print_level"] = 5
        try:
            solved = self.base._attempt(tee)
        except Exception as error:
            if tee:
                print(f"First-stage ACOPF IPOPT error: {error}")
            solved = False
        if not solved:
            self.base._flat_start()
            try:
                solved = self.base._attempt(tee)
            except Exception as error:
                if tee:
                    print(f"First-stage ACOPF retry error: {error}")
                solved = False
        if tee:
            self.solver.options["print_level"] = 0
        if not solved:
            raise RuntimeError("IPOPT did not solve the first-stage ACOPF")
        return self.extract(current_pd, current_qd)

    def extract(self, current_pd: np.ndarray, current_qd: np.ndarray) -> dict:
        """提取 t=1 解，并复核 AC 平衡与支路热稳越限。"""
        case, m = self.case, self.model
        pg = np.array([pyo.value(m.pg1[g]) for g in m.G]) * case.base_mva
        qg = np.array([pyo.value(m.qg1[g]) for g in m.G]) * case.base_mva
        vm = np.array([pyo.value(m.vm1[i]) for i in m.N])
        va = np.array([pyo.value(m.va1[i]) for i in m.N])
        pnet, qnet = nodal_injections(case, vm, va)
        pgen, qgen = np.zeros(case.n_bus), np.zeros(case.n_bus)
        for g, bus in enumerate(case.generator_buses):
            pgen[bus] += pg[g]
            qgen[bus] += qg[g]
        balance = max(
            np.max(np.abs(pgen - current_pd - pnet)),
            np.max(np.abs(qgen - current_qd - qnet)),
        )
        branch = branch_states(case, vm, va)
        apparent = np.maximum(
            np.hypot(branch["pf"], branch["qf"]),
            np.hypot(branch["pt"], branch["qt"]),
        )
        return {
            "objective": float(pyo.value(self.model.first_stage_objective)),
            "first_pg": pg,
            "first_qg": qg,
            "vm": vm,
            "va": va,
            "max_balance_residual": float(balance),
            "max_thermal_overload": float(
                np.maximum(apparent - case.branch[:, 5], 0.0).max()
            ),
        }


class ValueApproximatedSMPCAcopf(FirstStageAcopf):
    """论文式 (10) 的价值函数缩减 SMPC 模型。

    一阶段仍使用精确 AC 约束，目标中的后续成本由固定权重的 SiLU MLP
    计算。未来场景是常量，因此第一层中它们的线性贡献预先合并到偏置项。
    """

    def __init__(
        self,
        weights: list[np.ndarray],
        biases: list[np.ndarray],
        pg_min: np.ndarray,
        pg_max: np.ndarray,
        future_min: np.ndarray,
        future_max: np.ndarray,
        cost_min: float,
        cost_max: float,
        n_scenarios: int,
        ipopt_path: Path = IPOPT_PATH,
    ):
        """构建缩减 SMPC；各数组分别为 VFA 权重、偏置和训练集归一化参数。"""
        super().__init__(ipopt_path)
        if n_scenarios < 1:
            raise ValueError("n_scenarios must be positive")
        self.n_scenarios = n_scenarios
        self.pg_min = np.asarray(pg_min, dtype=float)
        self.pg_max = np.asarray(pg_max, dtype=float)
        self.future_min = np.asarray(future_min, dtype=float)
        self.future_max = np.asarray(future_max, dtype=float)
        self.cost_min = float(cost_min)
        self.cost_max = float(cost_max)
        self.future_values = None
        self.solver.options["max_iter"] = 600
        self.solver.options["max_cpu_time"] = 120

        self.model.first_stage_objective.deactivate()
        self._build_network(weights, biases)
        self.update_network(weights, biases)

    def _build_network(self, weights: list[np.ndarray], biases: list[np.ndarray]) -> None:
        """在 Pyomo 中建立逐层 SiLU 网络等式和式 (10) 目标。"""
        if len(weights) != len(biases) or len(weights) < 2:
            raise ValueError("VFA requires hidden layers and one output layer")
        n_active = len(self.case.active_generators)
        if weights[0].shape[1] <= n_active:
            raise ValueError("VFA input must contain Pg and future-load features")

        m = self.model
        m.VS = pyo.RangeSet(0, self.n_scenarios - 1)
        m.H0 = pyo.RangeSet(0, weights[0].shape[0] - 1)
        m.vfa_w0 = pyo.Param(m.H0, range(n_active), mutable=True, initialize=0.0)
        m.vfa_b0 = pyo.Param(m.VS, m.H0, mutable=True, initialize=0.0)
        m.vfa_z0 = pyo.Var(m.VS, m.H0, initialize=0.0)
        m.vfa_a0 = pyo.Var(m.VS, m.H0, initialize=0.0)
        active = self.case.active_generators
        pg_span = np.maximum(self.pg_max - self.pg_min, 1e-12)

        def normalized_pg(j):
            power = self.case.base_mva * m.pg1[int(active[j])]
            return 2.0 * (power - self.pg_min[j]) / pg_span[j] - 1.0

        m.vfa_affine0 = pyo.Constraint(
            m.VS, m.H0,
            rule=lambda model, s, j: model.vfa_z0[s, j]
            == sum(model.vfa_w0[j, k] * normalized_pg(k) for k in range(n_active))
            + model.vfa_b0[s, j],
        )
        m.vfa_silu0 = pyo.Constraint(
            m.VS, m.H0,
            rule=lambda model, s, j: model.vfa_a0[s, j]
            == 0.5 * model.vfa_z0[s, j]
            * (1.0 + pyo.tanh(0.5 * model.vfa_z0[s, j])),
        )

        previous = m.vfa_a0
        self.hidden_components = []
        self.hidden_states = []
        for layer in range(1, len(weights) - 1):
            n_in, n_out = weights[layer].shape[1], weights[layer].shape[0]
            units = pyo.RangeSet(0, n_out - 1)
            inputs = pyo.RangeSet(0, n_in - 1)
            weight = pyo.Param(units, inputs, mutable=True, initialize=0.0)
            bias = pyo.Param(units, mutable=True, initialize=0.0)
            pre = pyo.Var(m.VS, units, initialize=0.0)
            activation = pyo.Var(m.VS, units, initialize=0.0)
            setattr(m, f"vfa_units_{layer}", units)
            setattr(m, f"vfa_inputs_{layer}", inputs)
            setattr(m, f"vfa_weight_{layer}", weight)
            setattr(m, f"vfa_bias_{layer}", bias)
            setattr(m, f"vfa_z_{layer}", pre)
            setattr(m, f"vfa_a_{layer}", activation)
            affine = pyo.Constraint(
                m.VS, units,
                rule=lambda model, s, j, w=weight, b=bias, a=previous, ni=n_in, z=pre:
                z[s, j] == sum(w[j, k] * a[s, k] for k in range(ni)) + b[j],
            )
            silu = pyo.Constraint(
                m.VS, units,
                rule=lambda model, s, j, z=pre, a=activation:
                a[s, j] == 0.5 * z[s, j] * (1.0 + pyo.tanh(0.5 * z[s, j])),
            )
            setattr(m, f"vfa_affine_{layer}", affine)
            setattr(m, f"vfa_silu_{layer}", silu)
            self.hidden_components.append((weight, bias))
            self.hidden_states.append((pre, activation))
            previous = activation

        output_weight = np.asarray(weights[-1])
        if output_weight.shape[0] != 1:
            raise ValueError("VFA output must be scalar")
        m.vfa_output_weight = pyo.Param(
            range(output_weight.shape[1]), mutable=True, initialize=0.0,
        )
        m.vfa_output_bias = pyo.Param(mutable=True, initialize=0.0)
        m.vfa_normalized = pyo.Expression(
            m.VS,
            rule=lambda model, s: sum(
                model.vfa_output_weight[k] * previous[s, k]
                for k in range(output_weight.shape[1])
            ) + model.vfa_output_bias,
        )
        cost_range = max(self.cost_max - self.cost_min, 1e-12)
        m.vfa_cost = pyo.Expression(
            m.VS,
            rule=lambda model, s: self.cost_min
            + 0.5 * cost_range * (model.vfa_normalized[s] + 1.0),
        )
        m.reduced_objective = pyo.Objective(
            expr=self._generation_cost()
            + sum(m.vfa_cost[s] for s in m.VS) / self.n_scenarios,
        )

    def update_network(self, weights: list[np.ndarray], biases: list[np.ndarray]) -> None:
        """在不重建 AC 模型的情况下更新已训练 VFA 的所有权重。"""
        weights = [np.asarray(value, dtype=float) for value in weights]
        biases = [np.asarray(value, dtype=float) for value in biases]
        self.weights, self.biases = weights, biases
        n_active = len(self.case.active_generators)
        for j in range(weights[0].shape[0]):
            for k in range(n_active):
                self.model.vfa_w0[j, k] = float(weights[0][j, k])
        for layer, (weight_param, bias_param) in enumerate(self.hidden_components, start=1):
            for j in range(weights[layer].shape[0]):
                bias_param[j] = float(biases[layer][j])
                for k in range(weights[layer].shape[1]):
                    weight_param[j, k] = float(weights[layer][j, k])
        for k, value in enumerate(weights[-1][0]):
            self.model.vfa_output_weight[k] = float(value)
        self.model.vfa_output_bias.set_value(float(biases[-1][0]))
        if self.future_values is not None:
            self.set_scenarios(self.future_values)

    def set_scenarios(self, future_load: np.ndarray) -> None:
        """写入当前场景树，并更新 VFA 第一层的场景偏置。"""
        future_load = np.asarray(future_load, dtype=float)
        if future_load.shape[0] != self.n_scenarios or future_load.shape[2:] != self.future_min.shape:
            raise ValueError(
                f"future_load must have shape (S,T-1,n_load,2), got {future_load.shape}"
            )
        self.future_values = future_load.copy()
        span = np.maximum(self.future_max - self.future_min, 1e-12)
        normalized = 2.0 * (future_load - self.future_min) / span - 1.0
        normalized = normalized.reshape(self.n_scenarios, -1)
        n_active = len(self.case.active_generators)
        if normalized.shape[1] != self.weights[0].shape[1] - n_active:
            raise ValueError("future-load dimension does not match the trained VFA")
        offset = normalized @ self.weights[0][:, n_active:].T + self.biases[0]
        for s in range(self.n_scenarios):
            for j in range(offset.shape[1]):
                self.model.vfa_b0[s, j] = float(offset[s, j])

    def _initialize_network(self) -> None:
        """根据当前 Pg 和场景前向计算 MLP，为 IPOPT 提供满足网络等式的初值。"""
        m = self.model
        active = self.case.active_generators
        pg = np.array([pyo.value(m.pg1[int(g)]) for g in active]) * self.case.base_mva
        pg = 2.0 * (pg - self.pg_min) / np.maximum(self.pg_max - self.pg_min, 1e-12) - 1.0
        offset = np.array([
            [pyo.value(m.vfa_b0[s, j]) for j in m.H0] for s in m.VS
        ])
        pre = pg @ self.weights[0][:, :len(active)].T + offset
        activation = 0.5 * pre * (1.0 + np.tanh(0.5 * pre))
        for s in range(self.n_scenarios):
            for j in range(pre.shape[1]):
                m.vfa_z0[s, j].set_value(float(pre[s, j]))
                m.vfa_a0[s, j].set_value(float(activation[s, j]))
        for layer, (pre_var, activation_var) in enumerate(self.hidden_states, start=1):
            pre = activation @ self.weights[layer].T + self.biases[layer]
            activation = 0.5 * pre * (1.0 + np.tanh(0.5 * pre))
            for s in range(self.n_scenarios):
                for j in range(pre.shape[1]):
                    pre_var[s, j].set_value(float(pre[s, j]))
                    activation_var[s, j].set_value(float(activation[s, j]))

    def solve(self, current_pd: np.ndarray, current_qd: np.ndarray,
              future_load: np.ndarray, tee: bool = False) -> dict:
        """求解式 (10)，返回优化诱导的一阶段边界和逐场景 VFA 值。"""
        self.set_scenarios(future_load)
        self.set_load(current_pd, current_qd)
        self._initialize_network()
        if tee:
            self.solver.options["print_level"] = 5
        try:
            solved = self.base._attempt(tee)
        except Exception as error:
            if tee:
                print(f"Reduced SMPC IPOPT error: {error}")
            solved = False
        if not solved:
            self.base._flat_start()
            self._initialize_network()
            try:
                solved = self.base._attempt(tee)
            except Exception as error:
                if tee:
                    print(f"Reduced SMPC retry error: {error}")
                solved = False
        if tee:
            self.solver.options["print_level"] = 0
        if not solved:
            raise RuntimeError("IPOPT did not solve the value-approximated SMPC")
        result = self.extract(current_pd, current_qd)
        result["predicted_recourse"] = np.array([
            pyo.value(self.model.vfa_cost[s]) for s in self.model.VS
        ])
        result["objective"] = float(pyo.value(self.model.reduced_objective))
        return result


if __name__ == "__main__":
    from Data_generation.generate_smpc_scenarios import generate_bundle

    test_bundle = generate_bundle(2026, horizon=16, n_scenarios=20)
    solution = TwoStageSMPCAcopf(horizon=16, n_scenarios=20).solve(test_bundle)
    print("objective:", solution["objective"])
    print("control:", solution["control"])
    print("recovery difference:", solution["recovery_difference"])

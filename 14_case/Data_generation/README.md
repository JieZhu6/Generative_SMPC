# IEEE-14 DECS 与条件随机生成器

代码实现论文中的 differentiable equality completion surrogate（DECS），并将
冻结的 DECS 集成到 conditional neural ODE generator。代码采用简洁的论文复现
结构：基础负荷数据、pandapower 潮流标签、DECS 训练、generator 训练。

本文档中的命令默认使用以下 UV Python 3.12.12 解释器：

```text
C:\Users\JieZhu\AppData\Roaming\uv\python\cpython-3.12.12-windows-x86_64-none\python.exe
```

## 1. 生成 SMPC 负荷场景

```powershell
& "C:\Users\JieZhu\AppData\Roaming\uv\python\cpython-3.12.12-windows-x86_64-none\python.exe" Data_generation/generate_smpc_dataset.py `
    --n-instances 1000 --n-scenarios 20 --horizon 16
```

基础数据包含：

```text
current_load.npy       [N,n_load_bus,2]
future_load.npy        [N,S,T-1,n_load_bus,2]
future_pool.npy        [N,3,T-1,n_load_bus,2]
normalization_parameters.npz
metadata.json
split/{train,validation,test}_indices.npy
```

负荷 P/Q 使用同一个节点缩放因子，保持 PGLib 基准功率因数。默认扰动范围为
基准负荷的 `[0.85,1.15]`。

## 2. 用 PGM batch 生成固定 PV 的 DECS 标签

```powershell
& "C:\Users\JieZhu\AppData\Roaming\uv\python\cpython-3.12.12-windows-x86_64-none\python.exe" Data_generation/generate_decs_dataset.py `
    --base-data Data_generation/data/e2e14_N5000_S20_T16 `
    --n-samples 10000 --batch-size 1000 `
    --load-range-extension 0.02 --free-range-extension 0.05
```

采样规则：

- 不复用基础数据中的 current/future 经验样本；全部运行点直接生成；
- 每个负荷节点在基础场景生成器的全时段理论缩放范围内采样，P/Q 使用同一
  缩放因子；默认在上下两侧各外扩原跨度的 2%；
- free variables 默认在神经网络投影物理范围的上下两侧各外扩 5%，覆盖边界
  邻域；生成模型自身的投影范围仍保持原物理边界不变；
- PGM 默认每批求解 1000 个固定 PV 潮流，`--batch-size` 可直接调整；
- Qg 越界只影响后续可行性，不触发 PV 到 PQ 的转换。

DECS 输入和输出分别为：

```text
rho = [p_PV, V_PV, V_ref, p_PQ, q_PQ]   [N,27]
chi = [theta_PV, theta_PQ, V_PQ]         [N,22]
```

有功/无功 specification 使用 p.u.，角度使用 rad，电压使用 p.u.。

## 3. 训练 DECS

```powershell
& "C:\Users\JieZhu\AppData\Roaming\uv\python\cpython-3.12.12-windows-x86_64-none\python.exe" Neural_network/train_decs.py `
    --data Data_generation/data/e2e14_decs_pgm_fixedpv_N10000 `
    --hidden-dims 128 128 --lambda-physics 1.0 `
    --patience 30 --min-delta 0.0
```

默认标签维度为 `27 -> 22`，隐藏层使用 ReLU。损失为标准化监督
MSE 与 AC 有功/无功平衡残差 MSE 之和。测试指标报告角度误差、电压误差及
PF 残差。

早停只依据验证集总损失；测试集损失仅监控，不参与模型选择。验证效果可视化：

```powershell
& "C:\Users\JieZhu\AppData\Roaming\uv\python\cpython-3.12.12-windows-x86_64-none\python.exe" Neural_network/validation/decs/visualize_decs_validation.py `
    --data Data_generation/data/e2e14_decs_N10000 `
    --checkpoint Neural_network/decs.pt `
    --output Neural_network/validation/decs/decs_validation.png `
    --n-ood 500
```

输出包含 PQ 节点电压、支路相角差和支路两端视在功率的 parity plot，以及
MAE、95% 分位绝对误差和最大绝对误差；同时生成 `decs_validation_ood.png`
和同名 JSON 指标文件。OOD 图比较 IID 验证集、
负荷 OOD、free-variable OOD 和联合 OOD。每个 OOD
运行点均严格位于 `generate_decs_dataset.py` 的相应训练采样范围之外，并由
pandapower 重新求解 AC 潮流生成标签：负荷 OOD 默认外扩训练负荷因子跨度的
10%，free-variable OOD 默认在已有 10% 训练边界带之外再外扩物理跨度的 5%。
可通过 `--ood-load-extension` 和 `--ood-free-extension` 修改外扩宽度。

## 4. 训练 generator

```powershell
& "C:\Users\JieZhu\AppData\Roaming\uv\python\cpython-3.12.12-windows-x86_64-none\python.exe" Neural_network/train_generator.py `
    --data Data_generation/data/e2e14_base_N5000_S20_T16 `
    --decs Neural_network/decs.pt
```

generator 只调用冻结的 DECS 和解析 AC 重构。训练循环中不存在 pandapower、
Newton 潮流求解或 Jacobian 线性系统。pandapower 仅用于离线生成 DECS 标签。

## 5. 小规模验证

```powershell
& "C:\Users\JieZhu\AppData\Roaming\uv\python\cpython-3.12.12-windows-x86_64-none\python.exe" Data_generation/generate_decs_dataset.py `
    --base-data Data_generation/data/e2e14_base_N5000_S20_T16 `
    --n-samples 128 --output Data_generation/data/decs_smoke

& "C:\Users\JieZhu\AppData\Roaming\uv\python\cpython-3.12.12-windows-x86_64-none\python.exe" Neural_network/train_decs.py `
    --data Data_generation/data/decs_smoke --epochs 2 --patience 2 `
    --batch-size 32 --output Neural_network/decs_smoke.pt

& "C:\Users\JieZhu\AppData\Roaming\uv\python\cpython-3.12.12-windows-x86_64-none\python.exe" Neural_network/train_generator.py `
    --decs Neural_network/decs_smoke.pt --epochs 1 --stage1-epochs 1 `
    --candidates 2 --ode-steps 1 --output Neural_network/generator_smoke.pt
```

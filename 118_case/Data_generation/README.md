# IEEE-118 DECS 与条件随机生成器

本目录实现论文中的场景型 SMPC、differentiable equality completion
surrogate（DECS）以及 conditional stochastic neural generator（CSNG）。网络
数据统一来自 `pglib_opf_case118_ieee.m`。

IEEE-118 在线网络包含 118 个节点、54 台机组和 186 条支路，其中 99 个节点
带负荷。固定 PV 潮流分区包含 53 个 PV 节点、64 个 PQ 节点和节点 69 的
平衡节点。由此得到：

```text
u   = [Pg_nonreference_active, V_PV, V_ref]            [N,72]
rho = [p_PV, V_PV_target, V_ref, p_PQ, q_PQ]           [N,235]
chi = [theta_PV, theta_PQ, V_PQ]                       [N,181]
```

## 默认 Python 环境

本文所有命令默认使用以下 UV 管理的 CPython 3.12.12 环境：

```powershell
$IEEE118_PYTHON = "C:/Users/JieZhu/AppData/Roaming/uv/python/cpython-3.12.12-windows-x86_64-none/python.exe"
```

请先在当前 PowerShell 会话中执行上述赋值，后续命令统一通过
`& $IEEE118_PYTHON` 调用该解释器。

## 1. 生成 SMPC 负荷场景

```powershell
& $IEEE118_PYTHON Data_generation/generate_smpc_dataset.py `
    --n-instances 5000 --n-scenarios 20 --horizon 16 `
    --forecast-deviation 0.15 --load-scale-min 0.8 --load-scale-max 1.1
```

默认输出为 `Data_generation/data/e2e118_N5000_S20_T16`。基础数据包含：

```text
current_load.npy       [N,99,2]
future_load.npy        [N,S,T-1,99,2]
future_pool.npy        [N,3,T-1,99,2]
normalization_parameters.npz
metadata.json
split/{train,validation,test}_indices.npy
```

P/Q 使用相同节点缩放因子以保持 PGLib 基准功率因数。论文公共算例设置为
16 个时段、20 个不确定场景、相对标称值 ±15%，数据按 8:1:1 划分。

## 2. 生成固定 PV 的 DECS 标签

```powershell
& $IEEE118_PYTHON Data_generation/generate_decs_dataset.py `
    --base-data Data_generation/data/e2e118_N5000_S20_T16 `
    --solver pandapower --n-samples 50000 --batch-size 1000 `
    --load-range-extension 0.02 --free-range-extension 0.05
```

默认输出为
`Data_generation/data/e2e118_decs_pandapower_N50000`。默认的 pandapower 后端
逐点求解固定 PV 潮流，优先用上一个收敛结果热启动，失败后自动用平启动
重试。`--batch-size` 在该模式下只控制采样和进度分块，潮流仍逐点计算。

如需使用支持原生批处理的 PGM，可改为：

```powershell
& $IEEE118_PYTHON Data_generation/generate_decs_dataset.py `
    --solver pgm --n-samples 50000 --batch-size 1000 --verify-points 100
```

PGM 输出目录默认为 `Data_generation/data/e2e118_decs_pgm_N50000`，并用
pandapower 对确定性抽样点交叉验证。两个后端保存完全相同的数组、划分和
归一化文件。Qg 越界只参与后续不等式可行性判断，不触发 PV 到 PQ 切换。
生成器默认要求所有 PQ 节点电压不低于
`0.7 p.u.`，用于剔除 Newton 法偶尔收敛到的非目标低电压数学解支路；阈值可
通过 `--minimum-pq-voltage` 调整，并写入数据元数据。

## 3. 训练 DECS

```powershell
& $IEEE118_PYTHON Neural_network/train_decs.py `
    --data Data_generation/data/e2e118_decs_pandapower_N50000 `
    --hidden-dims 256 256 --epochs 700 --batch-size 1024 `
    --learning-rate 1e-3 --lambda-physics 3 `
    --lr-scheduler plateau --lr-decay-factor 0.3 `
    --lr-decay-patience 10 --lr-min-delta 1e-5 `
    --min-learning-rate 1e-5 --patience 80
```

训练损失为标准化监督 MSE 与 AC 功率平衡残差 MSE 之和。早停只使用验证
集；测试集在最佳 checkpoint 选定后评估一次。上述参数是固定 PV 的
IEEE-118 对比实验中综合状态误差、潮流残差和约束漏检率最优的默认配置。
标准输出文件仍为 `Neural_network/decs_pgm_fixedpv.pt`，以兼容生成器、TCN
和已有验证程序。

## 4. 训练生成器与基准模型

```powershell
& $IEEE118_PYTHON Neural_network/train_generator.py `
    --data Data_generation/data/e2e118_N5000_S20_T16 `
    --decs Neural_network/decs_pgm_fixedpv.pt
```

IEEE-118 在论文默认 `K=50,S=20,T=16` 下，每个外部训练样本会展开为
16,000 个 AC 运行点，因此 CSNG 默认 micro-batch 为 1，并通过 8 步梯度
累积得到有效 batch 8。默认生成器容量为 64 个隐藏通道、16 维潜变量和
32 维潜嵌入；可行性阶段为 60 轮，经济目标 warmup 为 30 轮。正式训练采用
`lambda_fea=0.01`、`lambda_eco=0.005` 和更关注最差约束的
`alpha_c=rho_c=0.05`。多样性核使用
归一化轨迹的维度均方距离，使 `sigma_div=0.5` 不随控制维数失效。确定性
TCN、S-CSNG 和 WD-CSNG 使用各自入口，但共享同一 118 节点数据和 DECS；
确定性模型的目标成本尺度默认由最多 128 个训练样本自动冻结。

## 5. 建议的小规模验证顺序

```powershell
& $IEEE118_PYTHON -m unittest discover -s tests -p "test_*.py"

& $IEEE118_PYTHON Data_generation/generate_smpc_dataset.py `
    --n-instances 10 --n-scenarios 2 --horizon 3 `
    --output Data_generation/data/e2e118_smoke

& $IEEE118_PYTHON Data_generation/generate_decs_dataset.py `
    --base-data Data_generation/data/e2e118_smoke `
    --solver pandapower --n-samples 16 --batch-size 8 `
    --output Data_generation/data/e2e118_decs_smoke
```

正式训练前先执行小规模数据、潮流和单批训练 smoke，以根据实际 GPU 显存
调整 batch size；不要减少网络节点、支路或约束来替代显存控制。

# Generator 全代码审查与下一步实验计划

## 1. 审查范围与总体判断

审查覆盖：

- `Neural_network/train_generator.py`
- `Neural_network/noncausal_tcn.py`
- `Neural_network/generator_benchmarks.py`
- `Neural_network/decs.py` 中 Generator 实际调用的重构与不等式约束
- Generator 相关单元测试和已保存 Stage-1 checkpoint 历史

粘贴代码与工作区代码逐行内容一致，仅文件编码或换行形式不同。现有 10 个相关单元测试全部通过，因此当前主要问题不是张量维度、递归投影或训练循环写错，而是训练目标、diversity 机制和最终 `hit_rate` 目标之间存在系统性错配。

代码中值得保留的部分包括：训练集独立归一化、固定数据划分、固定验证 latent、DECS 参数冻结但保留输入梯度、free-Pg 边界及爬坡递归投影、test 集不参与 checkpoint 选择、DECS 哈希绑定、Stage-2 重置 Adam、学习率衰减和原子 checkpoint 写入。

## 2. 最高优先级问题

### 2.1 `Lfea` 优化“50 个都好”，最终指标却只要求“50 个中至少一个好”

当前候选违反量为 `V[b,k]`，可行性损失使用：

\[
L_{\mathrm{fea}}=\operatorname{mean}_b\sum_{k=1}^{K}V_{bk}.
\]

它会把 50 个候选同时推向最容易到达的同一个低违反盆地。最终 `hit_rate` 则是：

\[
\operatorname{mean}_b \mathbf 1\{\min_k V^{\max}_{bk}\le\epsilon\}.
\]

两者的候选聚合方向不同。当前目标天然鼓励候选同质化，而 diversity loss 被迫与一个强烈的同质化目标对抗。

已保存旧 Stage-1 历史提供了直接证据：

- 最大验证 `hit_rate=7.6%` 出现在 epoch 34。
- 最小验证 total loss 出现在 epoch 56，但 `hit_rate` 已下降到 `5.6%`。
- epoch 56 的 `candidate_feasible=5.6%` 与 `hit_rate=5.6%` 相同。按 `K=50` 换算，命中的实例中平均约有 50/50 个候选同时可行，说明这些候选基本相同，而非 50 次有效搜索。

### 2.2 feasibility-aware diversity 在最需要工作时被关闭

当前实现：

\[
s_k=\exp(-V_k/\tau_f),\qquad
w_{ij}=s_i s_j,
\]

然后使用 `w_ij` 加权高斯相似度。纯 diversity 日志中 `V≈1140`、`tau_f=2`，所以 `exp(-570)` 在 float32 下溢为零，导致 `Ldiv=0` 且梯度为零。

即使进入较低违反区域，若每个实例只有一个候选接近可行，其余候选的 score 接近零，也没有足够的有效候选对产生 diversity 梯度。该机制要求“至少两个近可行候选”才能开始帮助，而训练最缺少的恰恰是第二个近可行候选。

### 2.3 高斯 diversity 在候选重合处也有零梯度

相似度核为：

\[
K_{ij}=\exp\left(-d_{ij}^2/(2\sigma_d^2)\right).
\]

当候选完全相同时，`d_ij=0`，此时损失最大但对输出的梯度仍为零。增加 `lambda_div` 只能放大已有梯度，不能恢复零梯度。

对旧 Stage-1 Generator checkpoint 的直接输出测量表明：

- 候选归一化 RMS 距离：min=0、p05=0、median=0、p95=`3.99e-4`、max=`8.30e-4`。
- 归一化坐标的候选标准差均值仅 `7.12e-5`。
- raw trajectory 的候选 RMS 距离中位数仅 `1.63e-4`。
- latent embedding 自身仍有变化，但通过后续输入层和 TCN 后几乎消失，说明模型学会了忽略 latent。

因此，历史的 `sigma=0.5` 与实际候选距离相差约三阶数量级。当前固定 `sigma=0.1` 仍没有基于实际距离校准；但不能只把 sigma 继续调小，因为大量候选已经精确重合，重合点本身仍是零梯度。

### 2.4 递归 Pg 投影的初始化会产生近似恒定轨迹

输出层 bias 为 0.5。对 free Pg：

- `t=1` 输出静态上下界中点；
- 后续时段输出动态可达区间的中点；
- 当上下爬坡对称且未碰静态边界时，该中点就是上一时段 Pg。

所以网络初始轨迹接近恒定出力。负荷随时间变化时，free generators 不跟踪变化，scenario-dependent reference generator 被迫吸收大部分功率变化，从而容易形成 reference ramp 违反。小输出权重与较弱 latent 通道进一步加强了该偏置。

### 2.5 Stage-1 checkpoint 选择与 Stage-2 门槛不合理

Stage 1 只按 validation total loss 选模型，因此会选择 hit-rate 更低但候选平均违反量稍低的 epoch。旧历史中 epoch 34 的 hit-rate 为 7.6%，而最终选中的 epoch 56 只有 5.6%。

Stage 2 只有在全部验证实例 `hit_rate=100%` 时才保存任何经济最优 checkpoint。在当前 Stage-1 hit-rate 远低于 100% 的情况下：

- 训练过程中不会留下“hit-rate 有改善但尚未满命中”的最佳模型；
- patience 到期后可能因没有 full-hit checkpoint 而直接报错；
- checkpoint 逻辑无法反映从 5% 到 50% 或 90% 的实质进步。

这不需要放松任何约束，只需要让模型选择指标与分阶段目标一致。

## 3. 第二优先级问题

### 3.1 当前 mean-CVaR 没有直接对齐“任一约束违反即失败”

代码先在每个约束 family 的分量轴上做 mean-CVaR，再对 scenario 和 time 求和。对于 Qg、thermal 等大 family，单个严重违反会被 top-5% 的平均稀释；对 scenario/time 求和也没有显式强调最坏场景和最坏时刻。

但最终 feasibility 使用所有 scenario、time、constraint 的最大值。训练 surrogate 与判定指标仍存在差异。建议使用“均值 + 平滑最大值”或两层 CVaR，使 family 内以及 scenario/time 两层都关注最坏项。

### 3.2 min/mean/max pooling 丢失场景联合结构

Generator 输入仅包含每个时刻、每个负荷位置的 min/mean/max。它无法知道：

- 各母线极值是否来自同一个场景；
- 场景间协方差和空间相关性；
- 跨时间的同一场景轨迹相关性；
- 分布尾部除最大值外的形状。

而热稳定、Qg 和 reference ramp 都取决于这些联合结构。损失问题解决后，如果 hit-rate 仍明显受限，下一步应将 20 个场景作为集合输入，用 DeepSets 或轻量 set-attention 编码，再与 TCN 时序特征融合。

### 3.3 DECS 会出现 Generator-induced distribution shift

DECS 在随机样本测试集上的监督误差较低，并不保证 Generator 梯度搜索到的轨迹仍处于同一分布。Generator 会主动寻找 surrogate 低违反区域，可能利用 DECS 的局部误差。

旧 Stage-1 历史中，validation feasibility loss 从 epoch 1 到 60 明显下降，但 DECS PF residual 从约 `0.426` 上升到 `0.454`。PF residual 本身虽不是目标约束，但它上升说明 reconstructed state 与真实 AC 方程的一致性没有同步改善，进而可能影响 Pg、Qg、热流和 ramp 违反量的准确性。

不建议通过人为增加约束裕度解决。更可靠的方法是：周期性将 Generator 轨迹送入 pandapower，获得精确状态与违反量，将这些 on-policy 样本加入 DECS 数据集并重新训练，即 active-learning / dataset aggregation。

### 3.4 损失尺度依赖 K、S、T，参数实验不总能迁移

`Lfea` 对 candidate、scenario 和 time 求和，而 `Ldiv` 是候选对加权平均，经济损失也是均值。因此：

- K、S、T 改变会改变 loss 比例；
- 用较小 K/S/T 做出的 `lambda` 结论不能直接迁移到 `K=50,S=20,T=16`；
- 全局梯度裁剪若频繁触发，还会使绝对权重进一步失去直观意义。

最终实验应始终保持 `K=50,S=20,T=16`，仅减少实例数和 epoch。或者把 `Lfea` 改成相应轴上的均值，使尺度与 K/S/T 无关。

## 4. checkpoint 一致性提醒

当前保存的 `generator_tcn_clip_stage1.pt` 绑定的 DECS SHA256 为：

`2cdbe828a66ce463bea70e1263915f84e3c571dbce5e1a822fcd078d76e7b96b`

当前默认 `decs_pgm_fixedpv.pt` 的 SHA256 为：

`a4243534f7ea700ce8128cec064c97ab96b4a328176dc84a1e1cd8a263e9a0dd`

两者不一致。该旧 Generator checkpoint 可用于观察候选塌缩，但不应作为当前正式训练的 resume 起点。现有代码的哈希检查会正确拒绝它。

## 5. 下一步实验：按成本和信息量排序

### 实验 A：一个 batch 的梯度与候选诊断（最高优先级）

固定一个 batch 和同一组 K=50 latent，记录：

1. 每个实例 `V_k` 的 min/median/max；
2. score 的 mean/max、非零比例和有效候选数；
3. 候选归一化距离的 min/p05/median/p95/max；
4. `||grad Lfea||`、`||grad Ldiv||` 及夹角；
5. 加权梯度比 `||lambda_div grad Ldiv|| / ||lambda_fea grad Lfea||`；
6. raw output 和 projected output 的候选标准差；
7. clip 边界比例及梯度裁剪触发率。

只有在 `Ldiv` 梯度非零后，才调整 `lambda_div`。sigma 应依据实际候选距离选择，使主要候选对处于高斯核的有效斜率区域，而不是固定猜测 0.1 或 0.01。

### 实验 B：确认可行性上限

从 30--50 个代表性 SMPC 实例中，用 pandapower 加数值优化或其他精确方法搜索共享于 20 个场景、16 个时段的可行轨迹。记录精确 hit-rate 和最难约束。

若一部分实例在当前 25% ramp 定义下本身没有鲁棒可行轨迹，Generator 不可能达到 100% hit-rate，应该先区分数据不可行性与学习失败。

### 实验 C：保持现有代码的基准消融

保持 `K=50,S=20,T=16`，仅用 30--100 个训练实例、固定 10--20 epoch：

1. `lambda_div=0, lambda_eco=0`：得到 feasibility-only 基线；
2. 当前 absolute-score Gaussian diversity；
3. 对三组实验使用完全相同的数据划分、初始化和验证 latent。

若当前 diversity 的 candidate distance 与 hit-rate均不超过 feasibility-only，则无需继续扫描 `lambda_div`。

### 实验 D：修改候选聚合目标

令 `V_bk` 为候选违反量，比较：

1. 当前 `sum_k V_bk`；
2. best-of-K 平滑损失
   \[
   L_{hit}=-\tau_k\log\left(\frac1K\sum_k\exp(-V_{bk}/\tau_k)\right);
   \]
3. 混合目标
   \[
   L_{fea}^{new}=L_{hit}+\eta\,\operatorname{mean}_k V_{bk},
   \]
   其中先测试 `eta=0.05,0.1,0.2`。

`Lhit` 负责让至少一个候选进入可行域，小权重 mean 项防止其余候选完全失控。该目标比单纯增大 diversity 权重更直接地对齐 hit-rate。

### 实验 E：修复 diversity 的候选选择与核宽度

建议顺序：

1. 用 detached `V_k` 选违反量最低的 top-M 候选，先测 `M=5,10`；
2. 在 top-M 内均匀计算 diversity，避免绝对 score 下溢；
3. 每个 batch 用 detached 候选距离中位数设置自适应 sigma；
4. 记录真实距离和 diversity gradient，而非只看 `Ldiv` 数值。

这没有改变约束边界，也没有增加裕度，只改变候选多样性的数值实现。

### 实验 F：修复 latent 被忽略

若实验 E 后 raw candidate distance 仍快速归零，建议比较：

1. 当前仅在输入端拼接 latent；
2. 在每个 TCN residual block 使用 latent FiLM；
3. 添加从 latent 到 raw trajectory 的小型直接 residual 分支。

`latent_dim=16` 不是首要问题。当前证据说明 latent 信息进入了 embedding，但被后续网络忽略；单纯把 latent_dim 调到 32 或 64 通常不会解决。

### 实验 G：reference-ramp 结构先验

将条件中的系统平均负荷变化通过固定或可学习 participation factors 映射成 nominal free-Pg trajectory，TCN 只生成 residual，再使用原递归边界/爬坡投影。这样初始 free generators 就能跟踪负荷变化，避免所有变化都落到 reference generator。

该方法不增加约束裕度，只改善输出参数化和初始化。

### 实验 H：pandapower on-policy 校准

每隔若干 epoch：

1. 从训练/验证实例生成候选；
2. 使用 pandapower 计算精确状态和六类不等式违反量；
3. 比较 DECS 与 pandapower 的违反量误差、误判率和 magnitude error；
4. 将误差最大的 Generator 样本加入 DECS 数据集并重训；
5. 最终 checkpoint 使用 pandapower hit-rate 做只读验证与选择。

这比人为约束裕度更符合当前研究目标。

## 6. 推荐的算法路线

建议不要继续在当前公式上只扫描 `lambda_div`。优先算法组合为：

\[
L_{stage1}=L_{hit}+\eta L_{cover}+\lambda_{div}L_{div}^{topM},
\]

其中：

- `Lhit` 对齐 50 个候选中至少一个可行；
- `Lcover` 以较小权重维持整体候选质量；
- `Ldiv_topM` 始终在当前最接近可行的若干候选上工作；
- sigma 按候选真实距离自适应；
- Stage-1 checkpoint 按 `(hit_rate, -max_violation, -loss)` 字典序选择；
- Stage 2 在 hit-rate 稳定后加入经济损失，未达到 full-hit 时仍保存 hit-rate 最优模型，达到 full-hit 后再按经济成本选择；
- Generator 训练期间用 pandapower 做 on-policy 校准，不添加任何约束裕度。

在上述损失修正完成后，再评估 DeepSets 场景编码和 latent FiLM。否则先增大网络宽度、latent_dim 或训练 epoch，大概率只会更充分地收敛到当前塌缩解。


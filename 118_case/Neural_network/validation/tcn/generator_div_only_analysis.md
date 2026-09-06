# Generator 纯 diversity 对照实验分析

## 结论

这次实验不能说明 `lambda_div` 仍然偏小。日志表明 diversity 项在初始不可行轨迹上被软可行性权重完全关断，训练目标从第 1 轮开始就是常数零，因此模型没有发生有效更新。

## 日志证据

- 第 1--5 轮训练与验证均为 `total=0`、`Ldiv=0`。
- 验证集的 cost、约束违反量和 `DECS_PF` 在五轮内保持到打印精度完全一致。
- 验证平均候选违反量约为 `1.140e3`，最大违反量均值约为 `8.781`。
- `candidate_feasible=0`、`hit_rate=0`，说明初始轨迹远未进入可行域。

## 数值原因

代码中的 soft feasibility score 为

\[
s_k=\exp(-v_k/\tau_f),
\]

而 diversity loss 的候选对权重为

\[
w_{ij}=s_i s_j.
\]

本次默认 `tau_f=2`，平均候选违反量约为 `v=1140`，因此

\[
s\approx\exp(-1140/2)=\exp(-570),
\]

该值在 float32 中下溢为精确的零。所有候选对权重随之为零，diversity loss 的分子为零，分母只剩 `epsilon`，最终 `Ldiv=0`，梯度也为零。

对应实现位于 `diversity_loss()`：候选相似度核乘以 `(score_i * score_j).detach()`。`detach()` 不是本次零梯度的唯一原因；即使不 detach，当 score 已数值下溢为零时也无法产生有效梯度。

## 与上一轮实验的关系

当前 diversity loss 存在两个不同的梯度死区：

1. 初始轨迹严重不可行时，soft feasibility score 下溢为零，`Ldiv=0` 且无梯度。
2. 候选已经塌缩为相同轨迹时，高斯核接近 1，但距离导数与候选差值成正比；候选完全重合处的排斥梯度同样为零。

因此，单纯增加 `lambda_div` 不能解决问题：权重只能放大已有梯度，不能恢复已经为零的梯度。

## 约束违反结构

本次初始轨迹的 mean-CVaR 以 Qg 为主：

- Qg：约 `2.87`
- Pg：约 `0.355`
- thermal：约 `0.294`
- ramp：约 `0.0428`
- angle：约 `0.00233`
- V：约 `2.84e-5`

这与先前已训练模型中爬坡违反占主导的状态不同，进一步证明本次日志主要反映未经训练的初始网络，而不是一个经过纯 diversity 优化后的网络。

## 建议

1. 结束当前纯 `Ldiv` 运行；继续到 500 轮不会改变结果。
2. 恢复非零 `lambda_fea`。当前 feasibility-aware diversity 的定义决定了它不能单独把随机初始轨迹带入可行域。
3. 若不修改损失逻辑，先进行 feasibility-only 预训练，再从其 checkpoint 测试 diversity，才是有效的纯 diversity 消融实验。
4. 不建议仅靠增大固定 `tau_f` 解决：使 `v≈1140` 的初始样本获得明显权重需要非常大的温度，但当 `v≈1--3` 后又会失去区分可行性的作用。
5. 后续至少记录 `score` 的 mean/max、非零比例、候选对归一化 RMS 距离和 diversity 梯度范数，以区分权重、核宽度及数值下溢问题。


# Improved Generator Stage-1 小规模实验

## 目的

验证两个已定位问题：

1. 原 feasibility-aware diversity 在初始严重不可行阶段是否被数值关断；
2. best-of-K feasibility、top-M adaptive diversity 和 worst-violation 项能否改善候选覆盖与严格约束代理指标。

原始 `train_generator.py` 未修改。实验使用独立程序 `Neural_network/train_generator_improved.py`。

## 实验设置

- 数据：`e2e118_generator_tune_newrange_N30_S20_T16`
- train/validation/test：24/3/3
- K=50、S=20、T=16
- hidden=64、latent=16、latent embedding=32
- learning rate=`5e-4`
- 相同 seed=`2026`
- 每组 30 个 Stage-1 epoch
- baseline：原候选求和 feasibility，绝对 score 加权 diversity，`lambda_fea=0.01`、`lambda_div=1`、`sigma=0.1`
- improved：top-5 feasibility + 0.1 population coverage、top-10 adaptive diversity、`lambda_fea=1`、`lambda_div=0.1`
- improved-worst：在 improved 候选排序违反量中增加 `10 * exact maximum normalized violation`
- 未使用任何约束裕度

## 代理模型训练结果

第 30 轮对比：

| 指标 | baseline | improved（无 worst 项） | improved-worst |
|---|---:|---:|---:|
| validation hit-rate | 0 | 0 | 0 |
| candidate distance | 0.00790 | 0.01307 | 0.01065 |
| candidate mean violation | 9.481 | 7.093 | 9.563 |
| mean candidate max violation | 0.8118 | 0.7063 | 0.7436 |
| diversity loss | 0.9963 | 0.5908 | 0.5997 |
| Qg mean-CVaR | 0.0146 | 0.00787 | 0.0154 |
| thermal mean-CVaR | 2.17e-4 | 2.11e-4 | 6.84e-5 |
| ramp mean-CVaR | 0.0158 | 0.0150 | 0.0154 |

30 轮中的最佳代理结果：

- baseline 最低候选平均违反量约 6.33，最佳 mean candidate max violation 约 0.775。
- improved 最低候选平均违反量约 6.03，最佳 mean candidate max violation 约 0.700。
- improved-worst 最佳 mean candidate max violation 约 0.680。

因此 diversity 修复是有效的：原始 loss 从早期近零最终上升并饱和到约 1，候选距离持续缩小；改进 loss 从 epoch 1 起稳定在约 0.58--0.61，候选距离得到维持或扩大。best-of-K 目标也降低了代理违反量，但 30 轮内尚未产生严格可行候选。

## pandapower 独立检查

对 baseline 与 improved-worst checkpoint 使用同一验证实例、同样 10 个候选，逐点求解 3010 个独立运行点：

- 两者均 3010/3010 收敛；
- 全部为高电压解；
- 因此当前失败不是潮流不收敛或低电压分支造成的。

运行点约束违反率：

| 约束 | baseline 精确违反率 | improved-worst 精确违反率 | baseline DECS 违反率 | improved-worst DECS 违反率 |
|---|---:|---:|---:|---:|
| Pg | 0% | 0% | 0% | 0% |
| Qg | 40.13% | 59.20% | 28.07% | 49.50% |
| voltage | 0% | 0% | 0% | 0% |
| angle | 0% | 0% | 0% | 0% |
| thermal | 27.14% | 14.62% | 27.31% | 11.30% |
| ramp | 10.43% | 9.73% | 8.60% | 8.20% |

DECS false-feasible 运行点比例：

- Qg：baseline 12.59%，improved-worst 11.26%
- thermal：baseline 2.69%，improved-worst 3.32%
- ramp：baseline 1.83%，improved-worst 1.53%

改进算法将搜索重点从 thermal/ramp 部分转移，但 exact Qg 违反变多。说明 Generator 已开始利用 surrogate 局部误差；继续调整 Generator loss 权重无法单独解决精确约束问题。

## 结论与下一步

1. top-M adaptive diversity 解决了初期 `Ldiv=0` 的数值死锁，并明显减缓候选塌缩。
2. best-of-K 加 coverage 比原候选求和更符合 hit-rate，代理指标有所改善。
3. 当前仍未得到严格可行候选，因此该程序是有效研究原型，不是正式算法终版。
4. 下一优先级不是继续增大 `lambda_div`，而是收集 Generator on-policy 轨迹的 pandapower 标签，补充训练 DECS，重点覆盖 Qg、thermal 和 reference-ramp 边界区域。
5. Generator checkpoint 应先按 validation hit-rate、best-of-K maximum violation 选择，再在 exact pandapower 子集上校准。
6. active-learning 后再比较 worst weight 0、3、10；当前单实例精确结果不支持直接把 10 固定为最终值。


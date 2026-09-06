# Deterministic TCN pure-penalty experiment

## Protocol

- Base data: `e2e118_N5000_S20_T16` (4000/500/500 split).
- Equality completion: current `decs_pgm_fixedpv.pt`.
- Model: non-causal deterministic TCN, 64 hidden channels.
- Loss: economic weight 0; no feasibility margin was added.
- Common settings: batch size 8, Adam learning rate `5e-4`,
  `alpha=0.05`, CVaR tail fraction `0.05`, ramp limit `0.25 Pmax`.
- Independent validation: pointwise pandapower NR, strict normalized
  inequality tolerance `1e-4`.

## Runs

| run | epochs | extra trajectory-max weight | best epoch | best validation loss | maximum proxy feasible rate |
|---|---:|---:|---:|---:|---:|
| baseline | 10 | 0 | 9 | 3.511696 | 1.6% |
| max-enhanced | 8 | 10 | 8 | 9.384872 (composite) | 2.4% |

On the same first 10 validation instances, every one of the 3200 pandapower
operating points converged. Neither checkpoint produced a strictly feasible
trajectory. The baseline and max-enhanced mean maximum normalized violations
were 0.5802 and 0.5586, respectively. The max-enhanced run reduced the worst
value from 0.8045 to 0.6983, but traded this against larger Pg error.

For the 10 validation instances with the smallest scenario load-ramp spans,
the max-enhanced checkpoint obtained a mean maximum violation of 0.05312 and a
best-instance value of 0.003873. Pg was feasible for all 10; remaining errors
were mainly thermal (mean 0.01037) and ramp (mean 0.04850). Strict feasibility
was still 0/10.

## Structural ramp diagnostic

The reference generator ramp is 295.5 MW per period. A shared deterministic
non-reference schedule can shift all scenarios by one common increment, but it
cannot remove the cross-scenario spread in net-load increments. Ignoring the
smaller AC-loss correction, 456/500 validation instances (91.2%) have at least
one time transition whose 20-scenario active-load increment span exceeds
`2 * 295.5 = 591 MW`. Their mean, p95, and maximum spans are 823.64, 1044.19,
and 1180.20 MW. This is strong evidence that the current shared-trajectory plus
reference-ramp formulation is infeasible for most validation instances,
independently of the penalty coefficient.

## Conclusion

Increasing the scalar feasibility coefficient alone cannot resolve the issue.
The optional trajectory-maximum penalty improves worst violations modestly but
does not overcome the scenario/ramp incompatibility. Before a long formal
training run, either the scenario construction or the intended treatment of
the scenario-dependent reference-generator ramp must be clarified. No margins
or relaxed feasibility thresholds were used in this experiment.

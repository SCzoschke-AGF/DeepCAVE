Performance ↔ Computational Cost
================================

How much performance do we gain for additional compute?

This analysis shows each successful trial's objective value against its duration in seconds.
It uses the run or group and objective selected in the sidebar, including the objective's
optimization direction. Repeated evaluations remain separate points.

The compact controls offer **Trial duration** as the cost metric, **All trials** (or an individual
multi-fidelity budget when available), and a **Pareto front** checkbox enabled by default.
The frontier is computed only from the displayed trial subset. It minimizes duration and respects
whether the selected objective is minimized or maximized. Equal duration/objective pairs remain
valid and can both be Pareto-efficient.

Failed and pruned trials, non-finite objective values, missing timings, and negative durations are
excluded. DeepCAVE reuses its conventional Time objective where available, including the precise
duration computed by the Optuna converter; otherwise it subtracts trial start time from end time.
Zero-duration trials remain visible on a linear axis. The x-axis uses logarithmic scaling only when
all durations are positive and span at least two orders of magnitude.

Hover shows the trial ID, objective value, duration, and native configuration details. Clicking a
point opens its configuration using DeepCAVE's existing configuration interaction.

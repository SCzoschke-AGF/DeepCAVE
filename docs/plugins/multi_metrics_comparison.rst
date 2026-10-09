Multi-Metrics Comparison
========================

Multi-Metrics Comparison evaluates two metrics for the same trials in one Optuna study. It can
compare their raw values or their study-wide rankings and highlights how strongly their Top-N
trial sets overlap.

Only successful trials with finite values for both selected metrics are included. Native Optuna
objectives and numeric user attributes discovered during import are available. Optimization
directions can be selected independently for both axes.

Controls
--------

* **Metric 1 / Metric 2:** Select the two metrics and their maximize or minimize direction.
* **Trials:** Analyze all valid trials or a Top-N selection.
* **Selection:** For Top N, select by Metric 1, by Metric 2, or compare the union of both
  Top-N sets.
* **Scale:** Show metric values or global study ranks, where rank 1 is best.
* **Display:** Show only the analysis set or add all other valid trials as context.
* **Show Pareto front:** Overlay the non-dominated trials from the analysis set.

For a Top-N comparison, the sidebar reports both set sizes, their intersection, differences,
Jaccard similarity, and overlap rate. Pearson and Spearman association and the number of
Pareto-optimal trials are always calculated for the current analysis set.

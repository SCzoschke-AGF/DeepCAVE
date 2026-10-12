Cost & Efficiency
=================

Connect the sidebar's selected objective to **trial duration**, using valid completed trials.
Select a budget when comparing trials at the same fidelity. Press **Process** to calculate
fANOVA main effects for both duration and objective performance on the same trial samples.
View controls redraw cached results without retraining the importance models.

Parameter-level analysis
------------------------

* **Cost Importance** ranks hyperparameters by fANOVA main effects for trial duration.
* **Cost vs. Objective Importance** plots cost importance on the x-axis and objective importance
  on the y-axis.
* **Parameter Importances** lists all parameters, sorted by descending cost importance.
  Highlighted rows identify candidates strictly above median cost importance and strictly below
  median objective importance. Unavailable importance estimates are left blank.
  Switch this panel to **Parameter vs. Objective** to compare the selected parameter's
  objective and duration profiles. Both plots share observations, trend bins and elite markers.
* **Parameter vs. Trial Duration** shows numerical observations or categorical distributions.
  Select a parameter with the control, an importance plot, or the candidate table.
  **Duration trend** is enabled by default for binned medians with a 25th–75th percentile band on numerical
  parameters. Bins keep equal values together; no values beyond observed trials are predicted.
  **Mark elite trials** highlights the currently selected elite fraction with outlined diamonds.

These are observed associations and model-based main effects, not causal evidence.
Candidates may still matter through interactions or within specific search-space regions.
Constant targets and insufficient data produce unavailable-importance states.

Configuration-level analysis
----------------------------

**Performance vs. Computational Cost** shows trials and an optional Pareto front (enabled by
default). Duration is minimized and objective direction follows the sidebar. Click a trial to
use DeepCAVE's configuration drilldown. No preferred trade-off is selected automatically.

**Near-Optimal Cost Efficiency** shows only the selected top 1%, 5%, 10%, or 20% (default 5%).
It uses the existing rank-based selection: ceil(fraction * trial count), at least one trial,
ordered by objective and then stable source/history order for ties. Repeated evaluations remain
separate trials. The highlighted star is the cheapest trial in this elite set. Summary statistics
show its size and minimum, median, maximum duration and max/min ratio. The ratio is undefined
when the minimum duration is zero.

Duration uses the existing converter timing abstraction (precise elapsed Time when recorded,
otherwise end minus start). Failed, pruned, missing/invalid timing and nonfinite objective
observations are excluded. Duration axes use log scaling only when all values are positive and
span at least two orders of magnitude.

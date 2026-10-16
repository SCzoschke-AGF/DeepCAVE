Run Overview
============

Summarizes the selected run and objective with Performance vs. Computational Cost
and fANOVA Feature Importance.

The main plot shows the objective of each successful trial against its duration in seconds.
A narrow horizontal histogram on the right shows the distribution of these same trials,
sharing the objective axis and zoom range with the scatterplot. Histogram counts appear
on hover. A dashed horizontal line marks the best objective across both views.
In None and Pareto front modes, this line starts at the fastest trial achieving the best objective
and extends right through the histogram.
The duration axis uses a logarithmic scale for strictly positive durations spanning at least
two orders of magnitude. The highlight selector offers None, Pareto front (the default),
and Elite trials. The adjacent Elite trial (Top N %) control sets the percentage of trials
with the best objective
values to highlight, respecting the objective's optimization direction. All trials remain
visible, and the histogram and importance estimates always use the full selected trial set.
The deactivate tool tips switch beside Elite trial (Top N %) disables hover tooltips in both
the main plot (including its histogram) and the importance plot. Tooltips are enabled by default.
In Elite trials mode, an additional dashed line marks the objective cutoff of the highlighted trials
across the scatterplot and histogram, with its value labeled on the right. If the cutoff equals
the best value, the shared line and label appear only once. Otherwise, the absolute difference
is shown as Δ on the right, vertically centered between the two lines.

The segmented Importance control below the importance plot switches between
Objective Importance bars (the default),
Cost Importance bars, and a scatterplot with Cost Importance on the x-axis and
Objective Importance on the y-axis. Switching views reuses the calculated estimates.
Feature importance uses the same objective and duration main-effect calculations and valid trials
as the Computational Cost analysis, restricted to the overview's selected budget.
Both targets are normalized before forest fitting: the objective uses its bounds and
optimization direction, matching the Importances analysis; duration uses the selected trial
range and is minimized. The plotted trial values remain in their original units.
The native importance figures are reused with their original labels and styling.
Unavailable importance estimates are explained in the plot.

2-D Analyses
============

The 2-D Analyses plugin examines the empirical relationship between two hyperparameters and a
selected objective in an Optuna study. Select a study, objective, two different hyperparameters,
and the proportion of plottable trials that should be considered elite.

The first plot shows every successful trial without binning. Point colors encode the global,
direction-aware objective rank quantile, where one is the best trial. The adjacent heatmap groups
both hyperparameters into joint bins and displays the percentage and number of elite trials in
each observed cell.

Empirical Objective Surface
---------------------------

The full-width plot shows the median objective in each joint bin. For two numeric parameters it is
rendered as a contour plot. Missing joint bins remain empty and no surrogate model or value
imputation is used. If either parameter is categorical or ordinal, an objective heatmap is shown
instead because interpolation between categories would not be meaningful.

Numeric axes use four equally wide bins, using logarithmic spacing for logarithmic
hyperparameters. Numeric parameters with no more than four observed values, categorical
parameters, and ordinal parameters are displayed discretely. Elite thresholds and rankings honor
the selected objective's optimization direction, and ties at the threshold are included.

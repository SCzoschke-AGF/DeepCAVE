2-D Analyses
============

The 2-D Analyses plugin examines the empirical relationship between two hyperparameters and a
selected objective in an Optuna study. Select a study, objective, two different hyperparameters,
and the proportion of plottable trials that should be considered elite.

The top-left plot shows every successful trial without binning. Point colors encode the selected
objective value, and the color direction follows the optimization direction. The top-right
heatmap groups both hyperparameters into joint bins and displays elite enrichment relative to the
selected global
elite proportion. One is the expected baseline, values below one are underrepresented, and values
above one are overrepresented. Cell labels retain the number of trials, while hover details also
show the raw elite rate and elite count. Unobserved cells remain neutral and distinct from observed
cells with zero enrichment. Favorable objective values and high enrichment use the bright end of
the shared Viridis color scale.

Empirical Objective Surface
---------------------------

The bottom-left plot shows the median objective in each joint bin. For two numeric parameters the
median is rendered as a contour plot. Missing joint bins remain empty and no surrogate model or
value imputation is used.
If either parameter is categorical or ordinal, an objective heatmap is shown instead because
interpolation between categories would not be meaningful. The Viridis direction follows the
objective direction, so the bright end consistently represents better values.

The bottom-right controls set the bin count independently for both selected features. Both the
elite-enrichment and objective-contour plots use this shared two-dimensional grid. Categorical
features continue to use their categories and therefore ignore the numeric bin count.

Numeric axes use four equally wide bins by default, using logarithmic spacing for logarithmic
hyperparameters. Numeric parameters with no more than four observed values, categorical
parameters, and ordinal parameters are displayed discretely. Elite thresholds and rankings honor
the selected objective's optimization direction, and ties at the threshold are included.

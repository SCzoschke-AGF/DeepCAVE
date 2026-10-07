Near-Optimal Configurations
===========================

The **Near-Optimal Configurations** plugin analyzes the observed Rashomon set of an Optuna
study: trials with similar objective performance that may use very different hyperparameter
configurations.

Choose the global objective and define the near-optimal set as a top percentage, an absolute
distance from the best value, or a relative distance from the best value. The plugin compares
pairwise normalized configuration distances, robust parameter spread, global fANOVA importance,
and hierarchically clustered configuration regimes.

Distances use ConfigSpace's normalized representation. Numerical and logarithmic parameters are
compared in their normalized search-space coordinates, categorical changes contribute one,
constants contribute zero, and active/inactive mismatches contribute one. The reported aggregate
is a root-mean-square distance in the interval from zero to one.

The random baseline repeatedly draws equally sized subsets from the completed observed trials.
It is a reference for the optimizer's explored region, not an estimate of every configuration that
could exist in the declared search space. This distinction is especially important for adaptive
samplers such as TPE.

Large heatmaps are limited to 100 representative trials. Diversity summaries, clustering, and the
trial table still use the complete selected set. Dendrogram branches are rendered in two combined
line traces, and cluster-axis labels are added together to keep rendering responsive as the number
of trials and parameters grows. Pairwise distance calculations scale quadratically with the number
of completed trials.

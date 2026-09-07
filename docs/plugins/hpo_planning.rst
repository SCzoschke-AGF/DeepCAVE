HPO Planning
============

The HPO Planning plugin compares the observed value ranges of every individual hyperparameter
across three configurable elite groups in an Optuna study. Select an objective and three distinct
elite proportions. The proportions are displayed from the broadest to the strictest group.

For each feature and group, the horizontal line shows the minimum and maximum, black ticks show
Q10 and Q90, and the orange point shows the median. Positions are normalized to the feature's
observed search range. Logarithmic hyperparameters are transformed before normalization, while a
constant observed feature is placed at the center.

Elite Groups and Conditional Parameters
---------------------------------------

Elite thresholds are calculated once across all successful trials with a finite selected
objective. For maximization objectives, trials at or above the corresponding upper quantile are
included; for minimization objectives, trials at or below the lower quantile are included. Ties at
the threshold are retained. For a conditional hyperparameter, each row then uses only elite trials
in which that parameter was active.

Categorical hyperparameters follow their ConfigSpace choice order and ordinal hyperparameters
follow their declared sequence. Their quantiles always refer to an observed category rather than
an interpolated value. Since this ordering is an encoding rather than a measured distance, the
original category values are included in the tooltips.

The plugin analyzes only individual ConfigSpace hyperparameters. It does not create or display
derived features, parameter products, or interactions.

HPO Planning
============

The HPO Planning plugin is an interactive search-space planner for the next optimization run.
Select an objective and three distinct elite proportions in the compact toolbar. Every parameter
then occupies one row, and the complete planned space is collected in the sidebar.

For numerical parameters, zero and one always represent the lower and upper bounds of the current
ConfigSpace definition. Each elite group is shown on a compact separate subtrack with a Q10–Q90
band and a median marker. A thinner, lighter line behind it shows the complete observed minimum to
maximum range of that elite group. The selected range can be changed with the range handles or by
entering original parameter values; both controls use one shared planning state. Integer and step
metadata are respected.

Linear parameters are normalized in their linear sampling space and logarithmic parameters in
logarithmic space. Consequently, a selection below zero or above one represents an extension
beyond the current search space. Such extensions are shown with striped accents and informational
warnings but are not blocked.

Elite Groups and Conditional Parameters
---------------------------------------

Elite thresholds are calculated once across all successful trials with a finite selected
objective. For maximization objectives, trials at or above the corresponding upper quantile are
included; for minimization objectives, trials at or below the lower quantile are included. Ties at
the threshold are retained. For a conditional hyperparameter, each row then uses only elite trials
in which that parameter was active.

Categorical hyperparameters follow their ConfigSpace choice order and ordinal hyperparameters
follow their declared sequence. They are planned with a compact multi-select rather than a
numerical range.

The sidebar reports wider, narrower and unchanged ranges, provides a live Python-dictionary
preview, and offers copy and reset actions. The planner compares only the current and planned
search spaces. It deliberately does not infer model- or library-specific validity limits.

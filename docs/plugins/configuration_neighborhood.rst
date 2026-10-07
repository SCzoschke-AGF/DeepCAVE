Configuration Neighborhood
==========================

The Configuration Neighborhood plugin compares every successful trial in an Optuna study with a
selected reference trial. Select a trial directly in the objective history, from the sortable and
filterable trial table, or with one of the quick-selection actions. The objective is controlled by
the global selector in the DeepCAVE sidebar.

The similarity view plots normalized configuration distance against the selected objective and
lists the reference followed by the ``k`` nearest other trials. The parameter-differences panel
shows which hyperparameters account for those local differences. After a selection, the history
and trial controls remain visible above the comparison. **Select new trial** hides the lower
comparison again while preserving the history's Plotly interaction state.

Distance definition
-------------------

Numerical parameters use ConfigSpace's normalized vector representation, including logarithmic
scaling. Categorical parameters contribute zero when equal and one when different, ordinal
parameters use their normalized sequence position, and constants contribute zero. Two inactive
conditional parameters contribute zero; an active/inactive mismatch contributes one. The final
distance is the root mean square of these per-parameter contributions. It is therefore bounded by
zero and one regardless of the number of non-constant hyperparameters.

In this first version, ``k`` counts other trials, so the nearest-configurations table contains up
to ``k + 1`` rows including the reference. Normalized Euclidean is the only available metric.

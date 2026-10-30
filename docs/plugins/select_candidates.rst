Select Candidates
=================

Collect concrete Optuna trials for a subsequent drill-down analysis. The sidebar selects
the primary study and central objective; **Additional studies** adds other loaded studies.
The local **Table objectives** selector controls only the table's objective columns.

The Trial Explorer supports column sorting and filtering. Press Enter to apply a column
filter. The scatterplot shows exactly that filtered set, excluding trials without a finite
central objective or valid duration. Completed trials are shown by default; other trial
states can be included in the table through **Trial states**.

Click a row, checkbox or scatter point to toggle a candidate. Box and lasso gestures add
all enclosed trials without clearing earlier selections. Selected Trials persists in the
browser session across filters, objective changes, study switches and page reloads.
Remove individual chips or use **Clear selection** to remove candidates explicitly.
Hover highlights matching trials in the table, scatterplot and configuration comparison.

**Show Pareto Frontier** visualizes non-dominated duration/objective pairs without changing
selection. Duration is minimized; the native objective direction comes from Optuna.
Derived user attributes use the direction chosen in the sidebar. When studies disagree
on a native objective's direction, the frontier is disabled with an explanation.
Native objectives are matched by metric name across studies; unavailable metrics are
left missing rather than substituted with another objective.

Parallel Coordinates shows only collected candidates, using their selection colors.
Hide parameters through **Hyperparameter axes** and drag axis labels to reorder them.
Logarithmic parameters use log scaling with original-value tick labels. Categorical
values keep their types, and missing values occupy a separate labelled lane rather
than numeric zero. Parameters with conflicting types or scales across studies are
shown as labelled categories with an explanatory note. Wide plots scroll horizontally.

**Export TOML** downloads ``[[selected_trials]]`` records containing ``study_name``,
``study_id``, the original internal ``trial_id``, study-local ``trial_number`` and
``storage`` identity. This references existing trials; it does not enqueue configurations.
Database snapshots retain the original IDs even while used offline. Older snapshots
without source IDs must be refreshed from the database before export. The download is
disabled when the selection is empty or contains a trial whose original IDs are unknown.

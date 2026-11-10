Select Candidates
=================

Collect concrete Optuna trials for a subsequent drill-down analysis. The sidebar selects
the primary study and central objective; **Additional studies** adds other loaded studies.
The local **Table objectives** selector controls only the table's objective columns.

Selected trials stay at the top of the Trial Explorer in their selection colors.
Column sorting applies within selected and unselected trials separately. Selecting a
trial returns the table to its first page; active filters still apply.

The Trial Explorer supports column sorting and filtering. Press Enter to apply a column
filter. The scatterplot shows exactly that filtered set, excluding trials without a finite
central objective or valid duration. Only completed trials are shown.

Click a row, checkbox or scatter point to toggle a candidate. Box and lasso gestures add
all enclosed trials without clearing earlier selections. Selected Trials persists in the
browser session across filters, objective changes, study switches and page reloads.
Remove individual chips or use **Clear selection** to remove candidates explicitly.
Hover highlights matching trials in the table, scatterplot and configuration comparison.

**Show Pareto Frontier** visualizes non-dominated duration/objective pairs without changing
selection. Duration is minimized; the native objective direction comes from Optuna.
Derived user attributes use the direction chosen in the sidebar. When studies disagree
on a native objective's direction, the frontier is disabled with an explanation.
**Show elites** zooms both axes to the best trials within the filtered scatterplot
and highlights them in amber. Switching it off restores the full filtered view.
Choose Top 1%, 5% (default), 10% or 20%; the central objective's direction determines
ranking. At least one valid trial is shown with padding around the elite range.
Elite zoom and highlighting are disabled when
studies disagree on direction and does not change the candidate selection.

Native objectives are matched by metric name across studies; unavailable metrics are
left missing rather than substituted with another objective.

Parallel Coordinates shows only collected candidates, using their selection colors.
Hide parameters through **Hyperparameter axes** and drag axis labels to reorder them.
Logarithmic parameters use log scaling with original-value tick labels. Categorical
values keep their types, and missing values occupy a separate labelled lane rather
than numeric zero. Parameters with conflicting types or scales across studies are
shown as labelled categories with an explanatory note. Wide plots scroll horizontally.

**Export TOML** downloads the selected parameter configurations as ``[[trials]]``.
Each entry contains only model parameters. When all selected trials share one source,
``source_optuna_study_id`` and ``source_study_name`` appear once at the top level.
No per-trial source metadata or objective-selection block is exported. The download is
disabled when the selection is empty or a source study ID is unknown.

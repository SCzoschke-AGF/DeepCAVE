Sensitivity Analysis
====================

Inspect how reproducible a configuration's performance is across random seeds. In
**General → Optuna metadata**, map **Model Seed** and/or **Data Seed** to existing
study or trial user attributes. Field names are not assumed. The sidebar selects the
study (or native group of studies) and central objective, including derived objectives.

The analysis defaults to **Combined**, using the observed model/data seed pairs.
When only one seed type is mapped, that seed type is used automatically.
With only one mapped seed type, observed variability can include
other unknown random sources and must not be interpreted as a causally isolated effect.

Configuration identity uses the complete original Optuna parameter dictionary, including
types and exact floating-point values. Mapped seed parameter names are removed, as are
conventional seed parameters whose values match the mapped seeds. ``config_id`` is an
audit label, not a configuration key. Trial identity includes source storage and study.

Cross-study grouping requires matching model, dataset, feature set, CV strategy and
objective-definition attributes. Supported context keys include ``model`` / ``model_type`` /
``model_family``, ``dataset`` / ``dataset_id`` / ``dataset_hash`` / ``data_hash``,
``features`` / ``feature_set`` / ``feature_set_id``, ``cv`` / ``cv_strategy`` and
``objective`` / ``objective_definition`` / ``metric``. All present context keys must
match, including ``experiment_id`` and ``n_splits``. An experiment ID alone does not
establish comparability. Missing definitions keep groups study-local. Conflicting study
and trial context values stay separate. The experiment control filters these groups;
it does not redefine their context.

The summary describes the current experiment context before the minimum-seeds filter;
the comparable count applies that threshold. Median seed count is calculated per
conditional configuration row. Non-completed trials and missing/nonfinite objective or
mapped seed values are excluded with counts. The minimum is fixed at two distinct seeds.
Multiple evaluations of the same seed pair remain visible as individual trials, but
contribute one equally weighted pair mean to seed statistics and bootstrap analyses.

The overview shows the configuration boxplots beside the mean-performance/variability
scatterplot. Click a boxplot trial for active details; lasso scatter points
for comparison. These are separate states. Pareto efficiency optimizes mean performance in the objective's
direction and minimizes sample SD. Diamonds flag estimates from fewer than four seeds.
Best-seed gap uses the best seed-pair mean and depends on the number of replications.

The overview shows one boxplot per visible configuration, with raw evaluations scattered
inside each box. Boxes summarize equally weighted seed-pair means. Only observed pairs
contribute. Trial tables retain original study/trial IDs,
seed values, durations, objective values and source configuration labels. Clicking a raw
point highlights its trial. Trial checkboxes select observations for the existing
**Select Candidates** session collection. Add the whole configuration or just checked
trials; the candidate chips link back to this analysis. Choose the corresponding sidebar
study/group when following a link. **Locate selected candidate** finds a collected trial
in the current context.

**Trials by Seed per Configuration**, in the Distribution tab, shows individual
evaluations grouped by each fixed seed. Blue points on the left show **Data seed variability**
at each fixed model seed; orange points on the right show **Model seed variability** at
each fixed data seed. Tick labels show only actual values of the fixed seeds. Colored
lines and headings above the plot identify both variability sources. Below the ticks,
**Model seeds** and **Data seeds** identify the fixed seeds, with the configuration label beneath.
Vertical lines span the raw minimum and maximum per seed. Transparent points show
individual evaluations. Horizontal marks summarize equally weighted seed-pair means, so repeated
evaluations stay visible without receiving extra weight in the mean. Hover identifies
the configuration, original study/trial, both seeds and objective. The plot follows the
current sensitivity mode, Main Objective, context and configuration filters.

In Distribution, **Show reference trial objectives** overlays red diamonds on the first
plot, one per configuration with an assigned reference and an available value for the
currently selected objective. Hover shows the original study, trial number and objective
value. Toggling the reference markers updates only the first plot and preserves its zoom
and the other distribution plots. These markers do not enter the seed statistics. Displaying them uses saved
references. Older assignments containing only native values are supplemented with the
known trial's numeric objective attributes by an exact, read-only lookup when enabled;
this does not search for matching configurations or change the assignment. Missing
objective values and read errors are reported below the plot.

The **Reference Trials** tab lists all visible configurations, each with its own reference
dropdown on the right. Each configuration appears once even when multiple fixed-seed
conditions are visible. No active-row selection is required.
**Find reference trials** searches the SQLite database configured under General, only
when clicked. It searches the currently visible configurations in other studies for
identical typed parameter values, ignoring the mapped seed parameters and conventional
seed aliases. Dataset metadata must match: ``dataset_sha256``, ``dataset_hash`` or
``data_hash`` is preferred, with ``dataset_id`` and ``dataset`` as fallbacks. Configurations
without dataset metadata are reported rather than matched by study name. Known model
families must also agree. Only completed trials are offered; the database is read-only.
Choose the original selection trial in the dropdown beside its configuration.
Options show study, trial number, objective values and seeds. The assignment is saved
with DeepCAVE's persistent settings, independently of the sensitivity mode and fixed-seed
condition. Saved assignments remain available without searching again and do not change
the seed statistics. Clear the dropdown to remove an assignment.

Seed Effects displays categorical seed identifiers, observed combination coverage,
missing pairs and counts for repeated cells. A coupled design cannot identify separate
model/data effects. Heatmaps show pair means explicitly, retaining raw records in
Distribution. Both mapped seed types appear in the configuration's seed grid.

Parameter sensitivity shows all observed hyperparameters in a grid with up to three panels
per row. Panels share a seed standard deviation axis and a mean-performance color scale;
marker area indicates seed count relative to the median of the visible configurations,
with shared legends on the right. Equal seed counts use the standard marker size.
Original categorical/integer/log parameter semantics are preserved. Conflicting
search-space scales are displayed categorically.
Associations in an adaptively sampled HPO space are descriptive, not causal effects.

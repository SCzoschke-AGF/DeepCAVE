HPO Planner
===========

The HPO Planner plugin is an interactive search-space planner for the next optimization run.
Select the main objective below the active study in the global sidebar and three distinct elite
proportions in the compact toolbar. Every parameter then occupies one row, and the complete
planned space is collected in the planner's summary card.

The planned search space is shared with the 2-D Analyses plugin for the active run. It remains in
browser memory while navigating between plugins and is cleared by reloading the page or opening a
new tab. Runs and groups retain independent selections during that browser session. Resetting the
planner restores the complete current ConfigSpace only for the active run.

In addition to native Optuna objectives, the main-objective selector offers trial user attributes
that contain a finite scalar number for every successful trial. User-attribute objectives can be
maximized or minimized with the direction selector; native objectives retain their Optuna
direction. Objective 0 remains the default. The selected main objective consistently controls
elite groups, parameter summaries, detail plots, and trial coverage.

For numerical parameters, zero and one always represent the lower and upper bounds of the current
ConfigSpace definition. Each elite group is shown on a compact separate subtrack with a Q10–Q90
band and a median marker. A thinner, lighter line behind it shows the complete observed minimum to
maximum range of that elite group. The selected range can be changed with the range handles or by
entering original parameter values; both controls use one shared planning state. Integer and step
metadata are respected.

The ``Show Top N`` field in the elite-groups toolbar controls how many of the globally best trials
can be overlaid on each parameter row (five by default). The marker button beside a parameter's
detail-plot button toggles this overlay independently for that parameter. Vertical orange strokes
show the exact positions of the requested top trials on numerical sliders; categorical parameters
use an equivalent category-position track. Conditional parameters omit trials in which they were
inactive.

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

The sidebar reports wider, narrower and unchanged ranges and provides a live Python-dictionary
preview. Every committed range or category change is automatically written to DeepCAVE's
persistent metadata for the active run and objective. A later change replaces that run-objective
pair directly; no revisions, timestamps, or older states are retained. The persisted selection is
restored after a reload and shared with 2-D Analyses for the same run and objective. Other runs and
objectives use separate entries. The planner compares only the current and planned search spaces.
It deliberately does not infer model- or library-specific validity limits.

Parameter Detail Plot
---------------------

The plot button below every parameter name opens a combined detail plot in a non-blocking panel on
the right. The parameter range remains visible and editable while the panel is open. A segmented
control switches the enrichment between the three configured elite groups. Bars use the matching
planner colors: purple for the broadest group, turquoise for the middle group, and orange for the
strictest group. The dashed line marks the ``1.0x`` baseline. Median and q90 objective values use
the same horizontal ranges and the right axis, making performance and elite concentration directly
comparable. Numbered diamonds below the ranges mark the bins containing the five best active
trials, using the same objective-aware ranking as the 1-D Analysis. Short guide lines connect the
diamonds to their horizontal bin labels. Small numerical ranges share an engineering scale factor
in the axis title (for example, ``learning_rate (×10⁻³)``) so their labels remain compact without
discarding relevant precision.
Integer parameters use individual integer bins when possible and otherwise use inclusive groups of
consecutive whole numbers; decimal bin boundaries are never shown for them.

Top-Trial Coverage
------------------

The coverage plot below the planner ranks a selectable 1–100 successful trials for the main
objective. The default is 10 trials; if fewer trials are available, all available trials are shown.
Green points are configurations that remain fully inside the planned search space; red points
violate at least one selected parameter range or categorical choice. Inactive conditional
parameters do not count as violations. The badge reports the included share and the hover details
identify every violated parameter, so range edits immediately show which strong configurations
would be excluded. Objective direction is stated on the vertical axis, and values spanning at
least two orders of magnitude use a logarithmic scale.

Optuna trial user attributes containing a non-empty finite numerical list for every successful
trial are available in the spread selector below the top-trial slider. Selecting an attribute
automatically enables ``Show spread``. The error bars can show either one population standard
deviation (``SD``) around each original main-objective point or the minimum-to-maximum ``Range``.
The optional ``Scatter`` overlay shows every individual attribute value with a small horizontal
offset at its trial rank. Individual values are also included in the hover details. No equality
between the list mean and the main-objective value is assumed or checked.

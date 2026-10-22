Search Space Coverage
=====================

The Search Space Coverage suite is available under Configuration Analyses and
uses the globally selected run or group. Coverage statistics are independent of
objective values; optional cell-performance coloring and elite-trial selection use objective scores.

Overview, marginal coverage, and the selected parameter-pair scatter are
available immediately. Opening the page, selecting parameters, changing settings,
and receiving additional trials never start an expensive analysis.

Diagnostic workspace
--------------------

Five internal tabs keep each question in its own view:

* **Overview** shows trial counts, parameter types, and the recorded sampler.
  The search-space definition is collapsed initially. Samplers
  are read from run metadata; missing sampler metadata is labelled as not recorded.
* **Univariate** shows compact parameter rows with plots and KPIs, with grouping,
  sorting, and a problem filter. Type groups remain visible. Switch between
  Distribution and ECDF for continuous and integer parameters; categorical and
  ordinal parameters retain frequency plots.
  Frequency tables and interpretation notes live in collapsed parameter details.
* **Pairwise** displays a clickable coverage matrix with occupancy, empty-cell,
  MMD distance, and maximum coverage-radius choices for continuous, integer, ordinal,
  categorical, and mixed pairs. A cell selects the scatter and pair KPIs.
  Manual parameter selection works before computing the matrix. The ranking table
  is available under **Show detailed table**.
  X and Y selectors span the plot panel. A fixed sidebar shows the pair KPIs at
  the top and reserves space for the cell-color legend below, keeping axes and metrics stationary.
  All matrix colors express **Coverage concern**: light cream means low concern,
  and dark rust/bordeaux means high concern. Cell labels retain the original
  metric. Heatmap tooltips are disabled; cells remain clickable.
  Text contrast follows the cell background.
  The 2-D plot combines three independent layers, controlled by a compact toolbar
  below the plot: **Cell coloring** (None, Density, Performance), **Trial markers**
  (None, All, Elites), and the **Highlight empty cells** switch. Defaults are
  Density, All, and highlighting enabled. Both content layers cannot be set to None.
  Density uses the existing occupancy grid and trial-count legend; Performance
  aggregates the best finite objective per cell and shows a continuous objective
  scale. Coverage metrics always use all jointly active trials, including when
  markers are hidden or restricted to elites.
  Continuous axes use sampling coordinates; discrete axes show
  original valid values, grouping support values when they exceed the bin limit.
  **Trial markers: Elites** shows only larger orange diamonds for the best successful
  trials. The header dropdown selects 20%, 10%, 5%, 2%, or 1% (default 10%).
  Ranking uses the first objective other than the synthetic Time objective and
  respects minimization/maximization; the objective name appears in the panel title.
  Trials without finite scores are excluded, and ties at the quantile threshold
  are included. Elite selection spans the run; only jointly active trials appear
  in the selected plot. Layer changes do not change coverage calculations or
  start worker jobs. Trial tooltips remain disabled; cell hover shows density or performance.
* **Global** places **Performance-aware Coverage** to the right of the existing
  Coverage Array and Improve Coverage panels (stacked on narrow screens). It uses
  **Main Objective** from the sidebar. Its own **Refresh** starts an independent
  worker job; changing the objective never starts a job or changes the left panels.
  All successful trials with a finite selected score and compatible configurations,
  at all budgets, enter the analysis, including duplicate configurations.
  The existing Rashomon importance-weighted **RMS** distance is reused unchanged,
  including logarithmic, categorical and inactive-parameter handling. Its global
  fANOVA main-effect weights are fitted to successful trials at the highest budget,
  with 16 trees and seed 0. The panel exposes the weights and their provenance;
  unavailable weights produce an explanation instead of an unweighted fallback.
  Every unique trial pair enters the computation. Small studies show every pair;
  larger studies show a full-count 2D density histogram. Median and P90 lines join
  empirical distance-quantile groups, merging equal distances. Pair summaries also
  retain mean, P95 and the fraction strictly above the relevance threshold.
  The nearest-neighbor table uses one leave-one-out observation per valid trial,
  grouped independently by nearest-neighbor distance quantiles. Distance ties
  select the lowest trial ID, then history row. Identical configurations can be
  neighbors at distance zero. The editable threshold defaults to 0.01 in the raw
  units of the selected objective. Exact threshold changes use cached sorted
  differences without recalculating configuration distances. Cache inputs include
  trial/configuration content, objective, normalization and importance weights.
  These diagnostics describe dependent observed pairs, potentially including
  evaluation noise; they provide no calibrated predictions for unseen configurations.
* The left **Global** column starts with **Combinatorial Coverage**: a measured resolution/strength
  **Coverage Array** for 2, 3, 4, and 8 levels. Each resolution shows the achieved
  strength and coverage at the next strength, so near-complete interactions remain
  visible even when the strict strength is low. Every combination of every group
  must be covered to reach a strength; all lower strengths are then complete too.
  Select a resolution to see its coverage strip and groups preventing the next
  strength. Compact KPIs count analyzed groups, fully covered groups and the largest
  missing count. The two-column group table combines missing counts and coverage;
  neutral parameter tags wrap even for long names. Groups are sorted by missing
  count across the complete result, with deterministic ties. Initially five groups
  are visible; show ten, twenty or all. An AND parameter filter updates the table,
  KPIs and discrete missing-count distribution, including fully covered groups.
  Click a group to load its actually missing level tuples, paginated in sets of
  twenty. Cells show the actual value ranges saved by the original discretization
  instead of abstract level labels; categorical values remain directly readable.
  Collapsed level definitions retain the mapping. Filtering, changing the display limit and tuple
  pagination reuse saved results without starting coverage or reference jobs.
  The distribution shows observed groups only: the global random baseline does not
  provide group-specific calibration. Older cached results remain visible and ask
  for a manual refresh before the new group diagnostics are available.
  Click the same row again to close its details; selecting another row switches
  the diagnosis to that resolution.
  Strength is shown relative to the number of varying parameters (for example,
  ``t = 4/6``). Beneath it, **max. feasible strength** shows the upper bound imposed
  by the trial count. This bound is necessary, not sufficient, for a covering array
  to exist. **Method & level definitions** provides the calculation details. Each resolution is evaluated
  up to its first incomplete strength, the parameter count, or the trial/work limit;
  there is no fixed cap at strength 6. Uncomputed strengths are marked explicitly.
  An exact strength of zero means marginal coverage is incomplete. There is no
  additional parameter-group selector or duplicate pairwise drill-down.
  A dash means full coverage requires more trials; an ellipsis means exact
  enumeration exceeds the work limit. Strength is shown as a lower bound when
  higher feasible strengths remain uncomputed. Work is bounded to 100,000 groups
  and 20 million trial/group projections per resolution; groups are never sampled.
  Continuous levels follow the theoretical sampling CDF; ordered discrete values
  are grouped near theoretical quantiles while keeping every level nonempty.
  Categories and ordinals retain their original values, so mixed group sizes are
  supported. Constants are excluded. This first version requires independent
  parameters without conditions or forbidden combinations. Only successful trials
  contribute; repeated successful configurations remain in the history.
  A single **Compute coverage** / **Refresh** button in the header updates both
  observed coverage and the random baseline in one worker job. The header also
  shows the reference count and update age. Each row displays observed next-strength
  coverage as a point on a zoomed track, alongside the random 5–95% band and
  dashed median. Its tooltip reports the percentile and whether coverage falls
  below, within or above the usual random range. All rows share the smallest lower
  and largest upper endpoint across their zoomed ranges, making the points and bands
  directly comparable. Each row labels these shared endpoints.
  The strength badge's tooltip includes the corresponding random strength
  distribution. Constant reference distributions
  have no percentile; censored strength estimates are labelled as lower bounds.
  Calibration simulates complete studies of the same size, preserving dependence
  between groups. Reference designs and seed apply; reference sample count does not.
  Each simulated study is evaluated at least through the observed next strength,
  even if the simulation encounters a gap earlier, so coverage comparisons always
  use all reference studies at the same interaction strength.
  Older references without medians or strength samples are marked for refresh;
  their available coverage bands remain visible without inventing missing values.
* **Growth** includes Coverage Radius and uncovered regions. Expanding radius details
  shows distances and a selectable list of region parameters. It also shows one
  growth metric at a time: p95 radius, maximum radius, centered
  discrepancy, pairwise occupancy, or marginal maximum gap. An optional random
  overlay is enabled only after its independent calculation and only when trial
  fingerprints and reference settings match the realised trajectory. The random
  job also contains its own realised trajectory and can be viewed independently.

**Analysis settings** in the header opens a drawer for simulation counts and seed.
Tab switches, metric changes, matrix clicks, region selection, and opening settings
only change the view; they never submit calculations.

Trial requirements forecast
---------------------------

The Coverage Array includes **Additional trials**, with separate **Next** and
**Full** targets. **Compute forecast** starts an independent background job;
opening the page or changing settings does not start computation. The default
80% means an 80% probability of reaching **100% coverage** of the target strength,
starting from the existing successful configurations. Full strength always means
all varying parameters, independently of the current trial-count feasibility bound.
Constants do not contribute to strength. If current strength is only a lower bound,
the next incomplete strength is unknown and its forecast is unavailable.

The sampling model is independent random sampling from the displayed ConfigSpace,
including its category weights, discrete support, steps and distributions.
Continuous levels partition the theoretical sampling CDF and have equal masses;
discrete bin masses are calculated from the configured distribution. Observed
TPE frequencies are never used to infer random probabilities. For groups, the
displayed merged ConfigSpace defines the future distribution; this is not an
inferred mixture of member studies. Forecasts assume future configurations
contribute to coverage; they do not predict failures, pruning or adaptive TPE
behaviour. Conditional spaces, forbidden combinations and unknown distributions
are explicitly unsupported.

Next-strength Monte Carlo continuations share the same historical missing-tuple
state and use independent reproducible seeds. The default is 2,000 continuations;
the detail view reports 95% confidence intervals for the estimated quantile.
Any incomplete continuation or exhausted runtime/work budget produces a censored
result without a quantile from the successful subset. A conservative sufficient
budget may still be available. Large t-way calculations use a union bound that
remains valid for dependent coverage events. Such a bound is labelled **bound**,
not displayed as an estimated quantile.

Full-strength forecasts count unique historical level configurations. Small
uniform spaces use an exact occupancy chain; nonuniform spaces with at most twelve
missing coupons use inclusion-exclusion. Larger supported coupon sets use a
labelled Poisson approximation, with rigorous bounds on the exact quantile.
Large uniform spaces require only total and missing counts, without enumerating
the Cartesian product. Large nonuniform spaces use conservative bounds when
explicit coupon enumeration would exceed the configured limit.

Analysis settings provide probabilities of 50%, 80% and 95%, simulation count,
trial horizon and a shared forecast time limit. The API additionally exposes tuple,
group, projection and explicit-coupon limits through ``forecast_options``.
Results retain per-target statuses, actual simulation counts, seed, limits,
binning definitions and probabilities, search-space hash and algorithm version.
They are cached separately from the random baseline. Changes to trials, search
space, settings or algorithm mark forecasts stale; only explicit recomputation
starts a new job. A forecast is shown beside measured coverage only when their
source fingerprints match. Existing coverage and baseline values are unchanged.

Use ``SearchSpaceCoverage.process(run, {"analysis": "forecast", ...})`` or
``compute_coverage_forecast(data, options, observed=existing_coverage)`` for API use.
The latter can reuse an existing measured snapshot.

Improve Coverage
----------------

The **Global** tab includes an expandable **Improve Coverage** planner. It uses
the selected run or the existing merged group search space, including the minimum
and maximum bounds across group members. The planner supports independent search
spaces regardless of the sampling method. Missing or incorrect sampler metadata
does not prevent planning. Conditions and forbidden clauses are rejected.

Select the resolution used by the Coverage Array (2, 3, 4 or 8 levels). Selecting
an array row also selects that resolution for planning. Categories and ordinals
retain all original values; constants do not increase strength. Choose:

* **Maximize Coverage**: generate up to the required additional trial budget.
* **Minimize Trials**: seek full target coverage with an optional maximum trial
  budget. This is a heuristic search, so the result need not use the fewest trials.

The default target is the next incomplete strength. Custom and maximum strength
are also available. Unlike the overview calculation, planning measures the selected
strength even when there are too few observed trials to reach it. Advanced settings
control search time (0.1–120 seconds) and seed. Numerical proposals use the same
sampling CDF, level boundaries and step lattices as the measured array.

Only **Generate plan** or **Recompute plan** starts a worker job. Opening the
section, selecting a resolution or changing settings never starts planning or
random-baseline jobs. **Cancel** retains the previous completed plan.

The saved result shows observed and projected coverage separately, a cumulative
projection and a paginated table of concrete parameter values and marginal gains.
Select a table row to inspect its validated configuration and the combinations it
would newly cover. Observed coverage changes only when successful trials are
actually recorded. New trials or changed settings mark the plan as outdated; it
remains available until an explicit recomputation. Plans persist with the other
coverage results across page navigation and server restarts; clearing plugin caches
removes them. Planning never executes trials or modifies the source study.

Search uses greedy selection with missing-tuple completion, random candidates,
mutations and local improvement. Redundant configurations are removed from complete
Minimize Trials plans. Gains count each combination once, including overlap between
proposals. A fixed-group lower bound reports the minimum number of additional trials
that could possibly complete the target. If the budget is below that bound, full
coverage is explicitly marked impossible within that budget, while the improving
partial plan remains available. Other partial results do not prove impossibility.
Optimality is only reported when full coverage proves the maximum coverage objective,
or a complete Minimize Trials plan matches the lower bound.

Exact tuple accounting is bounded to 10,000 parameter groups, five million tuples
and 20 million observed trial/group projections. Oversized problems are rejected
with a request to reduce strength or resolution. Unbudgeted plans also stop at
1,000 additional trials. Search timeouts retain validated progress. Approximate
coverage and exact solvers are not part of this version.

Independent analyses
--------------------

Visible modules have their own **Run analysis** / **Recompute** button and run
through the existing DeepCAVE worker queue. Legacy Sampling distribution and
Random comparisons remain available through the Python API, outside the dashboard:

* Pairwise Coverage: nominal grid occupancy, empty-cell fraction, MMD distance,
  and maximum nearest-trial coverage radius for every parameter type. Reference count
  and seed in Analysis settings control the legal reference configurations.
  Select a matrix or ranking-table cell to update the parameter-pair scatter.
* Sampling distribution: mixed-space MMD and centered, L2-star, and wrap-around
  discrepancy, including their trajectories over approximately twenty nested trial
  prefixes. MMD uses a product of Gaussian ordered-coordinate kernels (width 0.2)
  and category equality kernels, with inactive values as a separate state. The
  nonnegative biased empirical MMD is compared with random designs of the same
  trial count using a shared reference sample. To bound work, MMD uses up to 1,024
  trials per checkpoint and 512 reference configurations; reproducible subsampling
  follows the same rule for observed and random histories. Details disclose the
  effective counts. Reference count and seed apply to this module.
* Coverage Radius & Holes: nearest-trial distance distribution, median, p90,
  p95, p99, maximum, and up to twenty distinct worst-covered reference points
  decoded into parameter values.
* Random comparisons: coverage relative to identically sized random designs,
  reporting median, mean, 5–95% interval, midrank percentile, and standardized
  deviation. Each discrepancy definition has its own comparison and trajectory
  band. Discrepancy need not decrease with every added trial; it evaluates
  uniformity rather than optimization performance.
* Coverage Growth: approximately twenty logarithmically spaced trial prefixes,
  including the final trial, with all five growth metrics.
* Random Coverage Growth: a separately started simulation with random mean and
  5–95% bands. Random trajectories use nested prefixes of full designs.

The two growth jobs are independent. The random-growth job can also be started
directly and includes the realised trajectory for comparison. Starting realised
growth does not start its random baseline.

The full-space MMD is a candidate distribution diagnostic, not a validated
guarantee of coverage. Synthetic validation found that its current product kernel
loses sensitivity as the number of parameters grows, and can miss absent
categories or combinations. See the :download:`kernel review and sensitivity study
<../validation/coverage_mmd/README.md>` for the reproducible protocol, measured
detection rates, and limitations. A result within the random range does not rule
out a coverage problem.

A separate :download:`randomized partitioning experiment
<../validation/coverage_partitions/README.md>` compares unvisited cell mass
against MMD on the same synthetic defects, including jointly calibrated selection
and fixed-partition forecasts. It is not integrated into dashboard metrics.

Reference designs defaults to 500; reference samples defaults to 2,048. Both
baselines use the same fixed reference points for observed and random designs.
The seed makes simulations reproducible. Settings take effect only on an
explicit start, and only affect the modules that use them.

Sampling coordinates and interpretation
---------------------------------------

Preprocessing is cached once per loaded run revision and reused by all modules.
Continuous parameters use their sampling CDF, including linear, logarithmic,
Normal, and Beta distributions. Discrete parameters use probability-mass
midpoints; category weights and metadata-defined integer/float steps are retained.
Inactive conditional parameters remain missing and contribute only when active
to marginal and pairwise statistics. Reference designs obey conditions, forbidden
combinations, and step lattices without mutating the original search space's RNG.

Only trials with ``Status.SUCCESS`` contribute to every coverage analysis and
plan. Failed, pruned and other unsuccessful trials are excluded. Overview counts
still show how many recorded trials were excluded. Repeated successful
configurations at different seeds or budgets remain in the history; they do not
cover new combinatorial tuples. Unique configuration counts are also shown.
Growth follows successful-trial history order rather than sorting configurations.

Continuous marginal coverage is occupied-bin fraction; discrete marginal coverage
is visited-value fraction. Min/max range fraction is in original value coordinates,
while normalized gaps and edge gaps are in sampling coordinates. The largest raw
gap is between adjacent observed numerical values. Continuous plots mark the
largest normalized gap with an orange bracket, boundary markers, and a trial rug.
This includes search-space edge gaps: an explicit label identifies a
search-space boundary, while orange rug ticks identify actual observed trials.
Hover shows original boundary values and the width in transformed coordinates.
Bins strictly inside a gap are empty, but a gap inside one occupied bin remains
visible through the rug. Expected frequencies use a thin grey dashed reference.
Rug marks retain all active trials; gap boundaries are never inferred from bins
or the subsampled ECDF. With no active trials, no bracket or trial marks are drawn.
Integer missing runs count
unvisited lattice values, including edges. Frequency tables for ranges above 1,000
values show visited values only; their coverage and unseen counts still use the
complete theoretical range. Continuous plots use at most 1,000 ECDF points, while
the statistics use all trials. The KS statistic is descriptive, without an IID
significance claim for adaptive or repeated trials.

Integer ECDFs use original support values, including the configured step, rather
than sampling coordinates. The observed curve accumulates empirical counts; the
expected curve accumulates the configured discrete probability masses. Both are
step functions. Explicit integer probabilities can be supplied as
``meta={"probabilities": [...]}``, in ascending valid-support order, and are
normalized to sum to one. Otherwise the configured distribution determines the
masses, including nonuniform logarithmic distributions.

Integer KS distance is the maximum absolute CDF difference evaluated on the
discrete support. It is descriptive; no continuous-distribution KS p-value is
computed. For support above 1,000 values the plot shows selected valid support
points, while the distance remains exact. Parameters with no active observations
have an expected CDF but no empirical CDF or KS distance.

Classical cube discrepancy applies to the unconditional continuous projection;
the selected parameters are shown beside each result. It is unavailable when
there are no eligible parameters or the space contains forbidden combinations.
Discrete and conditional axes are not silently imputed into a uniform cube.
Discrepancy growth and baselines use the same projection as the observed design.

Nearest-trial distance is the root mean square of per-parameter squared distances.
Numerical and ordinal parameters use sampling coordinates. Categorical mismatch
and active/inactive mismatch contribute one; two inactive values contribute zero.
Constant parameters are excluded. The resulting radius lies between zero and one.
Uncovered regions are sampled candidates, rather than certified global holes.

Pairwise occupancy uses nominal cells and co-active trials. Some nominal cells
may be unreachable under conditions or forbidden combinations, so low occupancy
is a potential concern rather than proof of a sampling problem. Discrete axes
with more values than bins are grouped.
Pairwise MMD compares the empirical joint distribution against legal reference
configurations conditioned on both parameters being active. It uses the biased
empirical estimator from `Gretton et al. (2012)
<https://jmlr.org/papers/v13/gretton12a.html>`_, with a product of Gaussian kernels
for ordered parameters (fixed scale 0.2) and equality kernels for categories.
Continuous coordinates follow the sampling CDF; discrete ordered coordinates use
the valid support ranks scaled to [0, 1]. Category ordering has no effect.
MMD uses reproducible subsets of at most 1,024 observations and 512 references;
it is independent of histogram binning. Scores depend on kernel scale and sample
counts and are not interchangeable with the former centered discrepancy.

Pairwise coverage radius is the maximum normalized RMS distance
from a joint-active reference configuration to its nearest joint-active trial.
It uses the same coordinates, categorical mismatches contribute one, and all
observations and references are considered. Both scores are descriptive and lower
is better; no p-value is computed. If there are no joint observations or no
joint-active reference samples, the reason is recorded explicitly. Expected
reference frequencies describe the defined sampling distribution, including
steps, weights, conditions, and forbidden combinations. Changing metric
definitions invalidates old cached results; Recompute supplies the new scores.

Matrix colors use a separate attention score on a fixed 0–1 scale, never the
minimum and maximum of the visible matrix. Occupancy uses ``1 - occupancy``;
empty cells use their raw fraction, giving identical colors for these complementary
metrics. Radius uses ``sqrt(radius)`` and MMD uses ``sqrt(MMD / sqrt(2))``.
These fixed transforms use the mathematical bounds of normalized RMS distance
(0–1) and MMD with a nonnegative, unit-bounded kernel (0–sqrt(2)). The square root
makes smaller distance differences easier to see. The scale remains stable across
runs with comparable metric settings; it does not imply calibrated statistical
thresholds, and MMD still depends on kernel and sample counts. Unavailable pairs
and self-pairs remain uncolored.

Caching and status
------------------

Modules expose ``not_run``, ``running`` (including queued), ``completed``, ``stale``,
and ``failed``. Each result records its trial fingerprint, trial count, analysis
version, and relevant settings. New trials or changed settings mark old results
outdated and leave them visible. Recalculation always requires **Recompute**.
An in-flight job uses its submitted snapshot; if more trials arrive, its completed
result is immediately marked outdated. Failures preserve any previous result.
Each active module exposes **Cancel**, which stops only that module's job and
preserves its previous result. Completed cards show the calculation timestamp and
simulation settings used. Older cached results remain visible as outdated after
the analysis version changes; new growth metrics become available on recomputation.

Results persist separately per run and module in the shared plugin-cache tree,
outside the run directories invalidated by run refreshes. **Clear Plugin Caches**
removes these results and the shared preprocessing cache. Jobs continue across
page navigation, and are collected when the suite is reopened.

Python API
----------

``SearchSpaceCoverage.process(run, {})`` returns only overview and univariate
results. For a single explicit expensive analysis, pass ``analysis`` as one of
``combinatorial``, ``combinatorial_baseline``, ``pairwise``, ``discrepancy``,
``radius``, ``baseline``, ``growth``, or
``growth_baseline``. The evaluator also exports separate calculation functions
for callers that want to reuse one normalized ``CoverageData`` snapshot.

For planning, use ``analysis="planning"`` with ``strategy="maximize_coverage"``
or ``"minimize_trials"``, ``resolution``, ``strength_mode="next"`` (or ``"custom"``
with ``target_strength``, or ``"maximum"``), ``max_additional_trials``,
``solver_timeout`` and ``seed``. The returned dictionary is the validated plan;
no trials are executed. Sampler metadata is not used to determine compatibility.

Search Space Coverage
=====================

The Search Space Coverage suite is available under Hyperparameter Analysis and
uses the globally selected run or group. It is independent of objective values.

Overview, marginal coverage, and the selected parameter-pair scatter are
available immediately. Opening the page, selecting parameters, changing settings,
and receiving additional trials never start an expensive analysis.

Diagnostic workspace
--------------------

Five internal tabs keep each question in its own view:

* **Overview** shows trial counts, parameter types, the recorded sampler, and a
  live status list. The search-space definition is collapsed initially. Samplers
  are read from run metadata; missing sampler metadata is labelled as not recorded.
* **Univariate** places parameter KPIs beside one plot. Switch between Distribution
  and ECDF for continuous parameters; discrete parameters use frequency plots.
  Frequency tables and interpretation notes live in collapsed parameter details.
* **Pairwise** displays a clickable coverage matrix with occupancy, empty-cell,
  discrepancy, and hole-score choices. A cell selects the scatter and pair KPIs.
  Manual parameter selection works before computing the matrix. The ranking table
  is available under **Show detailed table**.
* **Global** shows separate Discrepancy, Coverage Radius, and Random Baseline
  cards. Each card has its own start and details controls. Only one detail section
  is visible at a time. Uncovered regions use a five-row paginated list of region
  numbers and distances; selecting a region displays its parameter values and
  explicitly marks inactive parameters. Baselines show the observed point against
  random intervals, with the axis direction indicating better/worse coverage.
* **Growth** shows one metric at a time: p95 radius, maximum radius, centered
  discrepancy, pairwise occupancy, or marginal maximum gap. An optional random
  overlay is enabled only after its independent calculation and only when trial
  fingerprints and reference settings match the realised trajectory. The random
  job also contains its own realised trajectory and can be viewed independently.

**Analysis settings** in the header opens a drawer for simulation counts and seed.
Tab switches, metric changes, matrix clicks, region selection, and opening settings
only change the view; they never submit calculations.

Independent analyses
--------------------

Each module has its own **Run analysis** / **Recompute** button and runs through
the existing DeepCAVE worker queue:

* Pairwise Coverage: nominal grid occupancy, empty-cell fraction, approximate
  numerical hole radius, and eligible two-dimensional centered discrepancy.
  Select a matrix or ranking-table cell to update the parameter-pair scatter.
* Global Discrepancy: centered, L2-star, and wrap-around discrepancy.
* Coverage Radius & Holes: nearest-trial distance distribution, median, p90,
  p95, p99, maximum, and up to twenty distinct worst-covered reference points
  decoded into parameter values.
* Random Baseline: coverage relative to identically sized random designs,
  reporting mean, 5–95% interval, midrank percentile, and standardized deviation.
* Coverage Growth: approximately twenty logarithmically spaced trial prefixes,
  including the final trial, with all five growth metrics.
* Random Coverage Growth: a separately started simulation with random mean and
  5–95% bands. Random trajectories use nested prefixes of full designs.

The two growth jobs are independent. The random-growth job can also be started
directly and includes the realised trajectory for comparison. Starting realised
growth does not start its random baseline.

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

Every recorded trial contributes, including failures and repeated configurations
at different seeds or budgets. Unique configuration counts are also shown.
Growth follows recorded history order, rather than sorting configurations or
deduplicating repeated trials.

Continuous marginal coverage is occupied-bin fraction; discrete marginal coverage
is visited-value fraction. Min/max range fraction is in original value coordinates,
while normalized gaps and edge gaps are in sampling coordinates. The largest raw
gap is between adjacent observed numerical values. Integer missing runs count
unvisited lattice values, including edges. Frequency tables for ranges above 1,000
values show visited values only; their coverage and unseen counts still use the
complete theoretical range. Continuous plots use at most 1,000 ECDF points, while
the statistics use all trials. The KS statistic is descriptive, without an IID
significance claim for adaptive or repeated trials.

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
may be unreachable under conditions or forbidden combinations, so occupancy is
not labelled good/bad. Discrete axes with more values than bins are grouped.
The approximate hole score is the grid distance to an occupied cell or grid edge
for numerical continuous pairs. Expected marginal frequencies describe the
defined parameter distribution; constraints can alter realised marginal weights.

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
``pairwise``, ``discrepancy``, ``radius``, ``baseline``, ``growth``, or
``growth_baseline``. The evaluator also exports separate calculation functions
for callers that want to reuse one normalized ``CoverageData`` snapshot.

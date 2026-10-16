Parameter Dependencies & Interactions
=====================================

This Configuration Analyses view separates the realized sampling design from observed
conditional performance. It supports Optuna studies and Optuna-only groups, uses the global
objective selector, and requires at least two parameters. Repeated configurations count once
per trial. For groups, results describe the pooled trials, including possible between-study
differences; they do not adjust for study membership.

Parameter Dependencies
----------------------

The symmetric matrix shows normalized mutual information (NMI) between every parameter pair:

.. math::

   \mathrm{NMI}(A,B) = \frac{2 I(A;B)}{H(A)+H(B)}.

The score ranges from zero to one and describes sampling dependency, with no performance or
causal interpretation. The diagonal, constant marginals, and pairs with fewer than two jointly
active trials are unavailable. Inactive or missing parameter values are excluded per pair;
the selected pair reports its excluded trial count. Sampling includes failed trials and trials
without a finite objective, so switching objective cannot change the dependency matrix.

Categorical and ordinal parameters retain their declared levels. Numeric parameters with few
observed values retain exact levels; otherwise adaptive quantile bins are fitted to all sampled
values. The same bins are used in the matrix, explanatory drilldown, and performance analysis.
Bin count is configurable. Finite-sample bias, cardinality, and binning affect NMI; it is an
empirical estimate, and zero does not prove independence.

Click a matrix cell to select the target and conditioning parameter. The joint-distribution
panel displays observed counts. The explanation panel displays relative deviations from
independence, using the same joint table as the MI estimate:

.. math::

   E_{ij} = \frac{n_{i\cdot} n_{\cdot j}}{N},
   \qquad \mathrm{deviation}_{ij} = \frac{O_{ij}-E_{ij}}{E_{ij}}.

Hover shows observed and expected counts. Cells with zero expectation are unavailable, while
unobserved combinations with positive expectation show -100%. This is an explanation of the
dependency score, not an additional dependency measure. Descriptive NMI bands are Low below
0.10, Medium below 0.30, and High otherwise; these are not significance thresholds.

Conditional Performance
-----------------------

Select a target parameter and click a conditioning parameter in its ranking. The matrix and
ranking use the same pair-selection controls. Conditional curves show the median objective per
joint bin among successful trials with finite objectives. The support heatmap shows their cell
counts. Cells below the configurable minimum support remain gaps, including in categorical and
mixed parameter pairs. Curves are observations, not fitted causal effects or uncertainty bounds.

An effect is comparable only if **every observed conditioning region** supports at least two
common target bins, at least 60% of the observed target bins, at least one marginally optimal
target bin, and an optimum from each supported regional curve. Unobserved declared categories
do not count as conditioning regions. Sparse observed
regions do count and cannot be silently discarded to claim robustness.

Performance interaction measures the RMS spread between centered conditional median profiles
over these common target bins, divided by the pooled objective standard deviation. Centering
removes differences consisting only of a conditioning-parameter offset. Its descriptive bands
are Low below 0.10, Medium below 0.30, and Strong otherwise. Rankings order estimable pairs by
this magnitude, with unestimable pairs last. There is no significance or causal-interaction claim.

Sweet-spot stability is High when the supported profiles share at least one optimal target bin
and Low when their optima disagree. Objective direction is respected. Differences within 5% of
the pooled objective standard deviation are treated as ties. Flat profiles have no identifiable
sweet spot. Assessments apply to the shared supported bins and the selected resolution.

Interpretability
----------------

* **Robust**: low sampling dependency, low interaction, and a stable supported optimum.
* **Interaction**: sufficient overlap, but conditional profiles or optima change. This remains an
  empirical hint even when sampling dependency is high; it does not establish causation.
* **Potentially confounded**: high sampling dependency and insufficient shared support.
* **Insufficient support**: conditional comparison is unavailable with lower or undefined dependency.
* **Stable within observed overlap**: supported effects are stable despite sampling dependency.
* **Inconclusive**: supported conditional profiles are flat, without an identifiable sweet spot.

High MI alone never establishes confounding. These are descriptive diagnostics without a combined
confounding score; other parameters, optimization history, and unmeasured factors can still matter.

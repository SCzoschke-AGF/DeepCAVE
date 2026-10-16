Importances
===========

The Importances plugin provides insights into how individual hyperparameters influence the
objective performance of your models.
It generates a bar plot displaying the importance scores for each hyperparameter.

This plugin is capable of answering following questions:

* Which individual hyperparameters have the greatest impact on the objective performance?
* Which selected pairs of hyperparameters have an additional interaction effect?
* How does the importance of each hyperparameter vary across different budgets?
* How does the local parameter importance compare to the global importance as determined by fANOVA?
* Which hyperparameters distinguish a selected top-performing region from the observed search?

.. image:: ../images/plugins/importances.png

Importance Methods
------------------

**Local Parameter Importance** (LPI) focuses on understanding the impact of individual
hyperparameters in the context of a specific configuration (here, the incumbent). Leveraging a
random forest surrogate model, it calculates the variance in performance caused by changes in each
hyperparameter, and expresses this as a fraction of the total variance.
This method is especially useful for pinpointing which parameters are crucial in the context of the
incumbent their local effects. For more details on LPI, please see the paper
`CAVE: Configuration Assessment, Visualization and Evaluation
<https://ml.informatik.uni-freiburg.de/wp-content/uploads/papers/18-LION12-CAVE.pdf>`_.

**fANOVA Importance** Functional ANOVA (fANOVA) evaluates hyperparameter importance by decomposing
the performance variance into contributions from individual hyperparameters and their interactions.
It uses a surrogate model, such as a random forest, to predict performance across different
hyperparameter configurations, and then analyzes how much each hyperparameter and its interactions
contribute to the overall variance in performance. This method provides a comprehensive view of
hyperparameter importance across the entire configuration space, thus helping in understanding the
global impact of hyperparameters. For more details on fANOVA, please refer to the paper
`An Efficient Approach for Assessing Hyperparameter Importance
<https://proceedings.mlr.press/v32/hutter14.html>`_.

**PED-ANOVA Importance** compares the hyperparameter distributions of the best trials with those
in the observed search region. The target quantile controls which top-performing trials form the
target region. For example, a target quantile of ``0.10`` analyzes the top 10% of successful
trials. DeepCAVE's implementation follows the Optuna 4.9 algorithm and Scott Parzen estimator while
operating directly on the common DeepCAVE run representation. PED-ANOVA currently supports only
single-objective analysis. For details, see the paper
`PED-ANOVA: Efficiently Quantifying Hyperparameter Importance in Arbitrary Subspaces
<https://arxiv.org/abs/2304.10255>`_.

For global single-objective fANOVA, the interaction order can be set to two. DeepCAVE keeps
all main effects and offers either screening of all parameter pairs or exact interactions
between two to four explicitly selected hyperparameters.
An interaction such as ``learning_rate × max_depth`` shows its individual
importance: the additional variance attributable to that pair after subtracting its main effects.
The total importance, which includes those main effects, remains available in the raw data.

Pairwise marginalization is substantially more expensive than calculating main effects. It is
performed by the background worker and can take several minutes for four selected parameters or
for runs with multiple budgets.

Interaction screening
---------------------

``Screen all pairs`` first evaluates every pair on an 8 × 8 grid in the forest's encoded
parameter space. It refines the leading pairs, pairs near the selection boundary, and pairs
with a large numerical offset on a 16 × 16 grid. A deterministic shifted-grid audit of some
lower-ranked pairs can also promote them to refinement. No main-effect threshold excludes
parameters, so pairs with small individual effects remain eligible.

The marginal predictions for the constant and individual terms are exact native forest
marginals. The squared pure pair interaction is integrated numerically and divided by each
tree's exact total variance before averaging over non-degenerate trees. Grid weights use the
same encoded bounds and categorical counting measure as the exact calculation, including
logarithmic parameter transformations.

The table separates approximate importance from exact importance. Offset is the mean of the
squared residual means divided by each tree's total variance. Resolution Δ compares the
8 × 8 and 16 × 16 scores; Shifted-grid Δ records the audit difference. ``Stable`` means the
resolution difference and offset meet the numerical tolerances. It is not statistical
confidence or a guarantee that narrow interactions were detected.
Once a finite exact result is available, the calculation status shows ``Exact`` instead of
the screening status, including ``Refine recommended``. Grid estimates and diagnostics
remain visible for comparison with the exact result.

``Exact top pairs`` controls how many of the ranked pairs are calculated exactly (default 5).
Set it to 0 for screening only. ``Additional exact pairs`` allows checking any other pairs;
click a pair in the table or select it in the dropdown, then click Process. Pairs calculated
exactly at every processed budget disappear from the picker. Their exact results remain in
the table and bar plot, including on subsequent calculations with the same forest.
The fitted forest, its exact marginal calculations, and grid
estimates are cached using the actual training data, parameter space, and forest settings.
Changing the pair selection reuses the original fitted trees. The bar plot contains main
effects and calculated exact interactions only; approximate scores remain in the table.

Multi-Objective Importance
--------------------------

Both LPI and fANOVA can be applied to two objectives displaying how the importance changes from one
objective to the other. Based on a weighting scheme, the two objectives are scalarized to form a
single objective for which the importance is calculated. The resulting plot then shows the
scalarized importance for different weightings. The weightings are calculated based on the points
on the pareto front which is the set of non-dominated solutions. For more detail on this, please
see the paper `Hyperparameter Importance Analysis for Multi-Objective AutoML
<https://arxiv.org/abs/2405.07640>`_.

.. image:: ../images/plugins/importances_mo.png


Options
-------

* **Objective 1 / 2**: Choose the objective you wish to calculate the importance for. Optionally, choose a second objective to see how the importance changes from one objective to the other.

* **Method:** Whether to calculate local parameter importance, fANOVA, or PED-ANOVA importance.

* **Trees:** Specify the number of trees for the random forest surrogate model used by LPI and
  fANOVA. This setting is disabled for PED-ANOVA.

* **Target quantile:** Select the top-performing fraction analyzed by PED-ANOVA. It must be greater
  than zero and smaller than one. This setting is disabled for other methods.

* **Interaction order:** Calculate main effects only (order 1), or main effects plus selected
  pairwise interactions (order 2). Order 2 is available only for global fANOVA with one objective.

* **Interaction hyperparameters:** Choose between two and four non-constant hyperparameters whose
  pairwise interactions should be calculated in Selected parameters mode.

* **Interaction analysis:** Screen all parameter pairs or calculate interactions among explicitly
  selected parameters. Shown only for global fANOVA with interaction order 2.

* **Exact top pairs:** Number of top-ranked screening pairs to calculate exactly; 0 screens only.

* **Additional exact pairs:** Other pairs to calculate exactly using the cached forest.

To refine your analysis, you can apply various filters after calculation:

* **Plot top N Importances**: Show only the top N main or interaction effects, considering all
  hyperparameters.

* **Budgets**: Filter the results to view importance scores specific to certain multi-fidelity budgets, allowing you to analyze how hyperparameter importance varies with budget changes.

.. warning::
    The fANOVA implementation relies on pyrfr, which cannot be applied to constant hyperparameters.
    As a workaround, we remove constant hyperparameters before calculation.
    This will break if the configuration space contains any conditions or forbiddens including
    constant hyperparameters.

.. note::
    PED-ANOVA is available only when a single objective is selected. Selecting PED-ANOVA clears and
    disables the second objective input.

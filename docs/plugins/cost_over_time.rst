Cost Over Time
==============

One of the goals of using AutoML is to achieve high peak performance. However, it is also important
to track how the objectives evolve over time. This plugin visualizes these changes based on time
or the number of trials evaluated.

With support for multiple Optuna studies, you can easily see which study performs best at various
points in time. The objective distributions of the selected studies are shown as overlaid
histograms next to their running-best trajectories.

.. note::
    The configuration spaces of the selected runs should be equal. Otherwise, a good comparison
    is not possible. They can, however, still be displayed in the same graph.

.. note::
    For non-deterministic runs (i.e. multiple seeds evaluated per configuration), only
    configurations evaluated on the maximum number of seeds are considered for choosing the best
    configuration at a given point in time.

This plugin is capable of answering the following questions:

* Does the optimizer converge?
* How does optimizer A compare to optimizer B? Is optimizer A more effective than optimizer B?
* How long does it take for the optimizer to reach a specific objective value?

.. image:: ../images/plugins/cost_over_time.png


Interpretation
--------------

The histogram on the left shows the objective values of all successful trials at the selected
budget. Histograms are overlaid when multiple studies are selected, and a dashed vertical line
marks each study's best value. The graph on the right shows the corresponding running-best
trajectory. By examining both plots, you can compare both the explored objective distributions and
the convergence behavior of the selected studies.


Options
-------

* **Optuna Studies:** Select one or more of the loaded studies. The first loaded study is selected
  initially; the plugin does not display every study by default.

* **Objective:** Select the objective function you wish to analyze.

* **Budget**: Select the multi-fidelity budget to be used. The plugin will only consider trials evaluated
  on the selected budget. The *Combined* budget option displays all configurations but shows scores only
  from the highest budget if a configuration was evaluated with multiple budgets.

To refine your analysis, you can apply various filters after calculation:

* **X-Axis:** You can adjust the x-axis to suit your needs using the available options.
  For long-running processes, a logarithmic scale is often more useful.

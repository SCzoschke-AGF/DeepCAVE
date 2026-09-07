1-D Analyses
============

The 1-D Analyses plugin examines how one hyperparameter relates to a selected objective in an
Optuna study. Study, objective, hyperparameter, and the proportion of trials that should be
considered elite are selected in the toolbar at the top of the page.

The page is one analysis view instead of three separate charts: **Observed trials** on the left is
the raw-data and coverage check, and the two panels on the right are the range analyses. Both use
the same ten bins, the same width, and the same axis, so they can be read as one block. The
navigation sidebar can be collapsed with the button next to the DeepCAVE logo to give the panels
the full window width.

Observed trials
---------------

Every successful trial, colored by the original Optuna trial number.

Performance by range
--------------------

Per bin, the median and the good tail of the objective distribution. For maximization objectives
the tail is the 90th percentile; for minimization objectives it is the 10th percentile.

Elite enrichment
----------------

Click **Apply** to display how strongly elite trials accumulate in each bin:

.. math::

   \mathrm{enrichment} = \frac{\text{elite share in the bin}}{\text{elite share of the study}}

A bin at ``1.5x`` therefore contains 1.5 times as many elite trials as the study average, and the
dashed baseline is always at ``1.0x`` regardless of the selected elite proportion. Bar colors
encode the median global rank of the elite trials in the bin, where rank one is best and is drawn
darkest; bins without elite trials are gray. Trial count, elite count, elite rate, enrichment, and
median elite rank are available in the tooltip.

Changing the study, objective, hyperparameter, or elite proportion resets the enrichment panel
until it is processed again. Ties at the elite threshold are included, so the resulting number of
elite trials can be slightly larger than the requested proportion; the enrichment is measured
against the elite share the study actually contains.

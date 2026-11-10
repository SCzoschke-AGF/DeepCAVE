2-D Analyses
============

The 2-D Analyses plugin examines the empirical relationship between two hyperparameters and a
selected objective in an Optuna study. Select a study, objective, two different hyperparameters,
and the proportion of plottable trials that should be considered elite.

The X and Y cards group each hyperparameter selection with its search range and bin count.
The Elite card sets the elite proportion and resets both feature ranges.

The upper-left heatmap shows approximate fANOVA interaction importances for all pairs of
non-constant features. Click **Calculate** to run the same heuristic screening used in Importances:
an 8 × 8 grid with 16 × 16 refinement of promising or numerically uncertain pairs and shifted-grid
checks. It uses 10 trees and the highest budget, reusing the shared fANOVA model cache. Scores show
the pure pair interaction after removing individual effects. Each pair appears once in the upper
triangle as a rounded tile with its score in percent, matching the hover; the diagonal and lower
triangle remain empty. A muted
gray–green–yellow scale runs from low to high interaction relative to the largest displayed score.
The currently selected feature pair has an outline. Hover
details show the approximate importance and screening status. Click a cell to select its column as
X and its row as Y for all three empirical plots. Changing the study or objective clears the
heatmap; calculate again for the new selection. Feature, range and bin changes do not trigger
screening. No exact pair integrations are performed for this heatmap.

Below the empirical plots on the left, **Search space coverage** shows a dedicated density heatmap
without a trial scatter. It follows the selected X and Y features, including selections made in the
fANOVA heatmap, and uses each feature's bin count independently. The **20 × 20 bins** button sets
both feature controls to the recommended 20 bins. While 20 × 20 is selected, the button shows the
previous X × Y bin counts and restores them when clicked. Discrete support smaller than the requested bin
count uses fewer cells. Cell colors indicate the number of successful trials and highlight empty
cells; the only summary metric is **Empty cells**, shown as a percentage and count. Coverage always
uses the full configured space, including unobserved values, independently of planned ranges.

To its right, **Cell performance · Best trial** uses the same coverage grid and colors each cell
by its best finite value of the selected objective, honoring minimization or maximization. Brighter
colors indicate better values. Beige cells have no finite objective value. This view has no trial
scatter or coverage summary metrics. Feature selections, bin changes and the 20 × 20 toggle apply
to both plots; changing the objective updates the performance colors.

The upper-right plot shows every successful trial without binning. Point colors encode the selected
objective value, and the color direction follows the optimization direction. A switch in the plot
header can instead color points by trial rank. The lower-left
heatmap groups both hyperparameters into joint bins and displays elite enrichment relative to the
selected global
elite proportion. One is the expected baseline, values below one are underrepresented, and values
above one are overrepresented. Cell labels retain the number of trials, while hover details also
show the raw elite rate and elite count. Unobserved cells remain neutral and distinct from observed
cells with zero enrichment. Favorable objective values and high enrichment use the bright end of
the shared Viridis color scale.

Empirical Objective Surface
---------------------------

The lower-right plot shows the median objective in each joint bin. For two numeric parameters the
median is rendered as a contour plot. Missing joint bins remain empty and no surrogate model or
value imputation is used.
If either parameter is categorical or ordinal, an objective heatmap is shown instead because
interpolation between categories would not be meaningful. The Viridis direction follows the
objective direction, so the bright end consistently represents better values.

The controls in the X and Y cards set the bin count independently for both selected features. Both the
elite-enrichment and objective-contour plots use this shared two-dimensional grid. Categorical
features continue to use their categories and therefore ignore the numeric bin count.

The search-range control in each card provides a reduced search-space planner. Numeric
features use range sliders, while categorical and ordinal features use multi-select controls. The
selection is shared with the HPO Planner and remains available while navigating between plugins in
the current browser page. Numeric extensions beyond the current ConfigSpace are shown as orange
striped slider regions but do not expand any analysis plot.

Gray plot overlays show the parts of the current space excluded by the planned range. Interval bins
are masked proportionally when a boundary crosses a bin; discrete values and excluded categories
are masked as complete rows or columns. These overlays are visual planning aids only: trials,
binning, elite enrichment, objective medians, colors, annotations, and hover values are not
filtered or recalculated.

Numeric axes use four equally wide bins by default, using logarithmic spacing for logarithmic
hyperparameters. Numeric parameters with no more than four observed values, categorical
parameters, and ordinal parameters are displayed discretely. Elite thresholds and rankings honor
the selected objective's optimization direction, and ties at the threshold are included.

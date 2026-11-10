All Runs Overview
=================

Use **Load all runs** to compare runs as table rows. This loads imported runs,
available run directories in the current working directory, and all studies from
the Optuna database selected in General. No active run or analysis worker is required.
Studies that cannot be loaded are reported without hiding the other results.

Open **Columns** and search the grouped list of run information, objectives, and
scalar numeric trial user attributes. Click an attribute to expand its statistics,
then check min, max, avg, median, std, or range. Multiple statistics can be selected;
changes appear immediately. Uncheck a statistic or use the column header's delete
button to remove it. Run information such as Successful Trials can be toggled directly
and does not require a statistic. Trial counts display as integers.
Run names remain visible in a compact column; hover to read the full name.
Click a database run's name to open its metadata page. It displays study user and
system attributes and expandable details for every trial, including failed and
running trials: parameters, distributions, attributes, timestamps, objective values,
and intermediate values. Metadata is read from the selected source database.
Click headers to sort by multiple columns and use the filter row for text or numeric
comparisons, such as ``> 0.9``.

Only successful trials contribute to counts, durations, and statistics. All budgets
are included without a budget selector. The sidebar's sampling-method filter does
not restrict this overview. Missing and non-finite metric observations are omitted;
unavailable results remain empty. Boolean and structured user attributes are excluded.
User attributes need not be present on every successful trial to appear in this table.
Std is the population standard deviation (zero for a single observation), and range
is max minus min. Total Trial Duration sums successful trial durations in seconds.

Database studies are read directly without creating or refreshing import snapshots.
Conditional and changing search spaces are supported here because this overview does
not convert hyperparameter configurations to ConfigSpace. Studies with no successful
trials remain visible with a count of zero and empty statistics. Sorting, filtering,
and column changes reuse the loaded summary; press **Load all runs** again to refresh.

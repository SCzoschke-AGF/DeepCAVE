# Randomized space partitioning: sensitivity measurements

80 trials; 200 fixed partitions per K; K=[20, 40, 80]; seed 20261017. Independent pools: 8192 for partition construction, 32768 for cell masses, 128 IID designs for standardization, 199 for calibration, 100 held-out designs per scenario.

Entries are upper-tail flag rates above the 95th midrank percentile of IID designs. `random` rows are held-out false-alarm rates. These are measurements at a nominal 5% threshold, conditional on fixed partitions, mass estimates, standardization, and calibration pools; not guaranteed exact 5% rates. JSON includes 95% Wilson intervals and two-sided rates. No statistic is selected using the held-out defect labels.

Mean averages unvisited probability mass across partitions. Max selects the largest per-partition standardized unvisited mass. Joint takes the largest standardized aggregate among mean and max of unvisited mass and TV at all three K values. The complete selection is calibrated against the same IID designs. All component statistics including TV are retained in JSON.

MMD columns reuse the earlier results only with matching evaluation and calibration seeds, counts and scenarios. Reference budgets differ: MMD used 512 points; partitions use the larger construction and mass pools above. This is a practical sensitivity comparison, not a comparison at equal computation or reference budget.

| Space | Scenario | Mean K=20 | Mean K=40 | Mean K=80 | Max K=20 | Max K=40 | Max K=80 | Joint | MMD product | MMD mixture |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mixed_6d | random | 6% [3, 12] | 4% [2, 10] | 5% [2, 11] | 3% [1, 8] | 7% [3, 14] | 1% [0, 5] | 3% [1, 8] | 6% [3, 12] | 8% [4, 15] |
| mixed_6d | missing_category | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] |
| mixed_6d | marginal_bias | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] |
| mixed_6d | missing_combinations | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 98% [93, 99] |
| mixed_12d | random | 4% [2, 10] | 7% [3, 14] | 4% [2, 10] | 4% [2, 10] | 6% [3, 12] | 5% [2, 11] | 5% [2, 11] | 6% [3, 12] | 8% [4, 15] |
| mixed_12d | missing_category | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 44% [35, 54] | 100% [96, 100] |
| mixed_12d | marginal_bias | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 17% [11, 26] | 100% [96, 100] |
| mixed_12d | missing_combinations | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 42% [33, 52] | 35% [26, 45] |
| mixed_20d | random | 1% [0, 5] | 10% [6, 17] | 10% [6, 17] | 5% [2, 11] | 5% [2, 11] | 5% [2, 11] | 5% [2, 11] | 7% [3, 14] | 2% [1, 7] |
| mixed_20d | missing_category | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 21% [14, 30] | 100% [96, 100] |
| mixed_20d | marginal_bias | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 11% [6, 19] | 100% [96, 100] |
| mixed_20d | missing_combinations | 100% [96, 100] | 83% [74, 89] | 93% [86, 97] | 100% [96, 100] | 100% [96, 100] | 95% [89, 98] | 100% [96, 100] | 16% [10, 24] | 9% [5, 16] |
| categorical_20d | random | 6% [3, 12] | 3% [1, 8] | 2% [1, 7] | 3% [1, 8] | 4% [2, 10] | 6% [3, 12] | 4% [2, 10] | 0% [0, 4] | 3% [1, 8] |
| categorical_20d | missing_category | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 0% [0, 4] | 100% [96, 100] |
| categorical_20d | missing_combinations | 100% [96, 100] | 87% [79, 92] | 59% [49, 68] | 88% [80, 93] | 100% [96, 100] | 81% [72, 87] | 100% [96, 100] | 0% [0, 4] | 7% [3, 14] |
| categorical_20d | three_way_parity | 9% [5, 16] | 11% [6, 19] | 9% [5, 16] | 4% [2, 10] | 9% [5, 16] | 13% [8, 21] | 9% [5, 16] | 0% [0, 4] | 3% [1, 8] |
| conditional_6d | random | 3% [1, 8] | 4% [2, 10] | 8% [4, 15] | 4% [2, 10] | 3% [1, 8] | 9% [5, 16] | 2% [1, 7] | 7% [3, 14] | 2% [1, 7] |
| conditional_6d | conditional_bias | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 93% [86, 97] | 100% [96, 100] |
| conditional_6d | missing_branch | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] |
| rare_category_6d | random | 3% [1, 8] | 2% [1, 7] | 1% [0, 5] | 8% [4, 15] | 4% [2, 10] | 1% [0, 5] | 3% [1, 8] | 2% [1, 7] | 2% [1, 7] |
| rare_category_6d | missing_rare_category | 5% [2, 11] | 2% [1, 7] | 1% [0, 5] | 9% [5, 16] | 2% [1, 7] | 2% [1, 7] | 3% [1, 8] | 2% [1, 7] | 3% [1, 8] |

## Cell balance and random occupancy

Mass is estimated on a separate reference pool, not assumed to be 1/K. The occupancy expectations use these estimated probabilities. IQR describes the IID distribution of the average unvisited mass, not a confidence interval.

| Space | K | Actual cells | Zero estimated mass cells | Median mass CV | Expected missing mass | Random median [IQR] |
|---|---:|---|---:|---:|---:|---|
| mixed_6d | 20 | 20–20 | 0 | 0.16 | 0.018 | 0.017 [0.014, 0.023] |
| mixed_6d | 40 | 40–40 | 0 | 0.16 | 0.132 | 0.132 [0.123, 0.141] |
| mixed_6d | 80 | 80–80 | 0 | 0.16 | 0.361 | 0.362 [0.353, 0.370] |
| mixed_12d | 20 | 20–20 | 0 | 0.11 | 0.018 | 0.017 [0.014, 0.020] |
| mixed_12d | 40 | 40–40 | 0 | 0.13 | 0.132 | 0.132 [0.125, 0.136] |
| mixed_12d | 80 | 80–80 | 0 | 0.15 | 0.362 | 0.361 [0.356, 0.366] |
| mixed_20d | 20 | 20–20 | 0 | 0.06 | 0.017 | 0.017 [0.015, 0.019] |
| mixed_20d | 40 | 40–40 | 0 | 0.12 | 0.132 | 0.132 [0.129, 0.136] |
| mixed_20d | 80 | 80–80 | 0 | 0.14 | 0.362 | 0.362 [0.360, 0.366] |
| categorical_20d | 20 | 20–20 | 0 | 0.31 | 0.024 | 0.024 [0.021, 0.026] |
| categorical_20d | 40 | 40–40 | 0 | 0.31 | 0.130 | 0.129 [0.124, 0.135] |
| categorical_20d | 80 | 80–80 | 0 | 0.31 | 0.346 | 0.345 [0.341, 0.351] |
| conditional_6d | 20 | 20–20 | 0 | 0.25 | 0.021 | 0.020 [0.015, 0.027] |
| conditional_6d | 40 | 40–40 | 1 | 0.29 | 0.130 | 0.129 [0.118, 0.142] |
| conditional_6d | 80 | 80–80 | 13 | 0.33 | 0.340 | 0.341 [0.331, 0.351] |
| rare_category_6d | 20 | 20–20 | 0 | 0.25 | 0.019 | 0.018 [0.015, 0.023] |
| rare_category_6d | 40 | 40–40 | 0 | 0.33 | 0.122 | 0.122 [0.113, 0.132] |
| rare_category_6d | 80 | 80–80 | 0 | 0.40 | 0.330 | 0.330 [0.322, 0.339] |

## Fixed-partition forecasts

Independent IID future configurations are appended to one observed design; each simulation uses the same future configurations across all partitions. Partitions and K remain fixed. The target is a 20% reduction in average unvisited mass. This is not the union of spatial holes across overlapping partitions. The budgets below are the first tested grid points whose 95% Wilson lower bound is at least 80%; they are not exact minimum trial counts. Intervals condition on the observed design, partitions and estimated masses. Forecasts assume future IID sampling from ConfigSpace, not an adaptive optimizer.

| Observed design | K | Initial missing mass | First tested sufficient budget |
|---|---:|---:|---:|
| random | 20 | 0.016 | 10 |
| random | 40 | 0.130 | 20 |
| random | 80 | 0.363 | 20 |
| missing_category | 20 | 0.107 | 10 |
| missing_category | 40 | 0.207 | 20 |
| missing_category | 80 | 0.422 | 20 |

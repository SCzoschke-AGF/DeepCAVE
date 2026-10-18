# Mixed-space MMD sensitivity measurements

Seed 20261017; 199 independent calibration designs, 100 held-out replicates per scenario; 512 shared reference configurations; kernel width 0.2.

The table measures the fraction with MMD above the 95th midrank percentile of correct IID designs. This is an upper-tail detection rate, not the dashboard's two-sided 5–95% status. Both rates and their 95% Wilson intervals are in results.json. Intervals are conditional on the fixed reference and calibration designs. Null evaluation seeds are disjoint from calibration and reference seeds.

`product` is the current production kernel. `marginal` is a negative control for joint-only defects. `interaction_mixture` is an experimental equal-weight mixture of averaged one-dimensional kernels, averaged pair products, and the complete product. No comparator is selected using evaluation labels.

| Space | Trials | Scenario | Product | Marginal control | Interaction mixture |
|---|---:|---|---:|---:|---:|
| mixed_6d | 80 | random | 6% [3, 12] | 9% [5, 16] | 8% [4, 15] |
| mixed_6d | 80 | missing_category | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] |
| mixed_6d | 80 | marginal_bias | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] |
| mixed_6d | 80 | missing_combinations | 100% [96, 100] | 9% [5, 16] | 98% [93, 99] |
| mixed_6d | 256 | random | 5% [2, 11] | 5% [2, 11] | 5% [2, 11] |
| mixed_6d | 256 | missing_category | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] |
| mixed_6d | 256 | marginal_bias | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] |
| mixed_6d | 256 | missing_combinations | 100% [96, 100] | 6% [3, 12] | 100% [96, 100] |
| mixed_12d | 80 | random | 6% [3, 12] | 7% [3, 14] | 8% [4, 15] |
| mixed_12d | 80 | missing_category | 44% [35, 54] | 100% [96, 100] | 100% [96, 100] |
| mixed_12d | 80 | marginal_bias | 17% [11, 26] | 100% [96, 100] | 100% [96, 100] |
| mixed_12d | 80 | missing_combinations | 42% [33, 52] | 10% [6, 17] | 35% [26, 45] |
| mixed_12d | 256 | random | 7% [3, 14] | 5% [2, 11] | 5% [2, 11] |
| mixed_12d | 256 | missing_category | 97% [92, 99] | 100% [96, 100] | 100% [96, 100] |
| mixed_12d | 256 | marginal_bias | 52% [42, 62] | 100% [96, 100] | 100% [96, 100] |
| mixed_12d | 256 | missing_combinations | 99% [95, 100] | 3% [1, 8] | 72% [63, 80] |
| mixed_20d | 80 | random | 7% [3, 14] | 2% [1, 7] | 2% [1, 7] |
| mixed_20d | 80 | missing_category | 21% [14, 30] | 100% [96, 100] | 100% [96, 100] |
| mixed_20d | 80 | marginal_bias | 11% [6, 19] | 100% [96, 100] | 100% [96, 100] |
| mixed_20d | 80 | missing_combinations | 16% [10, 24] | 5% [2, 11] | 9% [5, 16] |
| mixed_20d | 256 | random | 2% [1, 7] | 2% [1, 7] | 2% [1, 7] |
| mixed_20d | 256 | missing_category | 8% [4, 15] | 100% [96, 100] | 100% [96, 100] |
| mixed_20d | 256 | marginal_bias | 5% [2, 11] | 100% [96, 100] | 100% [96, 100] |
| mixed_20d | 256 | missing_combinations | 9% [5, 16] | 1% [0, 5] | 14% [9, 22] |
| categorical_20d | 80 | random | 0% [0, 4] | 4% [2, 10] | 3% [1, 8] |
| categorical_20d | 80 | missing_category | 0% [0, 4] | 100% [96, 100] | 100% [96, 100] |
| categorical_20d | 80 | missing_combinations | 0% [0, 4] | 3% [1, 8] | 7% [3, 14] |
| categorical_20d | 80 | three_way_parity | 0% [0, 4] | 3% [1, 8] | 3% [1, 8] |
| categorical_20d | 256 | random | 5% [2, 11] | 5% [2, 11] | 5% [2, 11] |
| categorical_20d | 256 | missing_category | 7% [3, 14] | 100% [96, 100] | 100% [96, 100] |
| categorical_20d | 256 | missing_combinations | 9% [5, 16] | 7% [3, 14] | 21% [14, 30] |
| categorical_20d | 256 | three_way_parity | 4% [2, 10] | 4% [2, 10] | 5% [2, 11] |
| conditional_6d | 80 | random | 7% [3, 14] | 4% [2, 10] | 2% [1, 7] |
| conditional_6d | 80 | conditional_bias | 93% [86, 97] | 100% [96, 100] | 100% [96, 100] |
| conditional_6d | 80 | missing_branch | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] |
| conditional_6d | 256 | random | 9% [5, 16] | 4% [2, 10] | 4% [2, 10] |
| conditional_6d | 256 | conditional_bias | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] |
| conditional_6d | 256 | missing_branch | 100% [96, 100] | 100% [96, 100] | 100% [96, 100] |
| rare_category_6d | 80 | random | 2% [1, 7] | 2% [1, 7] | 2% [1, 7] |
| rare_category_6d | 80 | missing_rare_category | 2% [1, 7] | 2% [1, 7] | 3% [1, 8] |
| rare_category_6d | 256 | random | 5% [2, 11] | 3% [1, 8] | 5% [2, 11] |
| rare_category_6d | 256 | missing_rare_category | 6% [3, 12] | 3% [1, 8] | 5% [2, 11] |

A missing 2% category naturally occurs with probability 0.98^n under correct IID sampling. At n=80 this is about 20%; absence alone is not a reliable anomaly at a 5% false-alarm budget.

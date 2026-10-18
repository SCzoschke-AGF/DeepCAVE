API
^^^

If you want to interact with the API directly, you can make use of the API mode.
Have a look into the examples to get an overview how to use DeepCAVE in this way.

``validate_coverage_mmd.py`` provides a reproducible sensitivity study of the
candidate mixed-space MMD. Run it as a module from the repository root; its
arguments control sample sizes, dimensions, calibration designs, evaluation
replicates, and artifact output. It does not modify the dashboard kernel.

``validate_coverage_partitions.py`` compares experimental randomized partitions
with that MMD study on the same synthetic defects. It tests three cell counts,
calibrates selection across partitions and statistics, and simulates additional
IID trials with the existing partitions held fixed. The partition implementation
is in ``coverage_partitions.py``. Neither script changes dashboard metrics.

Optuna
------

Import from SQLite
^^^^^^^^^^^^^^^^^^

The General page can import studies directly from a local Optuna SQLite database. Enter the path
under ``SQLite database``. Relative paths are resolved against DeepCAVE's current Working
Directory, so ``optuna.db`` selects a database with that name in the displayed directory.

DeepCAVE lists the studies stored in the database. Select a study and click ``Load Optuna Study``.
The imported run is selected immediately and stored below ``deepcave_optuna/`` in the Working
Directory. The generated Pickle contains an in-memory copy of the study and no longer needs the
SQLite database when it is analyzed.

Click ``Refresh from Optuna`` to update an existing snapshot. A refresh is blocked while DeepCAVE
has pending or running analysis jobs for that run. Running and waiting Optuna trials are retained
in the snapshot but skipped during conversion. They become available after they reach a terminal
state and the snapshot is refreshed again. Automatic refresh is not supported.

Only local SQLite file paths are accepted by this importer. Other Optuna storage backends can
still be exported manually as described below.

Manual Pickle export
^^^^^^^^^^^^^^^^^^^^

To be able to load your Optuna run into DeepCave, there are a few points to
consider when running Optuna:

1. Save the Optuna ``study`` object as a Pickle file:

   .. code-block:: python

        import pickle
        with open("<path-to-save-study>/optuna_study.pkl", "wb") as f:
            pickle.dump(study, f)

   Replace ``<path-to-save-study>`` with the directory where you want to store the Pickle file.

.. note::
   There has to be exactly one Pickle file.

2. Optional: If you want to display the names of your objectives in DeepCAVE, you can set metric names in Optuna using the following command:

   .. code-block:: python

        study.set_metric_names(["Accuracy", "Flops"])

   This step is optional but recommended for better visualization and understanding of your study's objectives in DeepCAVE.

Furthermore, to load an Optuna run into DeepCAVE, it is necessary to install Optuna, e.g. via:

.. code-block:: bash

    pip install deepcave[optuna]

.. note::
   Optuna does not support instances.
   The hyperparameters in the Optuna study must be of type FloatDistribution, IntDistribution or CategoricalDistribution.
   Loading Optuna runs with conditional search spaces or dynamic hyperparameter value ranges
   is not supported.
   This limitation arises because Optuna leverages a dynamic search space, which cannot be
   trivially converted into a static search space as used in DeepCAVE.

.. warning::
   Pickling a Study connected to persistent Optuna storage does not necessarily create a detached
   snapshot. Prefer the SQLite importer above, which explicitly copies the Study into in-memory
   storage before serializing it.

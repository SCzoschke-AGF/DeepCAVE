Getting Started
===============

After you have followed the installation steps, you can start DeepCAVE via following command
(in any directory):

.. code:: bash

    deepcave --open

Adding `\-\-open` will automatically redirect you to the interactive dashboard.
In the dashboard, you will see the main page, in which you
can select the runs you want to analyze. Please have a look at the following image
to see the dashboard in action:


.. image:: images/plugins/general.png


Displaying the data is done via plugins. You can select the plugins you want to use in
the menu on the left.
This is further explained in the :ref:`plugins<Displaying Data with Plugins>` section.

DeepCAVE uses :ref:`runs<Runs and Converters>` to interpret HPO optimization runs.
You can add a run by navigating to the directory of the run and clicking on the ``+`` button on
the left of it.
After loading runs, you will be able to select them in the different plugins.
On the main page, you can also add them to groups to create and analyze combined runs.

DeepCAVE comes with some pre-evaluated runs to get a feeling for what DeepCAVE can do.

If you cloned the repository from GitHub via ``git clone https://github.com/automl/DeepCAVE.git``,
you can try out some examples by exploring the ``logs`` directory inside the DeepCAVE dashboard.
For example, if you navigate to ``logs/DeepCAVE``, you can view the run ``mnist_pytorch`` if you hit
the ``+`` button left to it.

Filter Optuna trials by sampling method
--------------------------------------

Under **General → Optuna metadata (connect your user attributes)**, map **Sampler**
to your trial attribute, for example **Trial: sampling_method**. The sidebar's
**Sampling methods** field then offers the distinct values found in loaded studies.
Select one or several methods to restrict the trials used by analyses. For example,
select only Bayesian optimization, or select random/grid methods to exclude it.
Clear the selection to include all trials again.

Trials without the mapped attribute are available as **Missing metadata**. They are
included when no filter is selected and otherwise only when that option is selected.
Study-level sampler mappings are also supported and select whole source studies.
The subset applies to individual studies, groups, Sensitivity Analysis and the candidate
explorer. Existing collected candidates remain in the session collection. Analyses with
an Apply/Process button still use that button to compute their results.

The filter is persisted with the other application settings. Changing the Sampler
mapping clears the method selection. Filtered views retain original trial/configuration
identities and leave imported studies unchanged; cached results are kept separate.

More Options
------------

In addition to `\-\-open`, there are two other options that can be passed when starting DeepCAVE:

.. code:: bash

    deepcave --open --n_workers=1 --config="path/to/your/local.py"


* **\-\-open**: Automatically opens the DeepCAVE GUI in the browser after the DeepCAVE server has been
  started.
* **\-\-n_workers**: Number of workers to use for queueing :ref:`static plugin <plugin_types>` jobs.
* **\-\-config**: Path to a configuration file in case you want to use a custom configuration,
  e.g. to change IPs, ports, or refresh rate. The argument accepts both relative and absolute paths.
  However, make sure that the path is enclosed in quotes.


.. warning::
    The configuration file must include a ``Config`` class, which inherits from
    :ref:`Config<Config>`. Also, you should avoid to name the file ``config.py`` as this is
    reserved.

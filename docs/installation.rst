Installation
============

DeepCAVE needs a reachable Redis server for the interactive mode. For a source checkout, the
included Docker Compose service is recommended:

.. code:: bash

    docker compose up -d redis

It exposes Redis only on ``127.0.0.1:6379`` and disables persistence to prevent stale queue jobs.
A system-wide ``redis-server`` installation is also supported. See :doc:`redis` for both options.

The following commands install DeepCAVE. We recommend using anaconda, as this way `swig` can be
installed directly. If you use a different environment, make sure that
`swig <https://www.swig.org/index.html>`_ is installed.

.. code:: bash

    conda create -n DeepCAVE python=3.9
    conda activate DeepCAVE
    conda install -c anaconda swig
    pip install DeepCAVE

To load runs created with Optuna or the BOHB optimizer, you need to install the
respective packages by running:

.. code:: bash

    pip install deepcave[optuna]
    pip install deepcave[bohb]

To try the examples for recording your results in DeepCAVE format, run this after installing:

.. code:: bash

    pip install deepcave[examples]

If you want to contribute to DeepCAVE, you can clone it from GitHub and install the dev package:

.. code:: bash

    git clone https://github.com/automl/DeepCAVE.git
    conda create -n DeepCAVE python=3.9
    conda activate DeepCAVE
    conda install -c anaconda swig
    make install-dev


.. warning::

    DeepCAVE is officially tested and supported on Linux platforms.

    While it is generally expected to function correctly on MacOS, some issues may arise due to
    compatibility with Swig. Specifically, users may encounter problems with the
    Parallel Coordinates and Importance Plugin on MacOS.
    NumPy currently needs to be installed with conda (as opposed to pip) to ensure it being linked to openblas
    and not the Mac specific accelerate. The latter, if used, will result in a plugin failure for
    Configuration Footprint.

    Currently, DeepCAVE cannot be run on Windows due to its reliance on a bash script for
    starting services such as Redis, workers, and the webserver.


Mac Related
^^^^^^^^^^^
If you want to run DeepCAVE on a M1 Mac, you need to add

.. code:: bash

    export DISABLE_SPRING=true
    export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES


to your ``~/.bash_profile`` to enable multi-processing.


Redis Server
^^^^^^^^^^^^

See :doc:`redis` for Docker and system-wide installation instructions.

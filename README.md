<img src="docs/images/DeepCAVE_Logo_wide.png" alt="Logo"/> 

# ATTENTION! SC DeepCave

Die Kernfunktionen (original und erweitert) funktionieren mit dem nativen Optuna Output der out-of-the-box in optuna.dbs abgelegt wird. Sie verwenden nur den Input der über den default optuna namespace geliefert wird. Damit können sie auf allen aktuellen und vergangenen optuna studies (optuna.db Inhalte) seit Optuna Version x.xx.xx problemlos genutzt werden.

Darüber hinaus gibt es Zusatzfunktionen, die ein spezifisches Präparieren des Optuna-Loggings erfordern und entsprechend nur dann angewählt werden können, wenn die Voraussetzungen erfüllt sind. 

# DeepCAVE

DeepCAVE is a visualization and analysis tool for AutoML, with a particular focus on
hyperparameter optimization (HPO). Built on the Dash framework, it offers a fully
interactive experience. The tool features a variety of plugins that enable efficient insight
generation, aiding in understanding and debugging the application of HPO.
Additionally, the powerful run interface and the modularized plugin structure allow extending the 
tool at any time effortlessly.

![Configuration Footprint](docs/images/plugins/configuration_footprint.png)


## Installation

DeepCAVE requires a reachable Redis server for its background job queue. For a source checkout,
the included Docker Compose service is the recommended option:

```bash
docker compose up -d redis
```

The container is published only on `127.0.0.1:6379` and persistence is disabled so stale queue
jobs do not survive restarts. A system-wide `redis-server` installation remains supported as a
fallback. DeepCAVE checks the configured Redis connection at startup, so neither `redis-server`
nor `redis-cli` needs to be installed on the host when Docker is used.

Afterwards, follow the instructions to install DeepCAVE:
```bash
conda create -n DeepCAVE python=3.9
conda activate DeepCAVE
conda install -c anaconda swig
pip install DeepCAVE
```

To load runs created with Optuna, the BOHB optimizer or RayTune, you need to install the
respective packages by running:
```bash
pip install deepcave[optuna]
pip install deepcave[bohb]
pip install deepcave[raytune]
```

To try the examples for recording your results in DeepCAVE format, run this after installing:
```bash
pip install deepcave[examples]
```

If you want to contribute to DeepCAVE, use the following steps instead:
```bash
git clone https://github.com/automl/DeepCAVE.git
cd DeepCAVE
conda create -n DeepCAVE python=3.9
conda activate DeepCAVE
conda install -c anaconda swig
make install-dev
```

Please visit the [documentation](https://automl.github.io/DeepCAVE/main/installation.html) to get
further help (e.g. if you cannot install redis server or if you are on MacOS).


## Visualizing and Evaluating

The webserver as well as the queue/workers can be started by simply running:
```bash
deepcave --open
```

If you specify `--open` your webbrowser automatically opens at `http://127.0.0.1:8050/`.
You can find more arguments and information (like using custom configurations) in the
[documentation](https://automl.github.io/DeepCAVE/main/getting_started.html).

### Importing an Optuna SQLite study

On the **General** page, enter a local SQLite database path in **SQLite database**. Relative paths
are resolved against DeepCAVE's current Working Directory. DeepCAVE discovers the studies in the
database, lets you select one, and creates a self-contained snapshot when you click **Load Optuna
Study**.

Snapshots are stored below `deepcave_optuna/` in the Working Directory and selected immediately.
Use **Refresh from Optuna** to replace a snapshot manually. Running and waiting trials remain in
the snapshot but are skipped by the analyses until a later refresh observes them in a terminal
state. Automatic refresh is intentionally not part of this first version.


## Example runs

DeepCAVE comes with some pre-evaluated runs to get a feeling for what DeepCAVE can do.

If you cloned the repository from GitHub via `git clone https://github.com/automl/DeepCAVE.git`,
you can try out some examples by exploring the `logs` directory inside the DeepCAVE dashboard.
For example, if you navigate to `logs/DeepCAVE/mnist_pytorch`, you can view its runs if you hit
the `+` button left to them.


## Features

### Interactive Interface
- **Interactive Dashboard:**  
  The dashboard runs in a webbrowser and allows you to self-analyze your optimization runs interactively.
  
- **Run Selection Interface:**  
  Easily select runs from your working directory directly within the interface.
  
- **Integrated Help and Documentation:**  
  Use help buttons and integrated documentation within the interface to better understand the plugins.

### Comprehensive Analysis Tools
- **Extensive Plugin Collection:**  
  Explore a wide range of plugins for in-depth performance, hyperparameter, and budget analysis.

- **Analysis of Running Processes:**  
  Analyze and monitor optimization processes as they occur, with automatic detection of run changes.
  
- **Group Analysis:**  
  Choose groups of runs for combined analysis to gain deeper insights.

### Flexible and Modular Architecture
- **Modular Plugin Architecture:**  
  Benefit from a modularized plugin structure with access to selected runs and groups, offering you maximum flexibility.
  
- **Asynchronous Execution:**  
  Utilize asynchronous execution of resource-intensive plugins and caching of results to improve performance.

### Broad Optimizer Support
- **Optimizer Support:**  
  Work with many frameworks and optimizers using our converters, including converters for SMAC, BOHB, AMLTK, and Optuna.
  
- **Native Format Saving:**  
  Save AutoML runs from various frameworks in DeepCAVE's native format using the built-in recorder.
  
- **Flexible Data Loading:**  
  Alternatively, load AutoML runs from other frameworks by converting them into a Pandas DataFrame.

### Developer and API Features
- **API Mode:**  
  Interact with the code directly through API mode, allowing you to bypass the graphical interface if preferred.


## Citation

If you use DeepCAVE in one of your research projects, please cite our [ReALML@ICML'22 workshop paper](https://arxiv.org/abs/2206.03493):
```
@misc{sass-realml2022,
    title = {DeepCAVE: An Interactive Analysis Tool for Automated Machine Learning},
    author = {Sass, René and Bergman, Eddie and Biedenkapp, André and Hutter, Frank and Lindauer, Marius},
    doi = {10.48550/ARXIV.2206.03493},
    url = {https://arxiv.org/abs/2206.03493},
    publisher = {arXiv},
    year = {2022},
    copyright = {arXiv.org perpetual, non-exclusive license}
}
```

Copyright (C) 2021-2024 The DeepCAVE Authors

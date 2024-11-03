# Detection and Enforcement of Non-Linear Correlations for Fair and Robust Machine Learning Applications

This repository contains the code to reproduce the experiments of my PhD Dissertation "_Detection and Enforcement of Non-Linear Correlations for Fair and Robust Machine Learning Applications_".

## Installation Setup

There are two possible ways to run the scripts in this repository.

#### Local Execution

The first is via a local Python interpreter available in the machine.
Our code was tested with `Python 3.12`, so we suggest to stick to this version if possible.
We also recommend the use of a virtual environment in order to avoid package conflicts due to specific versions.
At this point, you will need to install the requirements via:
```
pip install -r requirements.txt
```
and eventually run a script using the command:
```
python <script_name>.py
```
optionally passing the specific parameters.

#### Container Execution

The second way is by leveraging Docker containers.
Each python script is associated to a Docker service that can be invoked and executed in background using the command:
```
docker compose run <script_name>
```
If you do not specify the name of the script, _all_ the scripts will be launched together in parallel, so we do not recommend this option.
To interact with the repository using a terminal in order to, e.g., launch scripts with custom parameters, it is possible to use the command:
```
docker compose run main -it
```
which will simply open the container in the working directory without launching any Python script.

#### _Note on Gurobi_
> Three scripts (_degrees_, _projections_, and _gedi_) require `Gurobi 11` to be installed as a standalone software.
> Gurobi is a commercial product, but free academic licenses can be requested on their [website](https://www.gurobi.com/free-trial/).
> However, the software must be installed manually either in the host machine in case of local execution, or in the Docker container in case of container execution.

## Scripts Description

Each script contains the code to generate specific figures and tables of the dissertation.

When running a script, this will automatically store the serialized results in a new folder within the project named `results`, so that they are not recomputed in case of a new execution.
You can retrieve a `json` (or `csv`) file with all the executed experiments along with their keys and configurations by calling the `inspection.py` script.
It is also possible to call the `clear.py` script in order to remove certain or all the obtained results, calling the specific parameters described in the script documentation.
Output figures and tables are also stored in the `results` folder.
If you would like to change the position of the folder you can specify the `-f <folder_path>` parameter when running a script locally, or exporting the environment variable `$FOLDER = <folder_path>` when using containers.

Following, we provide a description of each script and link them with the Figures and Tables referenced throughout the dissertation.

#### Calibration

This script trains different unconstrained neural networks to calibrate the optimal hyperparameters for each benchmark dataset.
Results are shown in _Figure A.1_, and require a few hours to be executed.

#### Causality

This script uses HGR to infer the causal link between the two inspected variables.
Results are shown in _Figure 5.2_, and are very fast to obtain.

#### Copulas

This script inspects the copula transformations returned by different HGR implementations.
Results are shown in _Figure 3.9_, and are very fast to obtain.

#### Correlations

This script tests multiple HGR implementations on several synthetic datasets under various levels of noise.
Results are shown in _Figure 3.7_ and _Figure 3.8_, and require a few hours to be executed.

> Note: if no parameters is passed, the script will generate _Figure 3.7_.
> In order to generate _Figure 3.8_, it is necessary to set the `--test` flag when calling the script in local execution, or to run the service `correlations_test` if using container execution. 

#### Degrees

This script shows how constraining the GeDI value up to higher degrees results in different functional relationships being captured.
Results are shown in _Figure 4.4_, and are very fast to obtain.

> Note: this script requires `Gurobi 11` to be manually installed in the local machine or in the Docker container.

#### Determinism

This script compares different HGR implementations to show their robustness to algorithm stochasticity.
Results are shown in _Figure 3.6_, and require a few minutes to be executed.

#### Example

This script shows an example of kernel computation using our Kernel-Based HGR implementation, and how this brings more benefits with respect to other methods.
Results are shown in _Figure 3.4_, and are very fast to obtain.

#### GeDI

This script trains different constrained machine learning models, imposing fairness constraints through GeDI in its different definitions.
Results are shown in _Figure 4.6_ and _Table 4.3_, and require a few hours to be executed.

> Note: this script requires `Gurobi 11` to be manually installed in the local machine or in the Docker container.

#### HGR

This script trains different constrained neural networks, imposing fairness constraints through different HGR implementations.
Results are shown in _Figure 4.2_ and _Table 4.2_, and require a few hours to be executed.

#### Importance

This script performs feature importance analysis on the three benchmark datasets using HGR in order to retrieve the continuous protected attributes and the protected surrogates.
Results are shown in _Figure 4.1_, and require a few minutes to be executed.

#### Limitations

This script generates two figures used to explain the limitations of HGR in fairness domains.
Results are shown in _Figure 4.3_, and are very fast to obtain.

#### LSTSQ

This script compares the time required to compute HGR-SK using global optimization versus least-square problem.
Results are shown in _Figure 3.3_, and require a few minutes to be executed.

#### Monotonicity

This script validates the monotonic property of HGR-KB by showing that the correlation increases monotonically with respect to degrees $h$ and $k$ on the three benchmark datasets.
Results are shown in _Figure 3.5_, and require a few minutes to be executed.

#### OneHot

This script provides a proof of concept of how to use one-hot kernels when computing correlations on categorical variables, and what are their limitations.
Results are shown in _Figure 5.1_, and are very fast to obtain.

#### Overfitting

This script illustrates an example of how HGR is ill-defined on data samples, as it can lead to severe overfitting.
Results are shown in _Figure 3.1_, and are very fast to obtain.

#### Projections

This script computes the level of %DIDI using several binning strategies when GeDI is used to enforce fairness constraints directly on the data.
Results are shown in _Figure 4.5_, and require a few minutes to be executed.

> Note: this script requires `Gurobi 11` to be manually installed in the local machine or in the Docker container.

## Contacts

In case of questions about the code or the paper, do not hesitate to contact **Luca Giuliani** ([luca.giuliani13@unibo.it](mailto:luca.giuliani13@unibo.it))
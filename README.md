# TFX Pipelines

This is a TFX example pipeline.

---

## Conceptual pipeline flow

The flow between TFX components is depicted in the graph below. The following can be said about the TFX components:
* `ExampleGen`:  
    Data is ingested into the pipeline and splitted into `Train` and `Eval` sets.

* `StatisticsGen`, `SchemaGen`, and `ExampleValidator`:  
    Data validation and anomalies detection.

* `Transform`:  
    Transformation and preprocessing of data.

* `Tuner`, and `Trainer`:  
    Estimator with tuned (or untuned) hyperparameters is trained.

* `Resolver`, and `Evaluator`:  
    Model analysis of trained model. The model will be assigned `BLESSED` or `UNBLESSED` depending on the evaluation metrics threshold(s).

* `InfraValidator`:  
    Model infrastructure validation. To guarantee the model is mechanically fine and prevents bad models from being pushed.

* `Pusher`:  
    Model validation outcomes. If a model is deemed `BLESSED` it will be pushed for serving.

---

Folder structure
```shell
.
├── data/                          # Data folder
├── notebooks/                     # Example TFX notebooks
├── outputs/                       # Local runs outputs folder
├── schema/                        # Custom defined schema
├── src/
│   ├── data
│   │   └── data.csv               # Source data
│   ├── models/                    # Directory of ML model definitions
│   │   ├── keras_model_baseline/
│   │   │   ├── constants.py       # Defines constants of the model
│   │   │   └── model.py           # Tuner/Trainer modules using Keras
│   │   ├── features_test.py
│   │   ├── features.py
│   │   └── preprocessing.py       # Defines preprocessing job using TF Transform
│   ├── pipeline/                  # Directory of pipeline definition
│   │   ├── configs.py             # Defines common constants for pipeline runners
│   │   └── pipeline.py            # Defines TFX components and a pipeline
│   ├── utils/                     # Directory of utils/helper functions
│   ├── data_validation.ipynb      # Data validation notebook
│   ├── local_runner.py            # Runner for local orchestration
│   └── model_analysis.ipynb       # Model analysis notebook
├── .dockerignore
├── .gitignore
├── Dockerfile
├── requirements.in                # Environment requirements
├── requirements.txt               # Compiled requirements
└── README.md
```

There are an example file with `_test.py` in its name. This is a unit test of the pipeline and it is recommended to add more unit tests as you implement your own pipelines. You can run unit tests by supplying the module name of test files with `-m` flag. You can usually get a module name by deleting `.py` extension and replacing `/` with `.`. For example:

```shell
# cd into src folder
$ cd src

# Run test file
$ python -m models.features_test
```

---

## Set up environment

This pipeline is running on Python 3.8.10. In order to create an environment to run locally follow these steps:

```shell
# Make sure you have the right Python activated

# Create virtual environment
$ python -m venv .venv

# Upgrade pip
$ .venv/bin/pip install --upgrade pip

# Install requirements
$ .venv/bin/pip install -r requirements.txt

# Activate environment
$ source .venv/bin/activate
```

This should spin up your local environment and you should be good-to-go running the pipeline and notebook scripts locally.

### Update environment
In case you need to update any requirements (e.g. update a package version or add new packages) do the following steps:
```shell
# Delete virtual environment (to make sure all old dependencies are removed)

# Make sure you have activated the right Python version and have pip-tools installed

# Update src/requirements.in with new package versions/added packages

# Compile requirements
$ pip-compile requirements.in

# Redo steps in section `Set up environment`
```

---

## Run pipeline locally (python CLI)

When you want to run the pipeline using the `local_runner.py` script, simply run:
```shell
$ python src/local_runner.py
```

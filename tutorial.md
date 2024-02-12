# Time Series Analytics Software Tutorial

This resource is a step-by-step tutorial to get familiar with the time series analytics software and its various components: [darts](https://unit8co.github.io/darts/index.html), [hydra](https://hydra.cc/), [pytorch-lightning](https://lightning.ai/docs/pytorch/latest/), [tensorboard](https://www.tensorflow.org/tensorboard), [mlflow](https://mlflow.org/docs/latest/tracking.html), etc. 

## 0. YAML Syntax

[Hydra](https://hydra.cc/docs/intro/) is configured through .yaml configuration files in the [configs](configs) directory. [See this resource for an introduction to YAML syntax](https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html). YAML is fairly similar to python (scope is controlled by indentation) and json. In particular the following syntax is frequently used:

```yaml
argument_name: argument_value  # in python this corresponds to variable_name = variable_value

argument_with_None_value: null  # null in yaml corresponds to None in python

# arguments that have no value but indented scope directly following is equivalent to a python dictionary
dict_name:  
  key_one: value_one
  key_two: value_two

# arguments with no value but indented scope directly following AND hyphens are equivalent to python lists
list_name:
  - list_value_1
  - list_value 2

# the following syntax is also equivalent to python lists
list_name2: [list_value_1, list_value_2]
```

Finally, all command line calls in this tutorial assume you are in the project root folder, i.e. the folder where src, logs, data etc. are located.

<br>

## 1. Data Configuration and Exploration

### Initial Configuration
<details>
<summary><b>Note on custom datasets</b></summary>
To use a custom dataset you must first configure it in a format understood by the time series analytics software. This is described in <a href="README.md">README.md</a> section Getting started - New Datasets
</details>

In this tutorial we will focus on an example dataset from the darts library, [AirPassengers](https://unit8co.github.io/darts/generated_api/darts.datasets.html#darts.datasets.AirPassengersDataset). When using a new dataset, you start by creating a config-file for the dataset under configs/datamodule/name_of_dataset.yaml. We will iteratively build up the dataset config and explore some useful operations it can do. In this case [the dataset config is already created](configs/datamodule/example_airpassengers.yaml), and contains the following:

```yaml
defaults:
  - base_darts_example
  - _self_

dataset_name: "AirPassengers"
```

The [defaults](https://hydra.cc/docs/1.3/tutorials/structured_config/schema/) section at the top of the config-file tells us that this config inherits the arguments that are set in the files listed in this section. The listed files are merged in the order that they are listed (i.e. for arguments specified in multiple files the value in the last file will be chosen), and the \_self_ keyword refers to this file (which if omitted is automatically inserted at the bottom, i.e. the current file takes precedence by default).

In this case we can see that this config file simply sets the argument dataset_name and inherits everything else from the [configs/datamodule/base_darts_example.yaml](configs/datamodule/base_darts_example.yaml) file:

```yaml
defaults:
  - base_data

_target_: src.datamodules.DartsExampleDataModule
dataset_name: ???
data_variables:
  target: null
train_val_test_split:
  train: 0.5
  val: 0.25
  test: 0.25
```

This file again inherits from [configs/datamodule/base_data.yaml](configs/datamodule/base_data.yaml). In this way you can reuse configuration that is common across datasets and don't have to rewrite the whole datamodule configuration for every new dataset you use. Further, in this file ([configs/datamodule/base_darts_example.yaml](configs/datamodule/base_darts_example.yaml)) we see two other hydra special arguments:
```yaml
_target_: src.datamodules.DartsExampleDataModule
dataset_name: ???
```
The \_target_ argument specifies a dotpath to a python callable (e.g. a class or a function), and the other arguments in the same scope are considered arguments to that target. The dotpath is used in essentially the same way you would import something in a normal python script. In this case it will find the class [DartsExampleDataModule](src/datamodules/darts_example_datamodule.py) in the script [src/datamodules/darts_example_datamodule.py](src/datamodules/darts_example_datamodule.py) and pass the arguments to its \_\_init__ function when instantiating the datamodule. (note that the \_target_ path points not to the file the script is in but to the directory that the script is in because the directory [src/datamodules](src/datamodules) is a python-package, i.e. it has an \_\_init.py__, and the class can therefore be imported directly from the package in addition to from the script its source code is in.)

The ??? value means that the dataset_name argument is required but it has to be overwritten (i.e. in the source code for the DartsExampleDataModule.\_\_init__ function it is a required argument), either by another config which inherits from it (like we did for [configs/datamodule/example_airpassengers.yaml](configs/datamodule/example_airpassengers.yaml)) or from the commandline when executing the program, e.g.
```bash
python src/train.py datamodule=base_darts_example datamodule.dataset_name="AirPassengers"
```

is equivalent to defining the [configs/datamodule/example_airpassengers.yaml](configs/datamodule/example_airpassengers.yaml) as we did above. There will be more on [command-line overrides later](#command-line-overrides).

Now we have configured the AirPassengers dataset and are ready to proceed with some data exploration!

### Data Exploration
Whenever you are analyzing a new dataset, you typically want to start with exploring the data. The [notebooks/data_explorer.ipynb](notebooks/data_explorer.ipynb) notebook is made specifically for this purpose. It loads your data configuration and provides various methods for visualizing your data and key properties of your data such as correlations and distributions. Go through the notebook and explore the AirPassengers dataset. Ensure that the notebook is configured to load the AirPassengers dataset, i.e. the first cell under configuration looks like:

```python
config_path = os.path.join("..", "..", "configs", "train.yaml") # NB: relative to <project_root>/src/utils (must be relative path)

config_overrides_dot = ["datamodule=example_airpassengers"]          # same notation as for cli overrides (dot notation). Useful for changing whole modules, e.g. change which datamodule file is loaded
config_overrides_dict = dict()                                  # Dictionary with overrides. Useful for larger changes/additions/deletions that does not exist as entire files.

cfg = src.utils.initialize_hydra(config_path, config_overrides_dot, config_overrides_dict, return_hydra_config=True, print_config=False)  # print config to inspect if all settings are as expected

show_encoders = False
```


Note for instance that data is sampled monthly and has yearly seasonal trends, thus the functions under the Seasonality header tells us that the data is seasonal with a period of 12.


Now we can start to have some fun and customize the dataset to our liking. Try the suggestions below and then rerun [notebooks/data_explorer.ipynb](notebooks/data_explorer.ipynb) to see the effects (the notebook uses autoreload to catch changes in external files, so you only have to rerun the cells below the "Configuration" header). It is good practice to perform the changes in the [configs/datamodule/example_airpassengers.yaml](configs/datamodule/example_airpassengers.yaml) file to avoid changing the configuration of other datasets that also inherit from the base configuration files.

#### Dataset splits
First, in the [configs/datamodule/base_data.yaml](configs/datamodule/base_data.yaml) configuration file we have configured that we want three dataset splits containing 50%, 25%, 25% of the total data, respectively. Try changing the dataset splits by adding the following to [configs/datamodule/example_airpassengers.yaml](configs/datamodule/example_airpassengers.yaml) in order to use only two splits with 50% and 25% (i.e. leaving out the last 25%):

```yaml
defaults:
  - base_darts_example
  - _self_

dataset_name: "AirPassengers"
train_val_test_split:
  train: 0.5
  val: 0.25
  test: null
```
 Note that since we inherit from [configs/datamodule/base_data.yaml](configs/datamodule/base_data.yaml) who also has defined the argument train_val_test_split.test we have to provide a value for train_val_test_split.test, otherwise we would inherit the train_val_test_split.test = 0.25 value. We can also try to leave out the first 25%:
 
```yaml
defaults:
  - base_darts_example
  - _self_

dataset_name: "AirPassengers"
train_val_test_split:
  train: [0.25, 0.75]
  val: [0.75, 1.0]
  test: null
```
 You can try to set the number of datapoints in each split in absolute numbers, e.g. 50 datapoints for training and 10 datapoints for validation:
 
```yaml
defaults:
  - base_darts_example
  - _self_

dataset_name: "AirPassengers"
train_val_test_split:
  train: 50
  val: 10
  test: 10
```

Or you can choose the dataset splits by time instead, e.g. to use beginning of dataset until october 1956 for training and the rest for validation (the datamodule will take care not to use overlapping datapoints even though we provide the same 1956-10 end and start point for both splits):

```yaml
defaults:
  - base_darts_example
  - _self_

dataset_name: "AirPassengers"
train_val_test_split:
  train: ["start", "1956-10"]
  val: ["1956-10", "end"]
  test: null
```

#### Transformations pipeline

Many machine learning models learn faster and perform better when the input data has a nice distribution. We can configure transformation pipelines that calculate statistics over the training set and transforms the dataset splits provided to the machine learning models. To this end, the datamodule configuration has the argument processing_pipeline whose default value is set in the [configs/datamodule/base_data.yaml](configs/datamodule/base_data.yaml) config to fill in missing values (i.e. NaNs). It takes a [darts pipeline object](https://unit8co.github.io/darts/generated_api/darts.dataprocessing.pipeline.html) which in turn takes a list of [transformers](https://unit8co.github.io/darts/generated_api/darts.dataprocessing.transformers.html).

We can for instance normalize the data using a [Standard Scaler from sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) (note that we have to wrap non-darts scalers with the [Scaler transformer wrapper](https://unit8co.github.io/darts/generated_api/darts.dataprocessing.transformers.scaler.html)):

```yaml
defaults:
  - base_darts_example
  - _self_

dataset_name: "AirPassengers"
train_val_test_split:
  train: ["start", "1956-10"]
  val: ["1956-10", "end"]
  test: null
processing_pipeline:
  _target_: darts.dataprocessing.Pipeline
  transformers:
    - _target_: darts.dataprocessing.transformers.Scaler
      scaler:
        _target_: sklearn.preprocessing.StandardScaler
    - _target_: darts.dataprocessing.transformers.missing_values_filler.MissingValuesFiller
      fill: "auto"  # The default, will use pandas.Dataframe.interpolate()
```

The observant reader might notice that adding this pipeline config had no effect on the data, which is true and expected as the [configs/datamodule/base_data.yaml](configs/datamodule/base_data.yaml) config which we inherit from already contains this exact pipeline config.

The AirPassengers dataset has a clear increasing trend with time, which makes the mean of the training and validation/test splits very different. In order to make the time series stationary we can instead consider the difference between successive points in time using the [Diff-transformer](https://unit8co.github.io/darts/generated_api/darts.dataprocessing.transformers.diff.html).

```yaml
defaults:
  - base_darts_example
  - _self_

dataset_name: "AirPassengers"
train_val_test_split:
  train: ["start", "1956-10"]
  val: ["1956-10", "end"]
  test: null
processing_pipeline:
  _target_: darts.dataprocessing.Pipeline
  transformers:
    - _target_: darts.dataprocessing.transformers.Diff
      lags: 1
    - _target_: darts.dataprocessing.transformers.Scaler
      scaler:
        _target_: sklearn.preprocessing.StandardScaler
    - _target_: darts.dataprocessing.transformers.missing_values_filler.MissingValuesFiller
      fill: "auto"  # The default, will use pandas.Dataframe.interpolate()
```

Another useful transformer is the family of power transformations, e.g. do a log transformation using a [Box-Cox transformer](https://unit8co.github.io/darts/generated_api/darts.dataprocessing.transformers.boxcox.html):
```yaml
defaults:
  - base_darts_example
  - _self_

dataset_name: "AirPassengers"
train_val_test_split:
  train: ["start", "1956-10"]
  val: ["1956-10", "end"]
  test: null
processing_pipeline:
  _target_: darts.dataprocessing.Pipeline
  transformers:
    - _target_: darts.dataprocessing.transformers.BoxCox
      lmbda: 0
    - _target_: darts.dataprocessing.transformers.missing_values_filler.MissingValuesFiller
      fill: "auto"  # The default, will use pandas.Dataframe.interpolate()
```

#### Resampling
The AirPassengers dataset contains monthly data. If we instead want to analyse the data on a yearly-basis we can leverage the resampling functionality of the TimeSeriesDataModule:

```yaml
defaults:
  - base_darts_example
  - _self_

dataset_name: "AirPassengers"
train_val_test_split:
  train: ["start", "1956-10"]
  val: ["1956-10", "end"]
  test: null
processing_pipeline:
  _target_: darts.dataprocessing.Pipeline
  transformers:
    - _target_: darts.dataprocessing.transformers.BoxCox
      lmbda: 0
    - _target_: darts.dataprocessing.transformers.missing_values_filler.MissingValuesFiller
      fill: "auto"  # The default, will use pandas.Dataframe.interpolate()
resample:
  freq: "1Y" # resample to yearly frequency
  method: "sum" # with the new datapoints containing the sum of the downsampled datapoints.
```

#### Target and Covariates

Darts distinguishes between a few different data types and how they can be used by the models, [see this page for a complete explanation](https://unit8co.github.io/darts/userguide/covariates.html). Target variables are those variables that we wish our models to forecast into the future, i.e. they are both inputs and (the only) outputs of the forecasting models. We can feed additional data that has predictive power for the target variables as inputs to the models, these additional variables are called covariates. Further, covariates are categorized based on their availability in the future: past covariates are only available up until the point in time from which we are forecasting the future (e.g. because the covariate is some sensor measurement), while future covariates are available also for future points in time at least as far as we are forecasting ahead (e.g. because they are themselves forecast of weather). Finally, there are static covariates which do not depend on time (e.g. because they encode some property of the data such as a type of product for which we are forecasting sales).

Normally, when you define a new datamodule it is required to specify which variables in your dataset are targets, past covariates, future covariates, and static covariates. That is, the [configs/datamodule/base_data.yaml](configs/datamodule/base_data.yaml) config file we inherit from has defined the argument data_variables as follows:

```yaml
data_variables:
  target: ???
  past_covariates: null
  future_covariates: null
  static_covariates: null
```

Thus, one must at least specify which variables are target variables. These should be the names of columns in your pandas DataFrame. For [darts datasets](https://unit8co.github.io/darts/generated_api/darts.datasets.html) that use the [DartsExampleDataModule](src/datamodules/darts_example_datamodule.py) class there is a trick that simply chooses all variables in the dataset as target variables. This is why we didnt have to specify our variables for the [AirPassengers](https://unit8co.github.io/darts/generated_api/darts.datasets.html#darts.datasets.AirPassengersDataset) dataset.

The [AirPassengers](https://unit8co.github.io/darts/generated_api/darts.datasets.html#darts.datasets.AirPassengersDataset) dataset only has one variable: "#Passengers" representing the number of passengers, there are no covariates available in the source data. Instead, to illustrate the concept of covariates we can leverage dart's functionality for automatically generating time-based covariates through its [encoders](https://unit8co.github.io/darts/generated_api/darts.dataprocessing.encoders.encoders.html).

#### Encoders

Darts provides a number of [time-based past/future covariates](https://unit8co.github.io/darts/generated_api/darts.dataprocessing.encoders.encoders.html) which are generated by encoding the index of a dataset. For instance, you can generate covariates that indicate to the model the hour of the day, day of week, month of year, special holidays etc., see [here](https://unit8co.github.io/darts/quickstart/00-quickstart.html#Encoders:-using-covariates-for-free) for more examples. To use these encoders, the models have an argument add_encoders. 

For the [AirPassengers](https://unit8co.github.io/darts/generated_api/darts.datasets.html#darts.datasets.AirPassengersDataset) dataset knowing the month of the year is important to learn the seasonal component of the data, and knowing what the year is important to learn the strength of the seasonality (as the oscillations increase in amplitude with year). Thus, to give these features to our model (lets use the [XGBoost model](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.xgboost.html) as an example), we could modify its configuration file [configs/model/xgboost.yaml](configs/model/xgboost.yaml) by adding the following:

```yaml
add_encoders:
  cyclic:
    future: ["month"]
  datetime_attribute:
    future: ["year"]
  transformer:
    _target_: darts.dataprocessing.transformers.Scaler
```

A cyclic encoder will generate sinus and cosinus signals whose combination of values at any point in time uniquely encodes that timestamp. We configure this encoder as a future covariate, as this is a known function (sin/cos) that we can generate values for arbitrarily far into the future.

The datetime_attribute encoder encodes datetime features, here just taking the year of the datapoints as a feature. Since the year varies from 1949-1961 it has a very different range from the cyclic features (between -1 and 1), so we additionally add a Scaler transformer to ensure that our features share a common scale.

We can visualize these encodings with [notebooks/data_explorer.ipynb](notebooks/data_explorer.ipynb). Add the add_encoders argument to the [configs/model/xgboost.yaml](configs/model/xgboost.yaml) file as above and ensure the configuration cell of the data_explorer notebook looks as follows (namely, we must specify we want to use the xgboost model and that show_encoders is True):

```python
config_path = os.path.join("..", "..", "configs", "train.yaml") # NB: relative to <project_root>/src/utils (must be relative path)

config_overrides_dot = ["datamodule=example_airpassengers", "model=xgboost"]          # same notation as for cli overrides (dot notation). Useful for changing whole modules, e.g. change which datamodule file is loaded
config_overrides_dict = dict()                                  # Dictionary with overrides. Useful for larger changes/additions/deletions that does not exist as entire files.

cfg = src.utils.initialize_hydra(config_path, config_overrides_dot, config_overrides_dict, return_hydra_config=True, print_config=False)  # print config to inspect if all settings are as expected

show_encoders = True
```

Now, the encoders will be plotted alongside the other target and covariate variables of your dataset. You can also see the correlation of the encodings with the other variables under the Feature correlation header.

<br>

## 2. Model Training

After the data has been explored and properly configured, it is time to train some models! Before proceeding, ensure [configs/datamodule/example_airpassengers.yaml](configs/datamodule/example_airpassengers.yaml) now looks like this:

```yaml
defaults:
  - base_darts_example
  - _self_

dataset_name: "AirPassengers"
train_val_test_split:
  train: 0.75
  val: 0.25
  test: null
processing_pipeline:
  _target_: darts.dataprocessing.Pipeline
  transformers:
    - _target_: darts.dataprocessing.transformers.Diff
      lags: 1
    - _target_: darts.dataprocessing.transformers.Scaler
      scaler:
        _target_: sklearn.preprocessing.StandardScaler
    - _target_: darts.dataprocessing.transformers.missing_values_filler.MissingValuesFiller
      fill: "auto"  # The default, will use pandas.Dataframe.interpolate()
```

The [src/train.py](src/train.py) script handles everything related to training of models. This script uses the corresponding [configs/train.yaml](configs/train.yaml) file for configuration, which inherits from all the other configuration groups in order to construct one final configuration file that completely describes the state of the model and data used during the training run. 

<details>
<summary><b>Hydra configuration explanations</b></summary>
Of note here is the introduction of namespaces in the defaults list, e.g.:

```yaml
defaults:
  - _self_
  - fit: default.yaml
```

This syntax means that hydra looks in the [configs/fit](configs/fit) folder for a file called default.yaml, and places its content under the fit namespace, like so:

```yaml
fit:
  arguments: some_value
  from: another_value
  default: 3
```

In the final train.yaml config object (the cfg variable in the [src/train.py](src/train.py) script), all the arguments for fit are then available under the fit-namespace (cfg.fit).

Further, the defaults list contains a null-value:
```yaml
defaults:
  #...
  # debugging config (enable through command line, e.g. `python train.py debug=default)
  - debug: null
  #...
```
This syntax means that the debug namespace will have a None-value (cfg.debug = None), and importantly enables the debug config to be controlled through the command-line as described in the comment.

The final piece of new syntax is the optional keyword:
```yaml
defaults:
  # ...
  # optional local config for machine/user specific settings
  # it's optional since it doesn't need to exist and is excluded from version control
  - optional local: default.yaml
```
Hydra will then look for a file called default.yaml in [configs/local](configs/local) but will not throw an error if the file is not found.

Finally, notice that the default model is set to [tcn.yaml](configs/model/tcn.yaml) which configures a [temporal convolutional neural network pytorch-model](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tcn_model.html). This config file inherits from [configs/model/base_torch.yaml](configs/model/base_torch.yaml) which again configures arguments that are shared among the pytorch models. Here we encounter yet some more new hydra-syntax. The line at the top of the file signifies that even if this file is inherited from using a namespace as above (for fit etc.), the contents will be placed in the global (i.e. outermost) scope:

```yaml
# @package _global_
```

Thus, all model related arguments are placed under the model namespace in this file. Further, its default list has new syntax. Since this file specifies that it exists in the global namespace, when we inherit other files we use a leading forward slash to indicate that it should look for a config file in the [configs/logger](configs/logger) folder (otherwise it would look in the global namespace i.e. directly in the [configs](configs) folder):

```yaml
defaults:
  - /logger: tb_mlflow.yaml
```

Here we see that torch models by default sets both tensorboard and mlflow as loggers. We will get back to these loggers in a minute.

</details>

Notice that the defaults in the [configs/train.yaml](configs/train.yaml) is to train a [TCNModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tcn_model.html) on the [AirPassengers dataset](https://unit8co.github.io/darts/generated_api/darts.datasets.html#darts.datasets.AirPassengersDataset). Thus, if we simply call the training script without further arguments, that is what we get:

```bash
python src/train.py
```

### Run directory

When we start a training run hydra will create a run directory and collect all files belonging to that run and save them in the run directory. The run directory is located at logs/<task_name>/<run_type>/<DATE_TIME> (configured by the argument hydra.run.dir in [configs/hydra/default.yaml](configs/hydra/default.yaml)) where <task_name> is "train" in this case (configured by the argument task_name in [configs/train.yaml](configs/train.yaml)) or ["debug" if we use a debug config](#debug), <run_type> is either "runs" or ["multiruns"](#command-line-overrides) (see [Hydra sweeps](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/) below), and the run name is set as the current date and time. The contents of the run directory is as follows:

```
├── .hydra                       <- Hydra configuration files
│   ├── config.yaml              <- train.yaml used in the run with overrides
│   ├── hydra.yaml               <- Hydra configuration for the run
│   ├── overrides.yaml           <- Command line overrides for the run
│   └── resolved_config.yaml     <- train.yaml with variables resolved
│
├── checkpoints                  <- For pytorch models, files containing model weights and trainer state
│
├── datamodule                   <- Saved state for datamodule, e.g. data transformation pipeline
│
├── plots                        <- Plots, e.g. of datasets and predictions
│
├── tensorboard                  <- Tensorboard logs, if tensorboard is used
│
├── config.tree.log              <- Config as printed to terminal during run
├── eval_test_results.yaml       <- If test is run, the results (metric values)
├── eval_val_results.yaml        <- If validation is run, the results (metric values)
├── exec_time.log                <- Execution time in seconds of training run
├── model.pkl                    <- Saved model file (will have another name for pytorch models)
└── tags.log                     <- Tags configured for run
```

If the arguments validate or test in [configs/train.yaml](configs/train.yaml) are True, the model will after training be evaluated on the validation and/or test set, respectively (if these splits are configured in the datamodule). See the [section on evaluation](#4-evaluating-a-trained-model) for more information.

Go look at the config saved for your model run at .hydra/config.yaml, compare to the original [configs/train.yaml](configs/train.yaml) to get a sense of how the hierarchical config composition method of Hydra works. 

### Loggers

Since the [TCNModel](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tcn_model.html) is a pytorch model it will log information such as metric values to both Tensorboard and mlflow by default (see Hydra configuration explanations above).

<details>
<summary><b>Tensorboard</b></summary>
<a href="https://www.tensorflow.org/tensorboard">Tensorboard</a> is a useful program to monitor the state of your model training runs while they are undergoing, but also to compare different model runs after the fact. Tensorboard can be started by calling it from a terminal:

```bash
tensorboard --logdir logs
```

This will start tensorboard on port 6006 (can be changed with the --port port_number argument) and load all the models you have trained that include the tensorboard logger. You can see tensorboard by going to <a href="http://localhost:6006">localhost:6006<a/> in your browser.

The SCALARS tab at the top plots the metrics we have configured for our model as a function of time (i.e. number of epochs/gradient steps depending on what metric it is and for which stage: training or validation). In addition to the loss function of the model, you can add more metrics here by configuring the torch_metrics argument of your model configuration (default value is set in [configs/model/base_torch.yaml](configs/model/base_torch.yaml)). Open the TCN model config [configs/model/tcn.yaml](configs/model/tcn.yaml) and add for instance the [explained variance](https://torchmetrics.readthedocs.io/en/stable/regression/explained_variance.html) and the [cosine similarity](https://torchmetrics.readthedocs.io/en/stable/regression/cosine_similarity.html) metrics, see [Torchmetrics](https://torchmetrics.readthedocs.io/en/stable/) for more metrics:

```yaml
torch_metrics:
  _target_: torchmetrics.MetricCollection
  metrics:
    - _target_: torchmetrics.ExplainedVariance
    - _target_: torchmetrics.CosineSimilarity
```

The IMAGES tab shows figures generated during training. By default it shows the datasets that the model is trained/validated/tested on (if the plot_datasets argument of [configs/train.yaml](configs/train.yaml) is True), and examples of predictions on the validation set using the [PredictionPlotterCallback](src/callbacks/visualization.py) configured in [configs/callbacks/prediction_plotter.yaml](configs/callbacks/prediction_plotter.yaml). The same example predictions are plotted each validation epoch, and on the top of the figure there is a slider one can drag to select which epoch prediction to show. In this way you can get a sense of how your model predictions are evolving over time with more training.

</details>

<details>
<summary><b>Mlflow</b></summary>

<a href="https://mlflow.org/docs/latest/tracking.html#tracking-ui">The mlflow tracking ui</a> can be used in the same manner as Tensorboard to monitor the progress of your model training runs, but it is particularly useful for the task of comparing models. Mlflow ui can be started through a terminal:

```bash
cd logs/mlflow
mlflow ui
```

(note that you have to run mlflow ui from the logs/mlflow folder), or on macOS/linux using the scripts/run_mlflow_ui.sh bash script:

```bash
bash scripts/run_mlflow_ui.sh
```

This starts the mlflow ui service on the port 5000, and you can view it by opening your browser at <a href="http://localhost:5000">localhost:5000<a/>. Runs are organized into experiments, you can select which experiments to show by selecting them from the menu on the left hand side. The default experiment name is tsanalytics (set in [configs/logger/mlflow.yaml](configs/logger/mlflow.yaml)). Once you select experiments you can see all runs in the selected experiments. If you started from scratch and followed the tutorial up until now there will not be many runs to compare. Train a new model with different configuration, e.g. double the num_filters of the TCNModel:

```bash
python src/train.py model.num_filters=6
```

The refresh button on the right hand side should now show a circle icon indicating that there is a new run to display. With several runs in the mlflow ui, you can press the "(+) show more metrics and parameters" button in order to customize which metrics and parameters show up in the table overview. Choose for instance metrics/val_loss and under parameters model/num_filters. Further, you can use the search bar to filter only models that have a certain configuration, try for instance parameters."model/num_filters" = "3" to get only the first model with 3 filters.

By clicking on the name of a run you can see more information about that run, such as plots of the datasets that the model is trained/validated/tested on (if the plot_datasets argument of [configs/train.yaml](configs/train.yaml) is True), and examples of predictions on the validation set using the [PredictionPlotterCallback](src/callbacks/visualization.py) configured in [configs/callbacks/prediction_plotter.yaml](configs/callbacks/prediction_plotter.yaml). You can also plot metrics as a function of time (i.e. number of epochs/gradient steps depending on what metric it is and for which stage: training or validation). In addition to the loss function of the model, you can add more metrics here by configuring the torch_metrics argument of your model configuration (default value is set in [configs/model/base_torch.yaml](configs/model/base_torch.yaml)). Open the TCN model config [configs/model/tcn.yaml](configs/model/tcn.yaml) and add for instance the [explained variance](https://torchmetrics.readthedocs.io/en/stable/regression/explained_variance.html) and the [cosine similarity](https://torchmetrics.readthedocs.io/en/stable/regression/cosine_similarity.html) metrics, see [Torchmetrics](https://torchmetrics.readthedocs.io/en/stable/) for more metrics:

```yaml
torch_metrics:
  _target_: torchmetrics.MetricCollection
  metrics:
    - _target_: torchmetrics.ExplainedVariance
    - _target_: torchmetrics.CosineSimilarity
```

</details>

In order to configure the training runs we have a few different options which each have their pros and cons: command line overrides, editing the configuration files, and experiment configs.

### Command line overrides

Pros
+ Simple and quick 
+ Can easily create [sweeps](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/)
+ Easy to swap out whole config groups, e.g. change which model to use
+ Changes are isolated to this run, thus avoids impacting other runs by changing the default values

Cons
- Not as easily repeatable (saved to logs/<task_name>/<run_mode>/<run_name>/.hydra/overrides.yaml)
- Not scalable to many argument overrides

The simplest and quickest option is to override arguments on the command-line, similar to how one would use the python ArgParser. E.g. we can instead train an [LSTM model](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.rnn_model.html), set the number of parameters in each layer to 10, and set the maximum number of training epochs to 15:

```bash
python src/train.py model=rnn ++model.hidden_dim=10 trainer.max_epochs=15
```

Here, the model.hidden_dim argument has "++" in front. This is because this argument is not present in the model yaml files, so we have to use "+" to signal that we add a new argument, and by using "++" we say add if it does not exist and overwrite the value if it does exist. Further, since model is a namespace, when we write model=rnn hydra will look for a file called rnn.yaml in [configs/model](configs/model), and overwrite all model arguments of the default config [configs/train.yaml](configs/train.yaml) with the arguments in the [rnn.yaml](configs/model/rnn.yaml) file. 

You can verify that the argument overrides were used by inspecting the config printed to the terminal during script execution, by inspecting the model run in tensorboard or mflow, or by looking at the config saved for your model in .hydra/config.yaml as described in the [run directory section](#run-directory).


#### Hydra sweeps

Command line arguments are useful to create [Hydra sweeps](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/), i.e. creating multiple training runs over a grid of values. We can for instance train three different models, an [ARIMA model](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.arima.html), an [XGBoost model](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.xgboost.html), and a [Linear regression model](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.linear_regression_model.html):

```bash
python src/train.py -m model=arima,xgboost,linear_regression
```

Note here that we have to provide the -m flag (short for multirun) which tells Hydra that we want to create multiple runs (as opposed to model's value being a list with three elements). The default behavior of Hydra is to run these sequentially, but this can be changed through the [launcher option](https://hydra.cc/docs/plugins/joblib_launcher/), e.g. if you run this software on a slurm cluster and have executed the [scripts/configure_slurm.sh](scripts/configure_slurm.sh) bash script the runs will be submitted as slurm jobs and ran in parallel (given the cluster has the available resources). Again, hydra will look for .yaml files in [configs/model](configs/model) with the names we provided above.

Sweeps can also be used to do simple hyperparameter searches, e.g. to evaluate an [ARIMA model's](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.arima.html) performance with various number of lags of the target variables and number of differentations of the data:

```bash
python src/train.py -m model=arima model.p=4,6,8 model.d=0,1
```

will create 6 different training runs with all combinations of model.p and model.d. In general, you should check the [darts documentation](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.html) for the names of model arguments and what their values mean.

### Editing the configuration files
Pros
+ Simple and quick
+ Shared among all runs, thus reduces overhead of writing configuration

Cons
- Not as easily repeatable if you later change the defaults again
- Impacts all subsequent runs, thus could cause unintended side effects

The next simplest method is to change the main configuration file, which in the case of training models is the [configs/train.yaml](configs/train.yaml) file. If you will mostly be working with a single type of model, e.g. [an XGBoost model](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.xgboost.html), then changing the default model file will save you from having to change it on the command-line every time you train a new model:
```yaml
defaults:
  - _self_
  # ...
  - model: xgboost.yaml
  # ...
```

Then we can just run the training script with no arguments to train an [XGBoost model](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.xgboost.html):
```bash
python src/train.py
```

Recall that the [AirPassengers](https://unit8co.github.io/darts/generated_api/darts.datasets.html#darts.datasets.AirPassengersDataset) data is sampled monthly and that there are yearly seasonal trends, i.e. every 12 datapoints (as we were informed by the functions under the Seasonality header in [notebooks/data_explorer.ipynb](notebooks/data_explorer.ipynb)). Thus, when we are working with this dataset we probably want our models to take as input at least the last 12 data points so that they can see one full period of the season. Continuing with the [XGBoost model](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.xgboost.html) we therefore update its configuration file [configs/model/xgboost.yaml](configs/model/xgboost.yaml) by setting its argument lags to 12:

```yaml
defaults:
  - base_nontorch

_target_: darts.models.forecasting.xgboost.XGBModel
lags: 12
output_chunk_length: 1
```

Now whenever we run the training script we will train an [XGBoost model](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.xgboost.html) where the model lags is tailored to our dataset:
```bash
python src/train.py
```

### Experiment configs

Pros

* Suitable for modifying many arguments at once across namespaces
* Easily reusable
* Can be used to save configuration, e.g. the best hyperparameters found

Cons

* Some additional configuration overhead
* Not as easy to create sweeps

The experiment namespace in [configs/experiment](configs/experiment) is intended as a method to make modifications to many namespaces at once (e.g. changing both datamodule and model configurations) in a reusable manner. Take a look at the [configs/experiment/example.yaml](configs/experiment/example.yaml) file for instance:

```yaml
# @package _global_

# to execute this experiment run:
# python train.py experiment=example

defaults:
  - override /datamodule: example_airpassengers.yaml
  - override /model: xgboost.yaml
  - _self_

# all parameters below will be merged with parameters from default configurations set above
# this allows you to overwrite only specified parameters

tags: ["airpassengers", "xgboost", "diff_stationary"]

seed: 12345

model:
  lags: 12

datamodule:
  train_val_test_split:
    train: 0.75
    val: 0.25
    test: null
  processing_pipeline:
    _target_: darts.dataprocessing.Pipeline
    transformers:
      - _target_: darts.dataprocessing.transformers.Diff
        lags: 1
      - _target_: darts.dataprocessing.transformers.Scaler
        scaler:
          _target_: sklearn.preprocessing.StandardScaler
      - _target_: darts.dataprocessing.transformers.missing_values_filler.MissingValuesFiller
        fill: "auto"  # The default, will use pandas.Dataframe.interpolate()
```

In this file we configure many of the arguments we have looked at previously in this tutorial: We set the dataset to be the [AirPassengers dataset](https://unit8co.github.io/darts/generated_api/darts.datasets.html#darts.datasets.AirPassengersDataset), we set the model to be [XGBoost](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.xgboost.html), we configure the data pipeline to make the data stationary using the diff transformer, we set the model.lags argument to contain one whole seasonal period, and we fix the seed to make any randomness in sampling reproducible. We can then train a model with this configuration as follows:

```bash
python src/train.py experiment=example
```

This configuration is now isolated to this one file, we avoided making any changes to the default configuration files, and it is easily reproducible in the future. Another convenient use case for experiment configs is to save the best hyperparameters that we found for some combination of model and data configuration. For instance, lets say we did a grid search for the [ARIMA model](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.arima.html) and found model.p=12 and model.d=1 to work best when we used the diff transformer as in the example config above. We could then save these arguments to a file configs/experiment/arima_best_hyp_diff.yaml so that we have it archived for future use.

### Debug

The time series analytics software includes several debug configs in [configs/debug](configs/debug) that are useful for debugging your models. In addition to the effects described below, debug configs will disable multiprocessing and force training on CPU, making the training pipeline deterministic and play nicer with debuggers. Note that some of these features are specific to pytorch models. Debug mode is enabled by passing the name of a debug config as a command line override, e.g.:

```bash
python src/train.py debug=fdr
```

This will enable the [configs/debug/fdr.yaml](configs/debug/fdr.yaml) ([fast dev run](https://lightning.ai/docs/pytorch/stable/common/trainer.html#fast-dev-run)) config, which will run only one single batch and epoch for training, validation and testing (if validation and test sets are configured). This is useful to verify that your model can go through the whole training pipeline without failure and not have to wait a long time to find out.

There is also the [configs/debug/overfit.yaml](configs/debug/overfit.yaml) config which [uses only a few batches of data for training](https://lightning.ai/docs/pytorch/stable/common/trainer.html#overfit-batches), and that is useful to verify that your model is configured reasonably in the sense that it is able to learn to perform well on a very small dataset (i.e. overfit). If it cannot do this, then it will probably not perform well on the complete dataset:

```bash
python src/train.py debug=overfit
```

Along the same lines there is the [configs/debug/limit.yaml](configs/debug/limit.yaml) config, which enables you to [control the number of batches](https://lightning.ai/docs/pytorch/stable/common/trainer.html#limit-train-batches) for training/validation/testing either as a fraction of complete dataset or as an absolute number.

```bash
python src/train.py debug=limit
```

The limit debug config should fail for the configuration we have set up since the [AirPassengers dataset](https://unit8co.github.io/darts/generated_api/darts.datasets.html#darts.datasets.AirPassengersDataset) is so small that there is only a single validation batch and we cant limit it further. 

Finally, there is the [configs/debug/profiler.yaml](configs/debug/profiler.yaml) config which wille enable the [pytorch lightning profiler](https://lightning.ai/docs/pytorch/stable/tuning/profiler.html), which can be used to identify which parts of the code is slowing your model down and subsequently improving the throughput of your model training.

```bash
python src/train.py debug=profiler
```

<br>

## 3. [Optional] Running on a Slurm Cluster
The time series analytics software supports running on [Slurm](https://slurm.schedmd.com/documentation.html) clusters. This is useful for running experiments that take a long time to train, or for running many experiments in parallel. Through the magic of [Hydra plugins](https://hydra.cc/docs/plugins/submitit_launcher/), the time series analytics software can be configured to submit jobs through slurm, use GPU resources, and even use [Slurm arrays](https://slurm.schedmd.com/job_array.html) to run many experiments in parallel.

### Setup

To get started, clone the time series analytics repository to a cluster running slurm and set up your environment (it is advised to [install pytorch and CUDA](https://pytorch.org/get-started/locally/) through conda to easily enable GPU support). Then, execute the slurm configuration script which will make submitting slurm jobs the default behavior:

```bash
bash scripts/configure_slurm.sh
```

This script simply renames the [configs/local/slurm.yaml](configs/local/slurm.yaml) configuration file to [configs/local/default.yaml](configs/local/default.yaml), such that whenever we run a hydra command without specifying a local config, the slurm config will be used. Through the slurm config you can configure parameters of your jobs, such as how many how much memory each job should maximally use, how many CPUs for each job, how many jobs to run in parallel (the rest will be queued) etc.

### Running jobs and monitoring status

Now, whenever we run the software (e.g. one of the training commands above, or the hyperparameter optimization commands below) it will be submitted as a slurm job. Try e.g.:

```bash
python src/train.py model=rnn trainer.min_epochs=20 trainer.max_epochs=20
```

This will print out a message saying that your job(s) have been submitted to slurm, and will keep printing out the status of your jobs until they are finished if the requisite slurm features are setup on the system, otherwise it will print some errors. You can safely ignore these errors, and use ctrl + C to stop the status printing and return to the terminal, your job will keep running.

You can then check the status of your job by running (note that since the dataset is so small, the job will perhaps finish before you have time to run this command):

```bash
squeue
```

and check nvidia-smi to confirm that your job is using GPU:

```bash
nvidia-smi
```

The standard output of your job is redirected to a file in the run log directory under <run_log_directory>/.submitit/<slurm_job_id>/<slurm_job_id>\_<job_array_id>\_log.out, and you can therefore also check this file to see the status of your job, or check the _log.err file for any errors. Note that when using the [slurm launcher](https://hydra.cc/docs/plugins/submitit_launcher/), the job will always be of <run_type> multirun even if you are only training/evaluating a single model.

#### Hyperparameter optimization and other scripts that spawn new jobs

Note that if your script involves starting new jobs after previous jobs finish, as is the case when running hyperparameter optimization, you need to keep the initial job submission command running until all jobs are finished. To do this, you can use the nohup command as below:

```bash
nohup python src/train.py hparams_search=example_optuna &
```

nohup will prevent the job from stopping when you exit the terminal, and the & operator signals that the job should be run in the background.

### Tensorboard and mlflow

The tensorboard and mlflow loggers are useful to monitor the status of your runs on the cluster. To use them, you need to start the tensorboard and/or mlflow servers on the cluster, ideally as a slurm job (TODO scripts for this), and you need to tunnel the ports to your local machine. For example, to tunnel the tensorboard port you can use the following ssh command to connect to the cluster:

```bash
ssh -L 6007:localhost:6006 <username>@<cluster_address>
```

which will forward the port 6006 on localhost on the cluster (where tensorboard sends data), to the port 6007 on your local machine, and you can now access tensorboard in your browser at [https://localhost:6007](https://localhost:6007) as you would if tensorboard was running locally on your machine.


<br>

## 4. Hyperparameter Optimization
Performing hyperparameter optimization is easy with the time series analytics software. Many optimization frameworks are supported including [Ax](https://hydra.cc/docs/plugins/ax_sweeper/), [Nevergrad](https://hydra.cc/docs/plugins/nevergrad_sweeper/), and [Optuna](https://hydra.cc/docs/plugins/optuna_sweeper/). We will focus on Optuna in this tutorial (the others have not been tested with the time series analytics software). 

Hyperparameter optimization is configured in the [configs/hparams_search](configs/hparams_search) namespace, the [configs/hparams_search/example_optuna.yaml](configs/hparams_search/example_optuna.yaml) file shows an example with the [ARIMA model](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.arima.html#).


```yaml
# @package _global_

# example hyperparameter optimization of some experiment with Optuna:
# python train.py -m hparams_search=mnist_optuna experiment=example

defaults:
  - override /hydra/sweeper: optuna
  - override /model: arima

# choose metric which will be optimized by Optuna
# metrics have the naming pattern {data split}_{metric name} where metric name is the name of the function or class implementing the metric.
# make sure this is the correct name of some metric defined in:
  # Torch models:     model.loss_fn or model.torch_metrics
  # Non-torch models: eval.eval_kwargs.metric
optimized_metric: "val_mse"

# Sets callbacks to monitor same metric as hyperparameter optimization and same higher/lower is better.
callbacks:
  early_stopping:
    monitor: ${optimized_metric}
    patience: 2
    mode: ${eval:'"${hydra:sweeper.direction}"[:3]'}
  model_checkpoint:
    monitor: ${optimized_metric}
    mode: ${eval:'"${hydra:sweeper.direction}"[:3]'}

# here we define Optuna hyperparameter search
# it optimizes for value returned from function with @hydra.main decorator
# docs: https://hydra.cc/docs/next/plugins/optuna_sweeper
hydra:
  mode: "MULTIRUN" # set hydra to multirun by default if this config is attached

  sweeper:
    _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper

    # storage URL to persist optimization results
    # for example, you can use SQLite if you set 'sqlite:///example.db'
    # not possible to set to hydra.output_dir because db is created before output_dir
    storage: "sqlite:///logs/optuna/hyperopt.db"

    # name of the study to persist optimization results
    study_name: "arima_example"

    # number of parallel workers
    n_jobs: 5

    # 'minimize' or 'maximize' the objective
    direction: "maximize"

    # total number of runs that will be executed
    n_trials: 20

    # choose Optuna hyperparameter sampler
    # you can choose bayesian sampler (tpe), random search (without optimization), grid sampler, and others
    # docs: https://optuna.readthedocs.io/en/stable/reference/samplers.html
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 1234
      n_startup_trials: 5 # number of random sampling runs before optimization starts

    # define hyperparameter search space
    params:
      model.p: range(2, 12)
      model.d: range(0, 1)
```

Here we first set the sweeper to optuna and the model to be [ARIMA](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.arima.html#). Then we set the optimized_metric argument which is the metric that optuna will observe and attempt to improve. We configure the [callbacks](https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html) (for pytorch models) to also monitor this metric. Then under hydra.sweeper we configure the [Optuna optimizer](https://optuna.readthedocs.io/en/stable/index.html): We set where the results are saved (storage, study_name), how many jobs to run in parallel (if supported by [configs/local](configs/local) configuration, otherwise they are still run sequentially) and in total (n_jobs, n_trials), the [sampler](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html) and finally what parameter space we are searching over (params). The arguments under params should be some arguments that exist elsewhere in the final [configs/train.yaml](configs/train.yaml) config. In this case the [ARIMA](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.arima.html#) has two arguments called model.p and model.q, and we construct a grid over them with the range command. See [hydra sweep syntax](https://hydra.cc/docs/advanced/override_grammar/extended/#sweeps) for the types of spaces that can be created. Optuna will create a grid over all the parameter values that we list in params.

To run this example simply override the argument on the command line:

```bash
python src/train.py hparams_search=example_optuna
```

Optuna will create a database file on the path we configured in the hydra.sweeper.storage argument (by default logs/optuna/optuna_hyperopt.db), and in this database create an study with the name we set for hydra.sweeper.study_name. A single database file can hold multiple independent studies, or if we rerun the script with the same storage and study_name argument optuna will resume optimization from where it last left off. 

The optimization results can be inspected using [optuna-dashboard](https://github.com/optuna/optuna-dashboard):

```bash
optuna-dashboard sqlite:///logs/optuna/hyperopt.db
```

This will launch the optuna-dashboard service on port :8080 and can be viewed by going to [localhost:8080/](http://localhost:8080/) in your browser.


As hyperparameter optimization will usually involve creating a large amount of model training runs, it is advisable to run it on a slurm cluster, or otherwise configure runs to work in parallel. 

<br>

## 5. Evaluating a trained model

After we have trained a model we can run it on new data to evaluate how well it performs. This is what the [src/eval.py](src/eval.py) script is intended for. Just as the [src/train.py](src/train.py) script has the [configs/train.yaml](configs/train.yaml) configuration file, there is a [configs/eval.yaml](configs/eval.yaml) configuration file. This one is much simpler, mainly consisting of configuration of the eval namespace [configs/eval](configs/eval):

```yaml
# @package _global_

defaults:
  - eval: default.yaml

# passing model_dir is necessary for evaluation. Configuration and model is loaded from this directory.
# model_dir is either a path relative to content root (e.g. logs/train/runs/YYYY-MM-DD_HH-MM-SS) or a full absolute path
model_dir: ???
# if the model is a pytorch model, one can provide the name of the checkpoint to load weights from.
# the special value "best" will look for a checkpoint matching the default pattern for best checkpoints (epoch_xxx.ckpt)
ckpt: "best"

task_name: "eval"
tags: ["dev"]

eval:
  eval_split: "val"  # which dataset split to evaluate on. One of [test, val]

# The config loaded from the log folder can be overridden by the supplying new configuration, e.g.:
#datamodule:
#  train_val_test_split: some new split
```


By default it configures eval with the following arguments from [configs/eval/default.yaml](configs/eval/default.yaml):

```yaml
eval_split: "val"  # which dataset split to evaluate on. One of [test, val]
eval_runner: null  # which method to use to perform evaluation. One of [backtest, trainer, null] (null will choose trainer for pytorch and backtest if not)
#The default backtest works for all models, but pytorch models can achieve substantial speed improvements by using trainer
eval_kwargs:  # keyword arguments passed to the backtest eval runner. See backtest documentation
  verbose: True
  #retrain: True  # This argument is already set to False for Global models and True for Locals (which require True). Use this argument here to override.
  metric:
    - _target_: darts.metrics.metrics.mse
      _partial_: True
  forecast_horizon: 1
  stride: 1
  start: null

mc_dropout: False
plot_predictions: True
log_metrics: True
```
### Eval split and runner
The eval_split argument chooses which dataset split configured for the datamodule to evaluate the model on. Eval_runner chooses which method to run the evaluation: darts or pytorch lightning ([soon deprecated](https://github.com/unit8co/darts/issues/1531)), the latter gives large speed improvements for pytorch models. The eval_kwargs argument is passed to the [darts evaluator](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.arima.html#darts.models.forecasting.arima.ARIMA.backtest) to control its behaviour.

### Evaluation behaviour
Evaluation is performed by starting from the datapoint controlled by the start argument, making one prediction of length argument forecast_horizon steps ahead, then stepping forward by argument stride datapoints, make the next prediction and so on until the dataset is exhausted. The retrain argument controls if the model is retrained on the data from start of dataset until current prediction point before every prediction. For local models (those that inherit from [configs/model/base_local.yaml](configs/model/base_local.yaml), it is also stated in the darts model documentation, e.g. [the ARIMA model](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.arima.html) inherits from TransferableFutureCovariates**Local**ForecastingModel) this is required as it is how they operate, while for pytorch models and regression models this is optional (set to False by default for these models).

### Metrics
For each prediction, the metrics defined in the eval_kwargs.metric argument are calculated and then averaged over all predictions. These metric values are saved in run directory/eval_<eval_split>_results.yaml. Further, if forecast_horizon is 1 and the plot_predictions argument is True the predictions are concatenated in time and showed alongside the true values in figures to the configured loggers and additionally saved in the run directory/plots directory. If you want to predict more than 1 step ahead and get prediction visualizations, see the [section on prediction](#5-predicting-with-a-trained-model).

### Pytorch models: checkpoints and dropout
The checkpoint argument controls which model weights are used for pytorch models. "best" will select the checkpoint with the weights that gave the best validation epoch during training, otherwise the value should be the name of a checkpoint file found in run directory/checkpoints, e.g. "last.ckpt". Further, if the mc_dropout argument is set to False dropout will be disabled for models where this is used.

### Inverse data transformation
For the darts evaluator configured in [configs/eval/backtest.yaml](configs/eval/backtest.yaml), one can additionally choose to inverse transform the data before plotting and calculating metric values using the argument inverse_transform_data. In this way one can inspect the predictions in the original untransformed space, and compare metric scores for models with different transformation pipelines. The partial_ok argument signals if it is acceptable that not all transformers are inverted, e.g. a NaN-filler is not invertible. Most of the time this should be fine. 

```yaml
eval_runner: "backtest"  # which method to use to perform evaluation. One of [backtest, trainer, null] (null will choose trainer for pytorch and backtest if not)
inverse_transform_data:
  partial_ok: True
```

### Running evaluation

Lastly, there is the required argument model_dir. The model_dir argument should point to a [run directory](#run-directory) for a model trained with [src/train.py](src/train.py). The program state used to train the model can then be exactly recreated from the saved hydra configuration file.

```bash
python src/eval.py model_dir=logs/train/runs/run_name
```

After running evaluation, you can inspect how the model did by looking at metric results or the prediction plots. To compare different models the mlflow logger is a powerful tool, [see the section on mlflow ui above to see how it works](#loggers). For instance being able to filter for models with the same eval_kwargs.forecast_horizon is convenient as metric scores are not necessarily comparable across different forecast_horizons etc. 

Now you can try to train models with different configurations that we have explored above and compare how well they perform using the [mlflow ui](#loggers).

For instance, use the [XGBoost model](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.xgboost.html) and train the following models using the [configuration option](#2-model-training) of your choice:
1. No data transformations
<details>
<summary><b>Solution</b></summary>
Modify configs/datamodule/example_airpassengers.yaml as follows:

```yaml
defaults:
  - base_darts_example
  - _self_

dataset_name: "AirPassengers"
train_val_test_split:
  train: 0.75
  val: 0.25
  test: null
processing_pipeline: null
```

then run

```bash
python src/train.py model=xgboost datamodule=example_airpassengers
```

</details>

2. StandardScaler transformation

<details>
<summary><b>Solution</b></summary>
Modify configs/datamodule/example_airpassengers.yaml as follows:

```yaml
defaults:
  - base_darts_example
  - _self_

dataset_name: "AirPassengers"
train_val_test_split:
  train: 0.75
  val: 0.25
  test: null
processing_pipeline:
  _target_: darts.dataprocessing.Pipeline
  transformers:
    - _target_: darts.dataprocessing.transformers.Scaler
      scaler:
        _target_: sklearn.preprocessing.StandardScaler
    - _target_: darts.dataprocessing.transformers.missing_values_filler.MissingValuesFiller
      fill: "auto"  # The default, will use pandas.Dataframe.interpolate()
```

then run

```bash
python src/train.py model=xgboost datamodule=example_airpassengers
```

</details>

3. Diff and StandardScaler transformation


<details>
<summary><b>Solution</b></summary>
Modify configs/datamodule/example_airpassengers.yaml as follows:

```yaml
defaults:
  - base_darts_example
  - _self_

dataset_name: "AirPassengers"
train_val_test_split:
  train: 0.75
  val: 0.25
  test: null
processing_pipeline:
  _target_: darts.dataprocessing.Pipeline
  transformers:
    - _target_: darts.dataprocessing.transformers.Diff
      lags: 1
    - _target_: darts.dataprocessing.transformers.Scaler
      scaler:
        _target_: sklearn.preprocessing.StandardScaler
    - _target_: darts.dataprocessing.transformers.missing_values_filler.MissingValuesFiller
      fill: "auto"  # The default, will use pandas.Dataframe.interpolate()
```

then run

```bash
python src/train.py model=xgboost datamodule=example_airpassengers
```

</details>

4. (3.) and add the [month encoder](#encoders)

<details>
<summary><b>Important note about using covariates with XGBoost</b></summary>
While the add_encoder argument adds the encoder to the model, it doesn't actually tell the model to use the encoded covariates. To enable processing of covariates we have to use the lags_future_covariates argument to <a href="https://unit8co.github.io/darts/generated_api/darts.models.forecasting.xgboost.html">XGBoost</a> in the configs/model/xgboost.yaml file. Further, it expects a tuple, which is not supported by Hydra, so we have to do some magic with the _target_ keyword and specify we want to create a builtins.tuple object where the first element of the tuple is how far back in time we consider the covariate (should probably be equal to lags) and the second element says how far into the future (should probably be equal to output_chunk_length).

```yaml
lags_future_covariates:
  _target_: builtins.tuple
  _args_:
    - - 4
      - 1
```

In fact, since we probably want to set the tuple values based on the other arguments, we can use a fancy feature of [hydra called variable interpolation](https://hydra.cc/docs/patterns/specializing_config/) to automatically set these values based on the values of the other arguments:

```yaml
defaults:
  - base_nontorch

_target_: darts.models.forecasting.xgboost.XGBModel
lags: 4
output_chunk_length: 1
lags_future_covariates:
  _target_: builtins.tuple
  _args_:
    - - ${model.lags}
      - ${model.output_chunk_length}
add_encoders:
  cyclic:
    future: ["month"]
```

</details>

<details>
<summary><b>Solution</b></summary>
Modify configs/datamodule/example_airpassengers.yaml as follows:

```yaml
defaults:
  - base_darts_example
  - _self_

dataset_name: "AirPassengers"
train_val_test_split:
  train: 0.75
  val: 0.25
  test: null
processing_pipeline:
  _target_: darts.dataprocessing.Pipeline
  transformers:
    - _target_: darts.dataprocessing.transformers.Diff
      lags: 1
    - _target_: darts.dataprocessing.transformers.Scaler
      scaler:
        _target_: sklearn.preprocessing.StandardScaler
    - _target_: darts.dataprocessing.transformers.missing_values_filler.MissingValuesFiller
      fill: "auto"  # The default, will use pandas.Dataframe.interpolate()
```

and modify configs/model/xgboost.yaml as follows:

```yaml
defaults:
  - base_nontorch

_target_: darts.models.forecasting.xgboost.XGBModel
lags: 4
output_chunk_length: 1
lags_future_covariates:
  _target_: builtins.tuple
  _args_:
    - - ${model.lags}
      - ${model.output_chunk_length}
add_encoders:
  cyclic:
    future: ["month"]
```

then run

```bash
python src/train.py model=xgboost datamodule=example_airpassengers
```

</details>

5. (3.) and set model.lags = 12 (or 4 if it was already 12)

<details>
<summary><b>Solution</b></summary>
Modify configs/datamodule/example_airpassengers.yaml as follows:

```yaml
defaults:
  - base_darts_example
  - _self_

dataset_name: "AirPassengers"
train_val_test_split:
  train: 0.75
  val: 0.25
  test: null
processing_pipeline:
  _target_: darts.dataprocessing.Pipeline
  transformers:
    - _target_: darts.dataprocessing.transformers.Diff
      lags: 1
    - _target_: darts.dataprocessing.transformers.Scaler
      scaler:
        _target_: sklearn.preprocessing.StandardScaler
    - _target_: darts.dataprocessing.transformers.missing_values_filler.MissingValuesFiller
      fill: "auto"  # The default, will use pandas.Dataframe.interpolate()
```

and modify configs/model/xgboost.yaml as follows:

```yaml
defaults:
  - base_nontorch

_target_: darts.models.forecasting.xgboost.XGBModel
lags: 12
output_chunk_length: 1
lags_future_covariates:
  _target_: builtins.tuple
  _args_:
    - - ${model.lags}
      - ${model.output_chunk_length}
add_encoders:
  cyclic:
    future: ["month"]
```

then run

```bash
python src/train.py model=xgboost datamodule=example_airpassengers
```

</details>


Then open [mlflow ui](#loggers), add metrics.val_mse as a column and compare the models you have trained. You can also click on the model name to quantitatively inspect the model performance through the prediction plots. 

<br>

## 6. Predicting with a trained model

The prediction pipeline using the script [src/predict.py](src/predict.py) can be used to generate predictions beyond the end of the datasets, or to visualize predictions with forecast_horizon > 1. Similar to the evaluation script, the [configs/predict.yaml](configs/predict.yaml) mainly consists of configuration of the predict namespace and the model_dir argument:

```yaml
# @package _global_

# passing model_dir is necessary for prediction. Configuration and model is loaded from this directory.
# model_dir is either a path relative to content root (e.g. logs/train/runs/YYYY-MM-DD_HH-MM-SS) or a full absolute path
model_dir: ???
# if the model is a pytorch model, one can provide the name of the checkpoint to load weights from.
ckpt: last.ckpt

task_name: "predict"
tags: ["dev"]

predict:
  split: "val"  # which dataset split to predict on. One of [train, test, val]
  plot_encodings: True
  presenter: "savefig"
  inverse_transform_data:
    partial_ok: True
  metric:
    - _target_: darts.metrics.metrics.mse
      _partial_: True
  indices:
    - 0.0
    - 1.0
  kwargs:
    n: 5
```

### Predict split
The predict.split argument controls which dataset configured for the datamodule to predict from. 

### Inverse data transformation

See [evaluation section](#inverse-data-transformation).

### Metrics

See [evaluation section](#metrics)

### Indices
The datapoint indices from which to predict. These indices can be floats (looks up that percentile data point), int (absolute index), or timestamp (if the dataset is indexed with datetime). For each index, the model will predict n steps ahead, compute metrics for the prediction and generate a prediciton plot.

### Checkpoints

See [evaluation section](#pytorch-models--checkpoints-and-dropout)

### Predicting

The predict.kwargs.n is a required argument that controls how many steps into the future to predict, see [model.predict in darts documentation](https://unit8co.github.io/darts/generated_api/darts.models.forecasting.arima.html#darts.models.forecasting.arima.ARIMA.predict). For the configured data split, one prediction plot will be created for each index set in the predict.indices argument. If plot_encodings is True, then any encoders defined in model.encoders will also be included. These prediction plots will be presented according to the predict.presenter argument, see [src/utils/plotting.py](src/utils/plotting.py) for supported presenters (savefig/show). By default they will be saved in the run directory/plots directory. 

To generate these predictions run:

```bash
python src/predict.py model_dir=logs/train/runs/run_name
```

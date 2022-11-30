'''
https://github.com/optuna/optuna/issues/1883
'''
from cmath import pi
import os
import optuna
import tensorflow as tf
import pdb
import mlflow
import numpy as np
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
from optimization import AbstractOptunaSearch
from managers import CheckpointManager, StudyManager
from sklearn.metrics import recall_score, precision_score, confusion_matrix
from utils.file_system import update_json, load_json
from callbacks import TFKerasPruningCallback
import optuna.visualization as vis
import plotly


class ILSVRCOptunaSearch(AbstractOptunaSearch):
    """Hyperparameter optimization utility for ILSVRC models.
    """

    def __init__(self,
                 name: str,
                 n_trials: int,
                 model: object,
                 pipeline: object,
                 model_config: dict,
                 pipeline_config: dict,
                 pruners: bool = False,
                 objective: str = "val_loss",
                 fine_tuning_iterations: int = 1) -> None:
        """Initialize the class and create the persistent directory. Works on Linux only because of storage Path

        Args:
            name (str): Optimization run name. Will be name of directory.
            model (object): Uninstanciated model class
            n_trials: (int): Number of trials per run
            pipeline (object): Uninstanciated pipeline class
            model_config (dict): Configuration of the model as kwargs
            pipeline_config (dict): Configuration of the pipeline as kwargs
            objective (str): name of the objective value
        """
        super().__init__(name, n_trials, model, pipeline, model_config, pipeline_config)
        self.objective_name = objective
        self.pruners = pruners
        self.fine_tuning_iterations = fine_tuning_iterations

    def optimize(self, searcher: object, train_ds: tf.data.Dataset,
                 val_ds: tf.data.Dataset) -> dict:
        """Initiates and documents run performed by the searcher object.

        Args:
            searcher (object): Carries out 
            train_ds (tf.data.Dataset): _description_
            val_ds (_type_): _description_

        Returns:
            dict: _description_
        """
        #TODO! move to searcher init
        searcher.manager.unlock_data_base()
        if self.objective_name == "val_loss":
            direction = "minimize"
        else:
            direction = "maximize"

        study = optuna.create_study(study_name=f"{searcher.name}",
                                    direction=direction,
                                    storage=f'sqlite:////{searcher.manager.db_file}',
                                    load_if_exists=True)

        # Give the study manager the possibility to interrupt
        searcher.study = study
        # TODO! until here
        self.print_summary(searcher, study)
        if not searcher.manager.finished:
            self.visualize(searcher, study, searcher.visualization_params)
            searcher = self.run_trial(searcher, train_ds, val_ds)
            self.visualize(searcher, study, searcher.visualization_params)
            searcher.manager.finished = True

        searcher.copy_winner()

        best_trial = searcher.study.best_trial
        best_params = searcher.manager.best_params
        best_value = self.print_best_trial(searcher, best_params)

        return best_trial.params, best_value

    def run_trial(self, searcher: object, train_ds: tf.data.Dataset,
                  val_ds: tf.data.Dataset) -> object:
        """Runs the searchers study attribute with the objective function and methods provided
           by the searcher.

        Args:
            searcher (object): Custom searcher class
            train_ds (tf.data.Dataset): Train dataset
            val_ds (tf.data.Dataset): Validation datset

        Returns:
            object: updated searcher object
        """
        objective_function = lambda trial: searcher.objective(trial, train_ds, val_ds)
        """
        mlflow_callback = optuna.integration.MLflowCallback(
            tracking_uri=f"file:///{str(searcher.mlflow_path)}",
            metric_name="accuracy",
        )
        """
        searcher.study.optimize(objective_function, n_trials=self.n_trials, gc_after_trial=True)
        self.print_summary(searcher, searcher.study)

        return searcher

    def print_best_trial(self, searcher: object, best_trial: dict) -> None:
        """Prints relevant information of the winning trial

        Args:
            best_trial (dict): Optuna best trial dictionary
        """
        best_trial_name = searcher.manager.best
        best_value = searcher.manager.history[best_trial_name]
        print(f"Winner is: {best_trial_name}")
        print(f"  Value: {best_value}")

        print("  Params: ")
        for key, value in best_trial.items():
            print("    {}: {}".format(key, value))

        return best_value

    def print_summary(self, searcher: object, study: optuna.study.Study):
        """Prints and save summary of executed trials.

        Args:
            searcher (object): Custom searcher class
            study (optuna.study.Study): Optuna study object
        """
        summary_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
        print(summary_df)
        summary_df.to_csv(Path(searcher.model_path, "summary.csv"))

    def visualize(self,
                  searcher: object,
                  study: optuna.study.Study,
                  subset: list = ["depth", "dropout", "learning_rate", "pooling_type"]):
        """Generates summary describing optuna plots.

        Args:
            searcher (object): Custom searcher class
            study (optuna.study.Study): Optuna study object
        """
        if not searcher.finished_trials > 1:
            return
        fig = vis.plot_optimization_history(study)
        plotly.offline.plot(fig,
                            filename=str(Path(searcher.model_path, "optimization_history.png")))
        if self.pruners:
            fig = vis.plot_intermediate_values(study)
            plotly.offline.plot(fig,
                                filename=str(Path(searcher.model_path, "intermediate_values.png")))
        try:
            fig = vis.plot_param_importances(study)
            plotly.offline.plot(fig,
                                filename=str(Path(searcher.model_path, "param_importances.png")))
        except:
            print("Weights sum to zero!")

        fig = vis.plot_parallel_coordinate(study, subset)
        plotly.offline.plot(fig,
                            filename=str(Path(searcher.model_path, "plot_parallel_coordinate.png")))
        fig = vis.plot_contour(study, subset)
        plotly.offline.plot(fig, filename=str(Path(searcher.model_path, "plot_contour.png")))
        fig = vis.plot_slice(study, subset)
        plotly.offline.plot(fig, filename=str(Path(searcher.model_path, "plot_slice.png")))
        fig = vis.plot_edf(study)
        plotly.offline.plot(fig, filename=str(Path(searcher.model_path, "plot_edf.png")))

    def convert_config(
        self, name: str, config: dict,
        default_names: tuple = ("feature_extraction", "fine_tuning")) -> dict:
        """Converts generatl optimization dictionary to study specific trial dict, by resolving
           trial specific settings.

        Args:
            name (str): Study name
            config (dict): Configuration
            default_names (tuple, optional): Names of all to be executed study as iterable. Defaults to ("feature_extraction", "fine_tuning").

        Returns:
            dict: converted dictionary config
        """
        remainders = set(default_names) - set(name)
        converted_config = deepcopy(config)
        for key, value in config.items():
            if isinstance(value, dict):
                if name in value:
                    converted_config[key] = value[name]
                    continue
                for remainder in remainders:
                    if remainder in value:
                        del converted_config[key]
                        break

        return converted_config

    def run(self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset) -> dict:
        """Runs all implemented studies and returns best configuration.

        Args:
            train_ds (tf.data.Dataset): Train dataset
            val_ds (tf.data.Dataset): Validation dataset

        Returns:
            dict: best trial
        """
        mlflow.tensorflow.autolog()

        model_config = self.convert_config("feature_extraction", self.model_config)
        pipeline_config = self.convert_config("feature_extraction", self.pipeline_config)
        fe_searcher = FeatureExtractionOptunaSearch(self.name, self.model, self.pipeline,
                                                    model_config, pipeline_config, self.n_trials,
                                                    self.objective_name, self.pruners)

        best_trial, predecessor_value = self.optimize(fe_searcher, train_ds, val_ds)
        save_trial = deepcopy(best_trial)
        predecessor = "feature_extraction"
        for i in range(self.fine_tuning_iterations):
            name = f"fine_tuning{i}"
            predecessor_path = Path(self.model_path, f"winner_{predecessor}")
            model_config = self.convert_config("fine_tuning", self.model_config)
            pipeline_config = self.convert_config("fine_tuning", self.pipeline_config)
            ft_searcher = FineTuningOptunaSearch(name, self.name, self.model, self.pipeline,
                                                 model_config, pipeline_config, self.n_trials,
                                                 best_trial, predecessor_path, predecessor_value,
                                                 self.objective_name, self.pruners)
            best_trial, best_value = self.optimize(ft_searcher, train_ds, val_ds)
            best_trial.update(save_trial)
            if predecessor_value < best_value:
                # TODO! works only with val_loss
                print("Optimization gain decreased")
                break

            predecessor = name
            predecessor_value = best_value

        return best_trial


class ILSVRCAbstractStepSearch(AbstractOptunaSearch):
    """Abstract step searcher class. Implements methods common to feature extraction
       and fine tuning optimization.
    """

    def __init__(self, name: str, run_name: str, model: object, pipeline: object,
                 model_config: dict, pipeline_config: dict, n_trials: int, pruners: bool) -> None:
        """
        Initialize attributes an MLFlow.

        Args:
            name (str): Study name
            run_name (str): Name of overall run
            model (object): Uninstanciated model object
            pipeline (object): Uninstanciated pipeline object
            model_config (dict): Model configuration dictionary
            pipeline_config (dict): Pipeline configuration dictionary
            n_trials (int): Number of trials to generate
        """
        self.name = name
        self.run_name = run_name
        self.model = model
        self.pipeline = pipeline
        self.model_config = model_config
        self.pipeline_config = pipeline_config
        self.pipeline_config["save_weights_only"] = True
        self.n_trials = n_trials
        self.model = model
        self.manager = None
        self._study = None
        self._visualization_params = []
        self.pruners = pruners
        self.model_path = Path(os.getenv("MODEL"), f"{str(self.run_name)}", self.name)
        self.mlflow_path = Path(f"{self.model_path}/mlflow")
        self.mlflow_path.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"file:///{str(self.mlflow_path)}")
        self.set_experiment(self.mlflow_path)

    def set_experiment(self, path: Path) -> None:
        """Set MLFlow experiment

        Args:
            path (Path): MLFlow URI path
        """
        exp_name = f"{self.run_name}_{self.name}"
        if not mlflow.get_experiment_by_name(exp_name):
            self.experiment = mlflow.create_experiment(exp_name, artifact_location=path.as_uri())
        else:
            self.experiment = mlflow.get_experiment_by_name(exp_name)
        self.experiment = mlflow.set_experiment(exp_name)
        self.experiment_id = self.experiment.experiment_id

        return

    def suggest_model(self, trial: optuna.trial.Trial, model_config: dict) -> tuple:
        """ Create the model for the current trial iteration during the optimization run.
            Also allows for the learning rate tuning, but only with ADAM optimizer.

        Args:
            trial (object): Optuna trial object

        Returns:
            tuple: Configurated Model, Configurated Optimizer
        """
        # We optimize the numbers of layers, their units and weight decay parameter.
        model_config = self.make_suggestions(trial, model_config)

        if "learning_rate" in model_config.keys():
            learning_rate = model_config["learning_rate"]
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            self.manager.trial.learning_rate = learning_rate
            del model_config["learning_rate"]
        else:
            optimizer = None

        model = self.model(**model_config)
        self.manager.trial.model_config = model_config

        return model, optimizer

    def make_suggestions(self,
                         trial: optuna.trial.Trial,
                         config_json: dict,
                         exceptions: list = []) -> dict:
        """Convert the config jsons to trial kwargs for the optimization run.

        Args:
            trial (object): Optuna trial object
            config_json (dict): Configuration JSON
        
        Returns:
            dict: converted kwargs
        """
        suggestor_switch = {
            "categorical": trial.suggest_categorical,
            "discrete_uniform": trial.suggest_discrete_uniform,
            "float": trial.suggest_float,
            "int": trial.suggest_int,
            "loguniform": trial.suggest_loguniform,
            "uniform": trial.suggest_uniform
        }
        config_json = deepcopy(config_json)
        converted_json = deepcopy(config_json)
        for name, config in config_json.items():
            if name in exceptions:
                continue
            if not isinstance(config, dict):
                continue
            if not "type" in config.keys():
                continue
            if name == "layer_size":
                continue

            suggestion_type = config["type"]
            del config["type"]
            suggested_value = suggestor_switch[suggestion_type](name, **config)
            converted_json[name] = suggested_value
            mlflow.log_param(name, suggested_value)
            self.manager.trial.params = {name: suggested_value}

        if ("layer_size" in config_json.keys()) and (isinstance(
                config_json["layer_size"], dict)) and ("layer_size" not in exceptions):
            # Different logic for layer size
            suggestion_type = config_json["layer_size"]["type"]
            del config_json["layer_size"]["type"]

            layer_size_value = list()
            for layer in range(converted_json["depth"]):
                layer_name = f"layer_{layer}"
                layer_value = suggestor_switch[suggestion_type](layer_name,
                                                                **config_json["layer_size"])
                layer_size_value.append(layer_value)
                self.manager.trial.params = {layer_name: layer_value}

            converted_json["layer_size"] = layer_size_value
            mlflow.log_param("layer_size", layer_size_value)

        return converted_json

    def log_data(self, pipeline: object, history: dict, val_ds: tf.data.Dataset) -> None:
        """Logging data to history file and MLFlow

        Args:
            pipeline (object): Trained pipeline
            history (dict): generated history dictionary
            val_ds (tf.data.Dataset): Validation dataset
        """
        outer_shape = len(val_ds.element_spec)
        # Consider weights in dataset
        spliter_switch = {2: lambda x, y: (x, y), 3: lambda x, y, z: (x, y)}
        spliter_switch = spliter_switch[outer_shape]
        val_ds = val_ds.batch(10)
        val_data = [tuples for tuples in val_ds.map(spliter_switch).as_numpy_iterator()]
        val_labels = np.array([label for tuple in val_data for label in tuple[1]])
        val_samples = np.array([label for tuple in val_data for label in tuple[0]])
        pred = pipeline.predict_class(val_samples)

        metrics = {
            "val_loss": min(history["val_loss"]),
            "val_accuracy": max(history["val_accuracy"]),
            "recall_micro": recall_score(val_labels, pred, average='micro'),
            'recall_macro': recall_score(val_labels, pred, average='macro'),
            'precision': precision_score(val_labels, pred, average='micro')
        }

        for identifier, value in metrics.items():
            mlflow.log_metric(identifier, value)

        conf_matrix = confusion_matrix(val_labels, pred)
        mlflow.log_param("confusion_matrix", conf_matrix)
        self.manager.trial.confusion_matrix = conf_matrix.tolist()

        return

    def copy_winner(self) -> None:
        """Copys winner directory to parent directory
        """
        if self.finished_trials < self.n_trials:
            return

        winner = self.manager.best
        winner_location = Path(self.model_path, winner)
        target_location = Path(self.model_path.parent, f"winner_{self.name}")
        if target_location.is_dir():
            return

        print(f"Creating winner directory for {self.name} at {str(target_location)}")
        shutil.copytree(str(winner_location), str(target_location))

        return

    @property
    def study(self) -> optuna.study.Study:
        """Return study object

        Returns:
            optuna.study.Study: study object
        """
        return self._study

    @study.setter
    def study(self, value: optuna.study.Study) -> None:
        """Sets study object

        Args:
            value (optuna.study.Study)
        """
        self._study = value

    @property
    def visualization_params(self) -> list:
        """Return study object

        Returns:
            optuna.study.Study: study object
        """
        return self._visualization_params

    @property
    def finished_trials(self) -> list:
        """Return study object

        Returns:
            optuna.study.Study: study object
        """
        return len([
            trial for trial in self.study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE or
            trial.state == optuna.trial.TrialState.PRUNED
        ])


class FeatureExtractionOptunaSearch(ILSVRCAbstractStepSearch):
    """Feature extraction step searcher class. Implements feature extraction optimization.
    """

    def __init__(self,
                 run_name: str,
                 model: object,
                 pipeline: object,
                 model_config: dict,
                 pipeline_config: dict,
                 n_trials: int,
                 objective: str = "val_loss",
                 pruners: bool = False) -> None:
        """
        Initialize attributes an MLFlow.

        Args:
            run_name (str): Name of overall run
            model (object): Uninstanciated model object
            pipeline (object): Uninstanciated pipeline object
            model_config (dict): Model configuration dictionary
            pipeline_config (dict): Pipeline configuration dictionary
            n_trials (int): Number of trials to generate
        """
        super().__init__("feature_extraction", run_name, model, pipeline, model_config,
                         pipeline_config, n_trials, pruners)
        self.objective_name = objective
        self.pruners = pruners
        self.manager = StudyManager(run_name, self.model_path, objective)
        self._visualization_params = [
            "depth", "dropout", "learning_rate", "pooling_type", "batch_size"
        ]

    def suggest_pipeline_config(self, trial: optuna.trial.Trial, optimizer: object) -> dict:
        """Create trial pipeline configuration

        Args:
            trial (optuna.trial.Trial): Current tiral object
            optimizer (object): Current optimizer object

        Returns:
            dict: Current trial configurartion.
        """
        if self.manager.trial.pipeline_config is not None:
            return self.manager.trial.pipeline_config

        pipeline_config = self.make_suggestions(trial, self.pipeline_config)
        pipeline_config["model_name"] = self.manager.trial.path.name
        pipeline_config["root_path"] = str(self.manager.trial.path.parent)

        self.manager.trial.pipeline_config = pipeline_config

        if optimizer:
            pipeline_config["compiler_config"]["optimizer"] = optimizer

        if self.pruners:
            if "patience" in pipeline_config:
                patience = pipeline_config["patience"]
            else:
                patience = None
            pruning_callback = TFKerasPruningCallback(trial, self.objective_name, patience)
            if "callbacks" in pipeline_config:
                pipeline_config["callbacks"].append(pruning_callback)
            else:
                pipeline_config["callbacks"] = [pruning_callback]

        return pipeline_config

    def objective(self, trial: object, train_ds, val_ds) -> int:
        """Optimization of the top network. Currently only 'best validation set performance' available as objective. 

        Args:
            trial (object): Optuna trial object
            train_ds (_type_): Training dataset
            val_ds (_type_): Validation dataset

        Returns:
            int: objective
        """
        if self.finished_trials > self.n_trials:
            # Overshooting required trials
            self.manager.finished = True
            self.study.stop()
            return

        with mlflow.start_run(experiment_id=self.experiment_id):
            mlflow.log_param("trial_id", trial.number)
            model, optimizer = self.suggest_model(trial, self.model_config)
            pipeline_config = self.suggest_pipeline_config(trial, optimizer)

            pipeline = self.pipeline(model, **pipeline_config)
            history = pipeline.fit(train_ds, val_ds)

            if "store_model" in pipeline_config.keys():
                if not pipeline_config["store_model"]:
                    self.manager.trial.history = history

            # Return last validation accuracy.
            val_loss = self.manager.trial.best
            self.manager.history = val_loss
            self.log_data(pipeline, history, val_ds)
            if self.manager.is_best:
                pipeline.save_model()
            self.manager.counter += 1

        return val_loss


class FineTuningOptunaSearch(ILSVRCAbstractStepSearch):
    """Fine tuning step searcher class. Implements fine tuning optimization.
    """

    def __init__(self,
                 study_name: str,
                 run_name: str,
                 model: object,
                 pipeline: object,
                 model_config: dict,
                 pipeline_config: dict,
                 n_trials: int,
                 predecessor_params: dict,
                 predecessor_path: Path,
                 predecessor_value: float,
                 objective: str = "val_loss",
                 pruners: bool = False):
        """
        Initialize attributes an MLFlow.

        Args:
            run_name (str): Name of overall run
            model (object): Uninstanciated model object
            pipeline (object): Uninstanciated pipeline object
            model_config (dict): Model configuration dictionary
            pipeline_config (dict): Pipeline configuration dictionary
            n_trials (int): Number of trials to generate
        """
        super().__init__(study_name, run_name, model, pipeline, model_config, pipeline_config,
                         n_trials, pruners)
        self.predecessor_params = predecessor_params
        if not predecessor_params:
            raise ValueError
        self.objective_name = objective
        self.pruners = pruners
        predecessor_params = load_json(Path(predecessor_path, "configuration.json"))["params"]
        self.manager = StudyManager(run_name, self.model_path, objective, predecessor_params)
        self.predecessor_path = predecessor_path
        self.predecessor_value = predecessor_value
        self.model_config = self.preprocess_model_config(exceptions=["learning_rate"])
        self.pipeline_config = self.preprocess_pipeline_config(exceptions=["batch_size"])
        self._visualization_params = ["learning_rate", "batch_size", "trainable_blocks"]

    def preprocess_pipeline_config(self, exceptions: list = []) -> dict:
        """Preprocess pipeline config for fine tuning run.

        Args:
            exceptions (list, optional): Key word to be left out. Defaults to [].

        Returns:
            dict: preprocessed pipeline
        """
        pipeline_config = {
            key: (value if (key not in self.predecessor_params.keys()) or
                  (key in exceptions) else self.predecessor_params[key])
            for key, value in self.pipeline_config.items()
        }
        pipeline_config["model_source"] = str(self.predecessor_path)
        pipeline_config["save_weights_only"] = True
        pipeline_config["baseline"] = self.predecessor_value

        return pipeline_config

    def preprocess_model_config(self, exceptions=[]) -> dict:
        """Preprocess model config for fine tuning run.

        Args:
            predecessor_params (_type_): _description_
            exceptions (list, optional): _description_. Defaults to [].

        Returns:
            dirct: preprocessed model config
        """
        if "trainable_blocks" in self.predecessor_params:
            self.high = self.predecessor_params["trainable_blocks"]
        else:
            self.high = None

        if not "layer_size" in self.predecessor_params:
            self.predecessor_params["layer_size"] = [
                value for key, value in self.predecessor_params.items() if key.startswith("layer_")
            ]

        model_config = {
            key: (value if (key not in self.predecessor_params) or
                  (key in exceptions) else self.predecessor_params[key])
            for key, value in self.model_config.items()
        }

        return model_config

    def suggest_pipeline_config(self, trial: optuna.trial.Trial, optimizer: object) -> dict:
        """Suggest pipeline configuration for current trial

        Args:
            trial (optuna.trial.Trial): Current trial
            optimizer (object): suggested optimizer

        Returns:
            dict: Suggested pipeline config
        """
        pipeline_config = deepcopy(self.pipeline_config)
        pipeline_config["model_name"] = self.manager.trial.path.name
        pipeline_config["root_path"] = str(self.manager.trial.path.parent)
        pipeline_config = self.make_suggestions(trial, pipeline_config)

        self.manager.trial.pipeline_config = pipeline_config

        if optimizer:
            pipeline_config["compiler_config"]["optimizer"] = optimizer

        if self.pruners:
            if "patience" in pipeline_config:
                patience = pipeline_config["patience"]
            else:
                patience = None
            pruning_callback = TFKerasPruningCallback(trial, self.objective_name, patience)
            if "callbacks" in pipeline_config:
                pipeline_config["callbacks"].append(pruning_callback)
            else:
                pipeline_config["callbacks"] = [pruning_callback]

        return pipeline_config

    def objective(self, trial: object, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset) -> float:
        """Optimization of the entire network. Currently only 'best validation set performance' available as objective. 

        Args:
            trial (object): Optuna trial object
            top_params (dict): best top network parameters
            train_ds (tf.data.Dataset): Training dataset
            val_ds (tf.data.Dataset): Validation dataset

        Returns:
            float: objective
        """
        if self.finished_trials > self.n_trials:
            # Overshooting required trials
            self.manager.finished = True
            self.study.stop()
            return

        with mlflow.start_run(experiment_id=self.experiment_id):
            # Suggests learning rate
            model, optimizer = self.suggest_model(trial, deepcopy(self.model_config))
            pipeline_config = self.suggest_pipeline_config(trial, optimizer)

            # Suggest trainable blocks
            if self.high is None:
                self.high = model.num_blocks
            blocks = trial.suggest_int("trainable_blocks", low=0, high=self.high)
            model.trainable_blocks = blocks
            self.manager.trial.trainable_blocks = model.trainable_blocks
            pipeline = self.pipeline(model, **pipeline_config)
            print(pipeline.score(val_ds.batch(10)))
            history = pipeline.fit(train_ds, val_ds)
            if "store_model" in pipeline_config.keys():
                if not pipeline_config["store_model"]:
                    self.manager.trial.history = history
            # Return last validation accuracy.
            val_loss = self.manager.trial.best
            self.manager.history = val_loss
            self.log_data(pipeline, history, val_ds)
            if self.manager.is_best:
                pipeline.save_model()
            self.manager.counter += 1

        return val_loss
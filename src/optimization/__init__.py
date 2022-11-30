import os
import optuna
import tensorflow as tf
from pathlib import Path
from copy import deepcopy


class AbstractOptunaSearch():
    """ Interface for hyperaparameter search with optuns. Notably, the make converted dict method
        allows to use a uniform hyperparameter serach configuration file format.
    """

    def __init__(self, name: str, n_trials: int, model: object, pipeline: object,
                 model_config: dict, pipeline_config: dict) -> None:
        """Initialize the class and create the persistent directory.

        Args:
            name (str): Optimization run name. Will be name of directory.
            model (object): Uninstanciated model class
            pipeline (object): Uninstanciated pipeline class
            model_config (dict): Configuration of the model as kwargs
            pipeline_config (dict): Configuration of the pipeline as kwargs
        """
        self.name = name
        self.n_trials = n_trials
        self.model = model
        self.pipeline = pipeline
        self.model_config = model_config
        self.pipeline_config = pipeline_config
        self.model_path = Path(os.getenv("MODEL"), name)
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.counter = 0

    def create_model(self, trial: object):
        """ Create the model and trial configuration during the optimization run.

        Args:
            trial (object): _description_

        Returns:
            object: _description_
        """
        raise NotImplementedError
        return model, optimizer

    def objective(self, trial, timeseries_df):
        """_summary_

        Args:
            trial (_type_): _description_

        Returns:
            _type_: _description_
        """
        raise NotImplementedError
        return min(history['val_loss'])

    def make_converted_config(self, trial: object, config_json: dict) -> dict:
        """Convert the config jsons to trial kwargs for the optimization run.

        Args:
            trial (object): Optuna trial object
            config_json (dict): Configuration JSON
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
            if not isinstance(config, dict):
                continue
            if not "type" in config.keys():
                continue
            suggestion_type = config["type"]
            del config["type"]
            converted_json[name] = suggestor_switch[suggestion_type](name, **config)

        return converted_json

    def optimize(self, timeseries_df):
        """_summary_

        Args:
            timeseries_df (_type_): _description_

        Returns:
            _type_: _description_
        """
        raise NotImplementedError
        return trial

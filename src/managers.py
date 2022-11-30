from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
from re import search
import shutil
import numpy as np
import pdb
from pathlib import Path
import tensorflow as tf
from utils.file_system import load_json, update_json
from collections import Counter


class CheckpointManager():
    """Manages checkpoints and saved models
    """

    def __init__(self, directory: Path, train_epochs: int, custom_objects: list,
                 save_weights_only: bool) -> None:
        """_summary_

        Args:
            directory (Path): Path to checkpoint directory
            train_epochs (int): Number of epochs per run
            custom_objects (list): Custom objects for restore
            save_weights_only (bool): Flag for save type
        """
        if isinstance(directory, str):
            self.directory = Path(directory)
        else:
            self.directory = directory

        self.epochs = train_epochs
        self.custom_objects = custom_objects
        self.save_weights_only = save_weights_only

    @property
    def latest(self) -> int:
        """Returns latest epoch
        """
        return self.latest_epoch()

    def load_model(self) -> tf.keras.Model:
        """Loads full model

        Returns:
            tf.keras.Model: Restored model
        """
        model_path = Path(self.directory, f"cp-{self.latest:04d}.ckpt")
        print(f"Loading model from epoch {self.latest}.")
        model = tf.keras.models.load_model(model_path, self.custom_objects)

        return model

    def load_weights(self, model: tf.keras.Model) -> tf.keras.Model:
        """Loads weights to specified model

        Args:
            model (tf.keras.Model): Empty model

        Returns:
            tf.keras.Model: Model with restored weights
        """
        latest_cp_name = tf.train.latest_checkpoint(self.directory)
        model.load_weights(latest_cp_name)
        return model

    def latest_epoch(self):
        """Computes the latest epoch

        Returns:
            int: latest epoch
        """

        check_point_epochs = [
            i for i in range(self.epochs + 1) for folder in self.directory.iterdir()
            if f"{i:04d}" in folder.name
        ]

        if check_point_epochs:
            return max(check_point_epochs)

        return 0

    def clean_directory(self, best_epoch: int, keep_latest: bool = True):
        """Cleans directory from except best epoch

        Args:
            best_epoch (int): best epoch
            keep_latest (bool, optional): Cleans entire dir if false. Defaults to True.
        """
        if self.save_weights_only:
            # TODO! deletes the correct folder for some reason
            latest_cp_name = tf.train.latest_checkpoint(self.directory)
            [
                file.unlink()
                for file in self.directory.iterdir()
                if ((".index" in file.name) or
                    (".data" in file.name)) and not (Path(latest_cp_name).name in file.name)
            ]
        else:
            [
                shutil.rmtree(folder)
                for i in range(self.epochs + 1)
                for folder in self.directory.iterdir()
                if f"{i:04d}" in folder.name and (
                    (i != self.epochs) or not keep_latest) and i != best_epoch
            ]

    def is_empty(self) -> bool:
        """Check if dir contains checkpoints.
        """
        if self.latest == 0:
            return True
        return False

    def save_model(self, path: Path, model: tf.keras.Model, epoch: int) -> None:
        """Save passed model to specified path

        Args:
            path (Path): Directory where checkpoint is to be created
            model (tf.keras.Model): Model to be saved
            epoch (int): Epoch at which to be saved
        """
        checkpoint_path = Path(path, "cp-{epoch:04d}.ckpt")
        checkpoint_path = str(checkpoint_path).format(epoch=epoch)
        path.mkdir(parents=True, exist_ok=True)
        if self.save_weights_only:
            model.save_weights(checkpoint_path)
        else:
            model.save(checkpoint_path)

        return


class HistoryManager():
    """Manages the history of a single run, if run employs the history callback.
    """

    def __init__(self, directory: Path) -> None:
        """Initialize

        Args:
            directory (Path): Same path as specified in checkpoiont
        """
        self.directory = directory
        self.history_file = Path(directory, "history.json")

    @property
    def history(self) -> dict:
        """Read history json.

        Returns:
            dict: history dict
        """
        self._history = load_json(self.history_file)
        return self._history

    @history.setter
    def history(self, value: dict) -> None:
        """Set history

        Args:
            value (dict): history
        """
        self._history = value

    @property
    def best(self) -> tuple:
        """Return best value and best epoch

        Returns:
            tuple: best value, best epoch
        """
        if "val_loss" in self.history.keys():
            return min(self.history["val_loss"]), np.argmin(self.history["val_loss"]) + 1

        return None, None

    def update(self, items: dict) -> None:
        """Update history json

        Args:
            items (dict): update mapping
        """
        self._history = update_json(self.history_file, items)

    def is_finished(self) -> bool:
        """Check if optimization run finished.

        Returns:
            bool: True if finished
        """
        if "finished" in self.history.keys():
            return True
        return False

    def finished(self) -> None:
        """Set run as finished
        """
        self.update({'finished': True})


class TrialManager():

    def __init__(self,
                 number: int,
                 study_root: Path,
                 objective: str = "val_loss",
                 predecessor_params: dict = {}) -> None:
        """Manages values and files for a single optimization trial.

        Args:
            number (int): Trial number as per counter
            study_root (Path): Root of study in which trial is stored
            objective (str, optional): Objective name. Defaults to "val_loss".
        """
        self.name = f"iteration{number:03d}"
        self.study_root = Path(study_root)
        self.path = Path(study_root, self.name)
        self.path.mkdir(parents=True, exist_ok=True)
        self.history_file = Path(self.path, "history.json")
        self.config_file = Path(self.path, "configuration.json")
        self.objective = objective
        if not self.config_file.is_file():
            update_json(self.config_file, {"params": predecessor_params})

    @property
    def finished(self) -> bool:
        """Check if trial is finished.

        Returns:
            bool: True if finished
        """
        if not self.history_file.is_file():
            return False
        if "finished" in load_json(self.history_file).keys():
            return True
        return False

    @property
    def best(self) -> float:
        """Return value of best trial

        Returns:
            float: value of best trial
        """
        if self.objective == "val_loss":
            return min(load_json(self.history_file)[self.objective])
        return max(load_json(self.history_file)[self.objective])

    @property
    def config(self) -> dict:
        """Load trial configuration

        Returns:
            dict: trial configuration
        """
        self._config = load_json(self.config_file)
        return self._config

    @config.setter
    def config(self, value: dict) -> None:
        """Set trial configuration

        Args:
            value (dict): trial configuration

        """
        update_json(self.config_file, value)

    @property
    def learning_rate(self) -> float:
        """Get trial learning rate if exists
        """
        if "learning_rate" in self.config.keys():
            return self.config["learning_rate"]
        return None

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        """Set learning rate

        Args:
            value (float): learning rate
        """
        self.config = {"learning_rate": value}

    @property
    def trainable_blocks(self) -> int:
        """Get number of trainable blocks if exists

        Returns:
            int: number of trainable blocks
        """
        if "trainable_blocks" in self.config.keys():
            return self.config["trainable_blocks"]
        return None

    @trainable_blocks.setter
    def trainable_blocks(self, value: int) -> None:
        """Set number of trainable blocks 

        Args:
            value (int): number of trainable blocks 
        """
        self.config = {"trainable_blocks": value}

    @property
    def params(self) -> dict:
        """Get trial parameters

        Returns:
            int: number of trainable blocks
        """
        if "params" in self.config.keys():
            self._params = self.config["params"]
            if self.study_root.name in self._params:
                return self._params[self.study_root.name]
        return {}

    @params.setter
    def params(self, value: dict) -> None:
        """Update the parameter tracker

        Args:
            value (int): number of trainable blocks 
        """
        if "params" in self.config:
            self._params = self.config["params"]
        else:
            self._params = {}
        if self.study_root.name in self._params:
            self._study_params = self._params[self.study_root.name]
            self._study_params.update(value)
            self._params[self.study_root.name] = self._study_params
        else:
            self._params[self.study_root.name] = value

        self.config = {"params": self._params}

    @property
    def model_config(self) -> dict:
        """Get model config if exists

        Returns:
            dict: model config
        """
        if "model_config" in self.config.keys():
            return self.config["model_config"]
        return None

    @model_config.setter
    def model_config(self, value) -> None:
        """Set model config

        Args:
            value (dict): model config
        """
        self.config = {"model_config": value}

    @property
    def pipeline_config(self) -> dict:
        """Get pipeline config if exists

        Returns:
            dict: pipeline config
        """
        if "pipeline_config" in self.config.keys():
            return self.config["pipeline_config"]
        return None

    @property
    def history(self) -> dict:
        """Return trial history

        Returns:
            dict: trial history
        """
        return load_json(self.history_file)

    @history.setter
    def history(self, value: dict) -> None:
        """Set trial history

        Args:
            value (dict): trial history
        """
        update_json(self.history_file, value)

    @pipeline_config.setter
    def pipeline_config(self, value: dict) -> None:
        """Set pipeline config

        Args:
            value (dict): pipeline config
        """
        self.config = {"pipeline_config": value}

    @property
    def confusion_matrix(self) -> list:
        """Get confusion matrix

        Returns:
            value (list): confusion matrix
        """
        if "confusion_matrix" in self.history:
            return self.history["confusion_matrix"]
        else:
            return None

    @confusion_matrix.setter
    def confusion_matrix(self, value: list) -> None:
        """Set confusion matrix

        Args:
            value (list): confusion matrix
        """
        self.history = {"confusion_matrix": value}


class StudyManager():
    """Manages history, trials and file system for entire study.
    """

    def __init__(self,
                 name: str,
                 study_root: Path,
                 objective: str = "val_loss",
                 predecessor_params: dict = {}) -> None:
        """Initialize manager

        Args:
            name (str): Study name
            study_root (Path): Run path
            objective (str, optional): Objective name. Defaults to "val_loss".
        """
        self.name = name
        self.path = Path(study_root)
        self.objective = objective
        self.path.mkdir(parents=True, exist_ok=True)
        self.search_file = Path(self.path, "search.json")
        self.db_file = Path(self.path, "Optuna.db")
        self.predecessor_params = predecessor_params
        if not load_json(self.search_file):
            update_json(self.search_file, {"history": {}, "finished": False})

    def unlock_data_base(self) -> None:
        """Unlock locked SQL DB
        """
        if self.db_file.is_file():
            unlock_file = Path(self.db_file.parent, "OptunaTop_temp.db")
            shutil.copy(str(self.db_file), str(unlock_file))
            self.db_file.unlink()
            unlock_file.rename(self.db_file)

        return

    @property
    def counter(self) -> int:
        """Get value of counter

        Returns:
            int: counter
        """
        if "counter" in self.search_data.keys():
            self._counter = self.search_data["counter"]
        else:
            self.counter = 0

        return self._counter

    @counter.setter
    def counter(self, value: int) -> None:
        """Set value of counter

        Args:
            value (int): new value
        """
        self._counter = value
        update_json(self.search_file, {"counter": value})

    @property
    def trial(self) -> TrialManager:
        """Return trial manager

        Returns:
            TrialManager: Trial manager for current trial
        """

        self._trial = TrialManager(self.counter, self.path, self.objective, self.predecessor_params)
        return self._trial

    @property
    def search_data(self) -> dict:
        """Get search data

        Returns:
            dict: search data
        """
        self._search_data = load_json(self.search_file)
        return self._search_data

    @search_data.setter
    def search_data(self, value: dict) -> None:
        """Set search data

        Args:
            value (dict): new data
        """
        update_json(self.search_file, value)

    @property
    def finished(self) -> bool:
        """Indicates study state

        Returns:
            bool: True if finished
        """
        if "finished" in self.search_data.keys() and self.search_data["finished"]:
            return True
        return False

    @finished.setter
    def finished(self, value):
        """Indicates study state

        Args:
            bool: finish trial
        """
        self.search_data = {"finished": True}

    @property
    def history(self) -> dict:
        """Return search histroy

        Returns:
            dict: search history
        """
        if "history" in self.search_data:
            self._history = self.search_data["history"]
            return self._history
        else:
            return None

    @history.setter
    def history(self, value: dict) -> None:
        """Update serach history

        Args:
            value (dict): update dict
        """
        self._history = self.search_data["history"]
        self._history.update({self._trial.name: value})
        self.search_data = {"history": self._history}

        if self.is_best:
            if "best" in self.search_data:
                self._best = self.search_data["best"]
                self._best.update({"name": self._trial.name})
                self.search_data = self._best
            else:
                self.search_data = {"name": self._trial.name}

    @property
    def best(self) -> str:
        """Get name of best iteration

        Returns:
            str: best iteration name
        """
        k = Counter(self.history)
        if self.objective == "val_loss":
            return k.most_common()[-1][0]
        else:
            return k.most_common()[0][0]

    @property
    def best_params(self) -> dict:
        """Returns param dict of best trial

        Returns:
            dict: best param
        """
        best_counter = np.argmax(self.history.values())
        trial = TrialManager(best_counter, self.path, self.objective)
        return trial.params

    @property
    def is_best(self) -> bool:
        """True if best

        Returns:
            bool: 
        """
        if self.objective == "val_loss":
            return self.history[self._trial.name] <= min(self.history.values())
        else:
            return self.history[self._trial.name] >= max(self.history.values())


class SettingsManager():

    def __init__(self, path: Path) -> None:
        self.path = path
        self.convert_config = lambda config, params: {
            parameter: params[parameter] if parameter in params else value
            for parameter, value in config.items()
        }

    @property
    def config(self):
        self._config = load_json(Path(self.path, "configuration.json"))
        return self._config

    @property
    def model_config(self) -> dict:
        return self.config["model_config"]

    @property
    def pipeline_config(self) -> dict:
        pipeline_config = self.config["pipeline_config"]

        del pipeline_config["root_path"]
        del pipeline_config["model_name"]
        pipeline_config["save_weights_only"] = True
        pipeline_config["model_source"] = self.path
        return pipeline_config

    @property
    def feature_extraction_config(self) -> dict:
        params = self.config["params"]["feature_extraction"]
        model_config = self.model_config
        model_config = self.convert_config(model_config, params)
        pipeline_config = self.pipeline_config
        pipeline_config = self.convert_config(pipeline_config, params)
        pipeline_config["store_model"] = False
        pipeline_config["save_weights_only"] = True
        if "root_path" in pipeline_config:
            del pipeline_config["root_path"]
        if "model_source" in pipeline_config:
            del pipeline_config["model_source"]
        if "learning_rate" in params:
            learning_rate = params["learning_rate"]
        else:
            learning_rate = False
        return model_config, pipeline_config, learning_rate

    @property
    def fine_tuning_config(self) -> dict:
        params = self.config["params"]["fine_tuning"]
        model_config = self.model_config
        model_config = self.convert_config(model_config, params)

        params = self.config["params"]["fine_tuning"]
        pipeline_config = self.pipeline_config
        pipeline_config = self.convert_config(pipeline_config, params)
        pipeline_config["store_model"] = False
        pipeline_config["save_weights_only"] = True
        if "root_path" in pipeline_config:
            del pipeline_config["root_path"]
        if "model_source" in pipeline_config:
            del pipeline_config["model_source"]
        if "learning_rate" in params:
            learning_rate = params["learning_rate"]
        else:
            learning_rate = False
        return model_config, pipeline_config, learning_rate, params["trainable_blocks"]

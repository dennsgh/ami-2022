"https://stackoverflow.com/questions/69595923/how-to-decrease-the-learning-rate-every-10-epochs-by-a-factor-of-0-9"

import json
from tensorflow.keras import callbacks, backend
import tensorflow.keras.backend as K
from typing import Any
from typing import Dict
from typing import Optional
import warnings

import optuna

with optuna._imports.try_import() as _imports:
    from tensorflow.keras.callbacks import Callback

if not _imports.is_successful():
    Callback = object  # NOQA


class HistoryCheckpoint(callbacks.Callback):
    """
    Stores the history in a JSON file, which is updated at the end of every epoch.
    """

    def __init__(self, storage_path):
        """ Initialize with model storage location.

        Args:
            storage_path (Path/str): Folder at which the history is to be stored.
        """
        self.storage_path = storage_path

        super().__init__()

    def on_epoch_end(self, epoch: int, logs: object = None):
        """ 
        Args:
            epoch (_type_): _description_
            logs (_type_, optional): _description_. Defaults to None.
        """

        if ('lr' not in logs.keys()):
            logs.setdefault('lr', 0)
            logs['lr'] = K.get_value(self.model.optimizer.lr)

        if self.storage_path.is_file():
            with open(self.storage_path, 'r+') as file:
                eval_hist = json.load(file)
        else:
            eval_hist = dict()

        for key, value in logs.items():
            if not key in eval_hist:
                eval_hist[key] = list()

            eval_hist[key].append(float(value))

        with open(self.storage_path, 'w') as file:
            json.dump(eval_hist, file, indent=4)


class DecayLearningRate(callbacks.Callback):

    def __init__(self, freq, factor):
        """
        """
        self.freq = freq
        self.factor = factor

    def on_epoch_end(self, epoch, logs=None):
        """
        """
        if epoch % self.freq == 0 and not epoch == 0:  # adjust the learning rate
            lr = float(backend.get_value(self.model.optimizer.lr))  # get the current learning rate
            new_lr = lr * self.factor
            backend.set_value(self.model.optimizer.lr,
                              new_lr)  # set the learning rate in the optimizer


class TFKerasPruningCallback(Callback):
    """tf.keras callback to prune unpromising trials.

    This callback is intend to be compatible for TensorFlow v1 and v2,
    but only tested with TensorFlow v2.

    See `the example <https://github.com/optuna/optuna-examples/blob/main/
    tfkeras/tfkeras_integration.py>`__
    if you want to add a pruning callback which observes the validation accuracy.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or ``val_acc``.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str, patience: int) -> None:

        super().__init__()

        _imports.check()

        self._trial = trial
        self._monitor = monitor
        self._patience = patience

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if self._patience is not None:
            if epoch < self._patience:
                return

        logs = logs or {}
        current_score = logs.get(self._monitor)

        if current_score is None:
            message = ("The metric '{}' is not in the evaluation logs for pruning. "
                       "Please make sure you set the correct metric name.".format(self._monitor))
            warnings.warn(message)
            return

        # Report current score and epoch to Optuna's trial.
        self._trial.report(float(current_score), step=epoch)

        # Prune trial if needed
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)

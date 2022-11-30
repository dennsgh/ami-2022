import tensorflow as tf
import os
import pdb
import numpy as np
import keras.backend as K
from copy import deepcopy
from keras.callbacks import ModelCheckpoint, EarlyStopping
from pipelines import AbstractPipeline
from pathlib import Path
from managers import HistoryManager, CheckpointManager
from callbacks import DecayLearningRate, HistoryCheckpoint
from preprocessing.rand_augmenter import RandAugmenterWrapper


class AbstractILSVRCPipeline(AbstractPipeline):
    """Template for ILSVRC model from the keras API.
    """

    def __init__(self,
                 model: object,
                 batch_size: int,
                 compiler_config: dict,
                 epochs: int,
                 baseline: float = None,
                 lr_decay: float = None,
                 model_name: str = "",
                 model_source: Path = None,
                 store_model: bool = True,
                 root_path: str = None,
                 callbacks: list = [],
                 custom_objects: dict = {},
                 patience: int = 10,
                 save_weights_only: bool = True):
        """Initialize the hyperparameters of your pipeline

        Args:
            batch_size (int): 
            compiler_config (dict): Kwargs for compiler
            epochs (int): 
            model (object): Uncompiled model
            lr_decay (object): Percentage of how much learning rate should be in 2 epochs
            model_name (str): Model name to differentiate between runs
            model_source (Path): Path to model which is to be loaded before training.
            callbacks (list, optional): Additional callbacks
            custom_objects (dict, optional): For restoring saved model. Defaults to {}.
            patience (int, optional): For early stopping. Defaults to 10.
            test_fraction_split (float, optional): Fraction of data for test. Defaults to 0..
            validation_fraction_split (float, optional): Fraction fo data for validation. Defaults to 0..
        """
        self.patience = patience
        self.store_model = store_model
        self.lr_decay = lr_decay
        self.model_source = model_source
        self.baseline = baseline
        self.make_directories(root_path, model_name)
        if self.store_model:
            self.hist_manager = HistoryManager(self.directories["model_path"])

        self.history = None
        self.save_weights_only = save_weights_only

        super().__init__(batch_size, compiler_config, epochs, model, model_name, callbacks,
                         custom_objects, patience, 0, 0)

        self.augmentation_ops = [
            'Invert',
            'Rotate',
            'Posterize',
            'Solarize',
            'SolarizeAdd',
            'Color',
            'Contrast',
            'Brightness',
            'TranslateX',
            'TranslateY',
        ]

        self.augmentation_layer = RandAugmenterWrapper(num_layers=2,
                                                       magnitude=7,
                                                       op_list=self.augmentation_ops)
        self.prepend_layer(self.augmentation_layer)
        if self.model_source:
            self.make_checkpoint_manager(self.model_source)
        else:
            self.make_checkpoint_manager(self.directories["model_path"])

    def train(self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset = None) -> dict:
        """Train the model and return the history from check points. Fully resilient to crashes.


        Args:
            train_ds (tf.data.Dataset): Train dataset
            val_ds (tf.data.Dataset, optional): Validation dataset. Defaults to None.

        Returns:
            dict: history.history dictionary.
        """
        if self.batch_size:
            train_ds = train_ds.batch(self.batch_size)
            val_ds = val_ds.batch(self.batch_size)

        # Callbacks and Checkpoint manager
        es_callback = EarlyStopping(patience=self.patience,
                                    restore_best_weights=True,
                                    baseline=self.baseline)

        if self.store_model:
            cp_callback = ModelCheckpoint(filepath=Path(self.directories["model_path"],
                                                        "cp-{epoch:04d}.ckpt"),
                                          save_weights_only=self.save_weights_only,
                                          save_best_only=True,
                                          verbose=0)

            hist_callback = HistoryCheckpoint(self.history_file)
            default_callbacks = [es_callback, hist_callback, cp_callback]
        else:
            default_callbacks = [es_callback]

        if self.store_model and not self.model_source:
            initial_epoch = self.cp_manager.latest_epoch()

        else:
            initial_epoch = 0

        if self.lr_decay:
            default_callbacks.append(DecayLearningRate(2, self.lr_decay))

        self.model.compile(**self.compiler_config)

        # Enforce learning rate adoption after load model
        if not isinstance(self.compiler_config["optimizer"],
                          str) and self.store_model or self.model_source:
            K.set_value(self.model.optimizer.learning_rate, self.lr_save)

        # Return if model done training
        if self.store_model:
            if self.hist_manager.is_finished():
                return self.hist_manager.history

        # Add validation if dataset
        if val_ds is not None:
            validation_args = {
                "validation_data": val_ds,
            }
        else:
            validation_args = {}

        print("Training model.")

        # Train
        history = self.model.fit(train_ds,
                                 epochs=self.epochs,
                                 initial_epoch=initial_epoch,
                                 callbacks=default_callbacks + self.callbacks,
                                 **validation_args)

        # Evaluate training
        if self.store_model:
            self.hist_manager.finished()
            _, self.best_epoch = self.hist_manager.best
            self.cp_manager.clean_directory(self.best_epoch)
            return self.hist_manager.history
        else:
            self.best_epoch = np.argmin(history.history["val_loss"]) + 1

        print("Training complete.")

        return history.history

    def make_checkpoint_manager(self, model_directory: Path) -> None:
        """Create checkpoint manager and loads model from directory if checkpoint available.

        Args:
            model_directory (Path): Directory where checkpoints are saved
        """
        self.cp_manager = CheckpointManager(model_directory,
                                            self.epochs,
                                            custom_objects=self.custom_objects,
                                            save_weights_only=self.save_weights_only)
        if not self.cp_manager.is_empty():
            print(f"Loading model weights from {model_directory}")
            if self.save_weights_only:
                self.model = self.cp_manager.load_weights(self.model)
            else:
                self.model = self.cp_manager.load_model()

            if not isinstance(self.compiler_config["optimizer"], str):
                self.lr_save = deepcopy(self.compiler_config["optimizer"].learning_rate)
            self.model.compile(**self.compiler_config)
        return

    def transfrom(self, train_ds: tf.data.Dataset) -> tf.data.Dataset:
        """No transfromation is need for the EfficentNet.

        Args:
            train_ds (tf.data.Dataset): Training dataset.

        Raises:
            NotImplementedError: Implementation mandatory

        Returns:
            tf.data.Dataset: Dataset as fed to the model.
        """
        raise NotImplementedError
        return train_ds

    def make_directories(self, root_path: Path, model_name: str) -> None:
        """Create the set of directories needed by the pipeline to store the model on run.

        Args:
            root_path (Path): Root path where the base directory is to be setup.
            model_name (str): Name of the model directory

        Raises:
            ValueError: If model name is missing.
        """
        if not model_name:
            raise ValueError("If store_model=True you need to provide a model_name!")

        if root_path is None:
            self.root_path = Path(os.getenv("WORKINGDIR"), "models")

        if isinstance(root_path, str):
            self.root_path = Path(root_path)

        if isinstance(root_path, Path):
            self.root_path = root_path

        self.directories = {
            "root_path": self.root_path,
            "model_path": Path(self.root_path, model_name),
        }

        self.history_file = Path(self.directories["model_path"], 'history.json')

        for directory in self.directories.values():
            directory.mkdir(parents=True, exist_ok=True)

    def save_model(self, path: Path = None):
        """Save model to directory.

        Args:
            path (Path, optional): If directory differs from storage dir. Defaults to None.
        """
        if self.cp_manager is None:
            raise
        if path is None:
            path = self.directories["model_path"]

        print(f"Saving model weights at {self.directories['model_path']}")
        self.cp_manager.save_model(path, self.model, self.best_epoch)


class EfficientNetPipeline(AbstractILSVRCPipeline):
    """Pipeline for EfficientNet models.
    """

    def __init__(self,
                 model: object,
                 batch_size: int,
                 compiler_config: dict,
                 epochs: int,
                 baseline: float = None,
                 lr_decay: float = None,
                 model_name: str = "",
                 model_source: Path = None,
                 store_model: bool = True,
                 root_path: str = None,
                 callbacks: list = [],
                 custom_objects: dict = {},
                 patience: int = 10,
                 save_weights_only: bool = True):
        """Initialize the hyperparameters of your pipeline and add the augmentation layer.

        Args:
            batch_size (int): 
            compiler_config (dict): Kwargs for compiler
            epochs (int): 
            baseline (float): baseline value for early stopping
            model (object): Uncompiled model
            lr_decay (object): Percentage of how much learning rate should be in 2 epochs
            model_name (str): Model name to differentiate between runs
            model_source (Path): Path to model which is to be loaded before training.
            callbacks (list, optional): Additional callbacks
            custom_objects (dict, optional): For restoring saved model. Defaults to {}.
            patience (int, optional): For early stopping. Defaults to 10.
            test_fraction_split (float, optional): Fraction of data for test. Defaults to 0..
            validation_fraction_split (float, optional): Fraction fo data for validation. Defaults to 0..
        """
        super().__init__(model, batch_size, compiler_config, epochs, baseline, lr_decay, model_name,
                         model_source, store_model, root_path, callbacks, custom_objects, patience,
                         save_weights_only)

    def augment(self, train_ds: tf.data.Dataset) -> tf.data.Dataset:
        """No augmentation is need for the EfficentNet. Only use augmentation layer.


        Args:
            train_ds (tf.data.Dataset): Training dataset.

        Returns:
            tf.data.Dataset: Identical dataset
        """
        return train_ds

    def transfrom(self, train_ds: tf.data.Dataset) -> tf.data.Dataset:
        """No transfromation is need for the EfficentNet


        Args:
            train_ds (tf.data.Dataset): Training dataset.

        Returns:
            tf.data.Dataset: Identical dataset
        """
        return train_ds

import os
import tensorflow as tf
import os
from pathlib import Path
from utils.file_system import load_json, update_json
from pathlib import Path
from managers import SettingsManager
from models.ilsvrc import EfficientNetV2S
from pipelines.ilsvrc import EfficientNetPipeline
from flask_session import Session
from self_supervision.methods import (PiecewiseConstantDecayWithLinearWarmup, custom_crossentropy)
from preprocessing.data_loader import DataLoader


class SelfSupModelContainer:

    def __init__(self):
        self.saved_path = Path(os.getenv("MODEL"), "checkpoints-finetuning",
                               "checkpoints-finetuning")
        self.model = tf.keras.models.load_model(
            self.saved_path,
            custom_objects={
                "PiecewiseConstantDecayWithLinearWarmup": PiecewiseConstantDecayWithLinearWarmup,
                "scce_with_ls": custom_crossentropy
            },
            compile="False")


class TransferModelContainer:
    """Container for TL model.
    """

    def __init__(self) -> None:
        """Initialize the the model paths.
        """
        self.model_config = Path(os.getenv("CONFIG"), "model_config.json")
        self.data_config = Path(os.getenv("CONFIG"), "data_config.json")
        model_source = load_json(self.model_config)["current_model"]
        self.model_source = Path(os.getenv("MODEL"), model_source)
        self.manager = SettingsManager(self.model_source)
        self.init_model()

    def init_model(self) -> None:
        """Initialize the model
        """
        model = EfficientNetV2S(**self.manager.model_config)

        pipeline_config = self.manager.pipeline_config
        pipeline_config["model_name"] = "inference"

        self.pipeline = EfficientNetPipeline(model, **pipeline_config)

    def reset_model(self) -> None:
        """Reset the model to initial version
        """
        model_source = load_json(self.model_config)["current_model"]
        self.model_source = Path(os.getenv("MODEL"), model_source)
        self.manager = SettingsManager(self.model_source)
        self.init_model()

    def predict(self, image: tf.Tensor) -> tuple:
        """Predict

        Args:
            image (tf.Tensor): Image tensor

        Returns:
            tuple: _description_
        """
        return self.pipeline.predict(image)

    def retrain(self) -> None:
        """Retrain the model and store to new directory.
        """
        model_name = f"retrained_model{self.retrain_counter}"
        self.model_dir = Path(os.getenv("MODEL"), model_name)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        anotation_location = load_json(self.data_config)["anotation_location"]
        data_loader = DataLoader(json_directory=Path(os.getenv("DATA"), anotation_location),
                                 val_size=0.2,
                                 test_size=0.2,
                                 perf_crop=True,
                                 image_size=[224, 224],
                                 resizing_type='pad',
                                 batch_size=None,
                                 class_weights=[0.25, 0.25, 0.25, 0.25],
                                 data_type=tf.data.Dataset)
        train_ds, test_ds, val_ds = data_loader.run()

        self.retrain_feature_extraction(model_name, train_ds, val_ds)
        self.retrain_fine_tuning(model_name, train_ds, val_ds)
        test_ds = test_ds.batch(10)
        self.pipeline.score(test_ds)

    def retrain_feature_extraction(self, model_name: str, train_ds: tf.data.Dataset,
                                   val_ds: tf.data.Dataset) -> None:
        """Retrain the model on feature extraction step.

        Args:
            model_name (str): Name of current retrain iteration
            train_ds (tf.data.Dataset): Train dataset
            val_ds (tf.data.Dataset): Validation dataset
        """
        model_config, pipeline_config, learning_rate = self.manager.feature_extraction_config
        if learning_rate:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            pipeline_config["compiler_config"]["optimizer"] = optimizer
        pipeline_config["root_path"] = Path(os.getenv("MODEL"))
        pipeline_config["model_name"] = model_name

        model = EfficientNetV2S(**model_config)
        self.pipeline = EfficientNetPipeline(model, **pipeline_config)
        history = self.pipeline.fit(train_ds, val_ds)
        self.pipeline.save_model()

    def retrain_fine_tuning(self, model_name: str, train_ds: tf.data.Dataset,
                            val_ds: tf.data.Dataset):
        """Retrain the model on fine tuning setp.

        Args:
            model_name (str): Name of current retrain iteration
            train_ds (tf.data.Dataset): Train dataset
            val_ds (tf.data.Dataset): Validation dataset
        """
        model_config, pipeline_config, learning_rate, trainable_blocks = self.manager.fine_tuning_config
        if learning_rate:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            pipeline_config["compiler_config"]["optimizer"] = optimizer

        pipeline_config["root_path"] = Path(os.getenv("MODEL"))
        pipeline_config["model_name"] = model_name
        pipeline_config["model_source"] = Path(os.getenv("MODEL"), model_name)

        model = EfficientNetV2S(**model_config)
        model.trainable_blocks = trainable_blocks
        self.pipeline = EfficientNetPipeline(model, **pipeline_config)
        history = self.pipeline.fit(train_ds, val_ds)
        self.pipeline.save_model()
        self.pipeline.cp_manager.clean_directory(pipeline_config["epochs"] + 1)
        self.model_source = pipeline_config["model_source"]

    @property
    def retrain_counter(self) -> int:
        """Increase on each retrain iteration.

        Returns:
            int: current counter value
        """
        counter = load_json(self.model_config)["counter"]
        counter += 1
        update_json(self.model_config, counter)
        return counter


if __name__ == "__main__":
    import cv2
    import numpy as np
    container = TransferModelContainer()
    image = cv2.imread("/home/amadou/CodeWorkspace/Group04/data/cropped/503993.png")
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(np.array(image), 0).tolist()

    pred = container.predict(image)
    print(pred)

    container.retrain()
    pred = container.predict(image)
    print(pred)

    container.reset_model()
    pred = container.predict(image)
    print(pred)

    container.retrain()
    pred = container.predict(image)
    print(pred)

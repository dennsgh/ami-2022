from multiprocessing import dummy
import pytest
import tensorflow as tf
import pdb
import multiprocessing as mp
from pathlib import Path
from models.ilsvrc import EfficientNetV2S
from pipelines.ilsvrc import EfficientNetPipeline


@pytest.fixture(scope="session")
def model_config() -> list:
    """ Returns EfficientNet model configs to test.

    Returns:
        list: list of configurtions
    """
    config = {
        "depth": 1,
        "layer_size": (128, 64, 32),
        "dropout": 0,
        "input_shape": (224, 224),
        "pooling_type": "max"
    }
    return config


@pytest.fixture(scope="session")
def pipeline_config() -> list:
    """ Returns EfficientNet model configs to test.

    Returns:
        list: list of configurations
    """
    config = {
        "batch_size": 32,
        "epochs": 2,
        "model_name": "trial",
        "store_model": True,
        "patience": 10,
        "compiler_config": {
            "optimizer": "adam",
            "loss": "sparse_categorical_crossentropy",
            "metrics": ['accuracy']
        }
    }
    return config


@pytest.fixture(scope="session")
def dummy_dataset() -> tf.data.Dataset:
    """ Returns data set with single image per class

    Returns:
        tf.data.Dataset: dummy dataset
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=Path(Path(__file__).parent.parent, "test_data/model_data/images/"),
        seed=0,
        image_size=(224, 224),
        batch_size=16
    ) # yapf: disable

    return train_ds


def test_pipeline(model_config: list, pipeline_config: list,
                  dummy_dataset: tf.data.Dataset) -> None:
    """ Validation testing for EfficientNetV2S wrapper. 

    Args:
        model_configs (list): list of configurations to test.
        compiler_configs (dict): compiler configuration for the model
        dummy_dataset (tf.data.Dataset): tiny dataset
    """
    # Unbatch to test batch_size kword
    dummy_dataset = dummy_dataset.unbatch()
    model = EfficientNetV2S(**model_config)
    pipeline = EfficientNetPipeline(model, **pipeline_config)
    pipeline.fit(dummy_dataset, dummy_dataset)
    dummy_dataset = dummy_dataset.batch(10)
    # rebatch at inference
    score = pipeline.score(dummy_dataset)
    pred = pipeline.predict(dummy_dataset)
    pred_class = pipeline.predict_class(dummy_dataset)


if __name__ == "__main__":
    test_pipeline()
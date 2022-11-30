import pytest
import tensorflow as tf
import pdb
import multiprocessing as mp
from pathlib import Path
from models.ilsvrc import EfficientNetV2S, EfficientNetV2M, EfficientNetV2L


@pytest.fixture(scope="session")
def model_configs() -> list:
    """ Returns EfficientNet model configs to test.

    Returns:
        list: list of configurtions
    """
    config = [{
        "depth": 1,
        "layer_size": 50,
        "dropout": 0.5,
        "input_shape": (244, 244),
        "pooling_type": "avg"
    }, {
        "depth": 1,
        "layer_size": (128, 64, 32),
        "dropout": 0,
        "input_shape": (224, 224),
        "pooling_type": "max"
    }]
    return config


@pytest.fixture(scope="session")
def compiler_configs() -> dict:
    """ Returns the compiler configuration.

    Returns:
        dict: compiler config
    """
    config = {
        "optimizer": tf.keras.optimizers.Adam(0.001),
        "loss": tf.keras.losses.SparseCategoricalCrossentropy(),
        "metrics": ['accuracy'],
        "run_eagerly": True
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


def test_validation_efficientnetv2s(model_configs: list, compiler_configs: dict,
                                    dummy_dataset: tf.data.Dataset) -> None:
    """ Validation testing for EfficientNetV2S wrapper. 

    Args:
        model_configs (list): list of configurations to test.
        compiler_configs (dict): compiler configuration for the model
        dummy_dataset (tf.data.Dataset): tiny dataset
    """
    for model_config in model_configs:
        model = EfficientNetV2S(**model_config)
        model.compile(**compiler_configs)
    # Train on the last only due to shape mismatch
    model.trainable_blocks = 29
    model.summary()
    model.fit(dummy_dataset, epochs=1)


def test_validation_efficientnetv2m(model_configs: list, compiler_configs: dict,
                                    dummy_dataset: tf.data.Dataset) -> None:
    """ Validation testing for EfficientNetV2M wrapper. 

    Args:
        model_configs (list): list of configurations to test.
        compiler_configs (dict): compiler configuration for the model
        dummy_dataset (tf.data.Dataset): tiny dataset
    """
    for model_config in model_configs:
        model = EfficientNetV2M(**model_config)
        model.compile(**compiler_configs)
    # Train on the last only due to shape mismatch
    model.trainable_blocks = 57
    model.summary()


def test_validation_efficientnetv2l(model_configs: list, compiler_configs: dict,
                                    dummy_dataset: tf.data.Dataset) -> None:
    """ Validation testing for EfficientNetV2L wrapper. 

    Args:
        model_configs (list): list of configurations to test.
        compiler_configs (dict): compiler configuration for the model
        dummy_dataset (tf.data.Dataset): tiny dataset
    """
    for model_config in model_configs:
        model = EfficientNetV2M(**model_config)
        model.compile(**compiler_configs)
    # Train on the last only due to shape mismatch
    model.trainable_blocks = 79
    model.summary()
    model.fit(dummy_dataset, epochs=1)


if __name__ == "__main__":
    test_validation_efficientnetv2s()
    test_validation_efficientnetv2m()
    test_validation_efficientnetv2l()
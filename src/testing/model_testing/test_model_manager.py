import pytest
import tensorflow as tf
from model_manager import ModelManager


def test_prediction():
    """full system test, load, set mode and predict
    """
    # ---------------------------------- Arrange --------------------------------- #
    model_manager = ModelManager()

    arr_3d = tf.random.uniform(shape=[224, 224, 3])
    arr_3d = tf.cast(arr_3d, dtype=tf.uint8)
    arr_3d = arr_3d.numpy()
    to_test = model_manager.get_valid_modes()
    # ------------------------------------ Act ----------------------------------- #

    for mode in to_test:
        print("Testing mode {}...".format(mode))
        model_manager.set_mode(mode)
        label = model_manager.predict(arr_3d)
        assert len(label)


if __name__ == "__main__":
    test_prediction()
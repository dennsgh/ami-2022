import tensorflow as tf
from pathlib import Path
import os
from preprocessing.data_loader import DataLoader


def test_data_loader() -> None:
    """This test tests the data loader functionality of the DataLoader class
    """
    # ---------------------------------- Arrange --------------------------------- #

    data_loader = DataLoader(json_directory=Path(os.getenv('WORKINGDIR'), 'src', 'testing',
                                                 'pyunittests', 'dummy_data',
                                                 'dummy_annotation.json'),
                             val_size=0.25,
                             test_size=0.25,
                             image_size=[244, 244],
                             resizing_type='crop',
                             batch_size=1,
                             class_weights=[0.25, 0.25, 0.25, 0.25],
                             data_type=tf.data.Dataset)

    expected_train_len = 8
    expected_test_len = 4
    expected_eval_len = 4

    # ------------------------------------ Act ----------------------------------- #

    train_ds, test_ds, eval_ds = data_loader.run()
    actual_train_len = len(list(train_ds))
    actual_test_len = len(list(test_ds))
    actual_eval_len = len(list(eval_ds))

    # ---------------------------------- Assert ---------------------------------- #

    assert expected_train_len == actual_train_len
    assert expected_test_len == actual_test_len
    assert expected_eval_len == actual_eval_len
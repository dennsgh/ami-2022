from preprocessing.data_augmenter import DataAugmenter
import tensorflow as tf
from pathlib import Path
from preprocessing.data_loader import DataLoader
import os

def test_color_augmentation():
    """This test tests the color augmentation functionality of the DataAugmenter class
    """
    # ---------------------------------- Arrange --------------------------------- #
    seed = [1, 1]
    test_image = tf.random.uniform(shape=[224, 224, 3])
    expected_image = tf.image.stateless_random_brightness(test_image, 0.5, seed)
    expected_image = tf.image.stateless_random_contrast(expected_image, 0.3, 0.7, seed)
    expected_image = tf.image.stateless_random_saturation(expected_image, 0.3, 0.7, seed)
    expected_image = tf.image.stateless_random_hue(expected_image, 0.5, seed)

    method = {
        'brightness': {
            'max_delta': 0.5
        },
        'contrast': {
            'upper': 0.7,
            'lower': 0.3
        },
        'saturation': {
            'upper': 0.7,
            'lower': 0.3
        },
        'hue': {
            'max_delta': 0.5
        }
    }

    # ------------------------------------ Act ----------------------------------- #
    Aug = DataAugmenter()
    new_image = Aug._transform(test_image, seed, **method)

    # ---------------------------------- Assert ---------------------------------- #
    assert tf.math.reduce_all(tf.equal(expected_image, new_image))


def test_orientation_augmentation():
    """This test tests the orientation augmentation functionality of the DataAugmenter class
    """
    # ---------------------------------- Arrange --------------------------------- #
    seed = [1, 1]
    test_image = tf.random.uniform(shape=[224, 224, 3])
    expected_image = tf.image.stateless_random_flip_left_right(test_image, seed)
    expected_image = tf.image.stateless_random_flip_up_down(expected_image, seed)
    method = {'flip_lr': {}, 'flip_ud': {}}

    # ------------------------------------ Act ----------------------------------- #
    Aug = DataAugmenter()
    new_image = Aug._transform(test_image, seed, **method)

    # ---------------------------------- Assert ---------------------------------- #
    assert tf.math.reduce_all(tf.equal(expected_image, new_image))


def test_transform():
    """Test for augmentation of dataset with .transform function
    """
    # ---------------------------------- Arrange --------------------------------- #
    data_loader = DataLoader(json_directory=Path(os.getenv('WORKINGDIR'), 'src', 'testing',
                                                 'pyunittests', 'dummy_data',
                                                 'dummy_annotation.json'),
                             val_size=0.3,
                             test_size=0.3,
                             perf_crop=True,
                             image_size=[244, 244],
                             resizing_type='crop',
                             batch_size=30,
                             class_weights=[0.25, 0.25, 0.25, 0.25],
                             data_type=tf.data.Dataset)

    seed = [1, 1]
    method = {
        'brightness': {
            'max_delta': 0.5
        },
        'contrast': {
            'upper': 0.7,
            'lower': 0.3
        },
        'flip_lr': {},
        'flip_ud': {}
    }
    # ------------------------------------ Act ----------------------------------- #
    train_ds, test_ds, eval_ds = data_loader.run()

    data_aug = DataAugmenter()
    # This will test both internal_transform and randaug_transform
    transformed = data_aug.transform(train_ds, seed, **method)

    # ---------------------------------- Assert ---------------------------------- #
    assert transformed


if __name__ == '__main__':
    test_color_augmentation()
    test_orientation_augmentation()
    test_transform()

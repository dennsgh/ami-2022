import tensorflow as tf
from pathlib import Path
import os
from preprocessing.rand_augmenter import RandAugmenter
from preprocessing.data_loader import DataLoader


def test_transform():
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

    op_list = [
        #'AutoContrast', # problematic for 4D Tensors,but runs fine for single images [224 224 3]
        #'Equalize', # problematic for 4D Tensors,but runs fine for single images [224 224 3]
        'Invert',
        'Rotate',
        'Posterize',
        'Solarize',
        'SolarizeAdd',
        'Color',
        'Contrast',
        'Brightness',
        #'Sharpness',  # doesn't work with 4D Tensors yet! -> do not use for tf.Data.Dataset [32 224 224 3]
        #'ShearX', # doesn't work with 4D Tensors yet! -> do not use for tf.Data.Dataset[32 224 224 3]
        #'ShearY', # doesn't work with 4D Tensors yet! -> do not use for tf.Data.Dataset[32 224 224 3]
        'TranslateX',
        'TranslateY',
        #'Cutout' # doesn't work with 4D Tensors yet! -> do not use for tf.Data.Dataset[32 224 224 3]
    ]
    # ------------------------------------ Act ----------------------------------- #
    train_ds, test_ds, eval_ds = data_loader.run()

    rand_aug = RandAugmenter(num_layer=2, magnitude=7, op_list=op_list)
    # This will test both internal_transform and randaug_transform
    transformed = rand_aug.transform(train_ds, 3000)
    # ---------------------------------- Assert ---------------------------------- #
    # Transform currently only applies transformations, the number of images do not change
    assert transformed


def test_3d_operations():
    """Tests operations on 3D image tensors
    """
    # ---------------------------------- Arrange --------------------------------- #

    failed_set = []

    # ------------------------------------ Act ----------------------------------- #
    ra = RandAugmenter(2, 9)
    op_list = ra.op_list
    replace_value = [128] * 3
    arr_3d = tf.random.uniform(shape=[224, 224, 3])
    arr_3d = tf.cast(arr_3d, dtype=tf.uint8)
    for (i, op_name) in enumerate(op_list):
        # dummy to pass as arg to call the functions, making sure the parsing works
        prob = tf.random.uniform([], minval=0.2, maxval=0.8, dtype=tf.float32)
        func, _, args = ra._parse_policy_info(op_name, prob, ra.magnitude, replace_value)
        try:
            arr_3d = func(arr_3d, *args)
        except:
            failed_set.append(op_name)
    # ---------------------------------- Assert ---------------------------------- #
    assert not len(failed_set)


def test_4d_operations():
    """Tests operations on 4D image tensors (batch images)
    """
    # ---------------------------------- Arrange --------------------------------- #

    failed_set = []

    # ------------------------------------ Act ----------------------------------- #

    ra = RandAugmenter(2, 9)
    op_list = [
        'AutoContrast',  # problematic for 4D Tensors,but runs fine for single images [224 224 3]
        'Equalize',  # problematic for 4D Tensors,but runs fine for single images [224 224 3]
        'Invert',
        'Rotate',
        'Posterize',
        'Solarize',
        'SolarizeAdd',
        'Color',
        'Contrast',
        'Brightness',
        #'Sharpness',  # doesn't work with 4D Tensors yet! -> do not use for tf.Data.Dataset [32 224 224 3]
        #'ShearX', # doesn't work with 4D Tensors yet! -> do not use for tf.Data.Dataset[32 224 224 3]
        #'ShearY', # doesn't work with 4D Tensors yet! -> do not use for tf.Data.Dataset[32 224 224 3]
        'TranslateX',
        'TranslateY',
        #'Cutout' # doesn't work with 4D Tensors yet! -> do not use for tf.Data.Dataset[32 224 224 3]
    ]
    replace_value = [128] * 3
    arr_4d = tf.random.uniform(shape=[32, 224, 224, 3])
    arr_4d = tf.cast(arr_4d, dtype=tf.uint8)
    for (i, op_name) in enumerate(op_list):
        prob = tf.random.uniform([], minval=0.2, maxval=0.8, dtype=tf.float32)
        func, _, args = ra._parse_policy_info(op_name, prob, ra.magnitude, replace_value)
        try:
            arr_4d = func(arr_4d, *args)
        except:
            failed_set.append(op_name)
    # ---------------------------------- Assert ---------------------------------- #
    assert not len(failed_set)


if __name__ == '__main__':
    test_transform()
    test_3d_operations()
    test_4d_operations()

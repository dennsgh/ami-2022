from pathlib import Path
import os
import json
import pandas as pd
import numpy as np
from math import floor
from PIL import Image
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


class DataLoader:
    """This class creates and executes a Data Loader Pipeline
    """

    def __init__(self,
                 json_directory: str,
                 val_size: float,
                 test_size: float,
                 image_size: int,
                 resizing_type: str,
                 batch_size: int,
                 class_weights: float,
                 data_type,
                 perf_crop: bool = True,
                 file_type: str = "png"):
        """This function initializes the data loader.

        Args:
            json_directory (str): The json file directory starting from Group04/...
            val_size (float): validation set size
            test_size (float): testing se size
            image_size (int): array of two dim with width and height
            resizing_type (str): resizing type
            batch_size (int): batch size
            class_weights (float): class weight - array with 4 floats
            data_type (_type_): output type (tensorflow, numpy array)
            perf_crop (bool, optional): Should the images get cropped? Defaults to True.
            file_type (str, optional): data type of images (e.g. png,...). Defaults to "png".
        """
        current_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent
        self.json_directory = Path(current_dir, json_directory)
        self.file_type = file_type
        self.val_size = val_size
        self.test_size = test_size
        self.perf_crop = perf_crop
        self.image_size = image_size
        self.resizing_type = resizing_type
        self.batch_size = batch_size
        self.class_weights = class_weights
        self.data_type = data_type

    def run(self):
        """This function runs all relevant steps of the data loading pipeline with in the initial settings

        Returns:
            train_ds: This data frame represents the training data set in a tensorflow data set format
            test_ds: This data frame represents the testing data set in a tensorflow data set format
            eval_ds: This data frame represents the evaluation data set in a tensorflow data set format
        """
        annotations = self._open_json_as_df()
        dataset = self._create_data_set(annotations)
        train_ds, test_ds, eval_ds = self._split_data_set(dataset)
        train_ds, test_ds, eval_ds = self._apply_batch_size(train_ds, test_ds, eval_ds)
        train_ds, test_ds, eval_ds = self._transform_datatype(train_ds, test_ds, eval_ds)

        return train_ds, test_ds, eval_ds

    def _open_json_as_df(self):
        """This function returns a json format in form of a data frame with having the path of the file

        Returns:
            json_data_frame: A data frame with having all information of the json file
        """
        with self.json_directory.open() as f:
            json_data = json.load(f)

        return pd.json_normalize(json_data, record_path=['annotations'])

    def _create_data_set(self, annotations):
        """This function loads and generates a tensorflow data set and crops the pictures if wanted.

        Args:
            annotations (pandas data frame): A data frame with having all information of the annotation file

        Returns:
            tensorflow data set: The loaded pictures with their labels are returned.
        """

        # ------------------------------ Help functions ------------------------------ #
        def _clean_annotations(annotations):
            """Filters out empty annotations and sets the right type for columns.

            Args:
                annotations (pandas data frame): A data frame with having all information of the annotation file

            Returns:
                pandas data frame: A cleaned annotations data frame is returned
            """
            annotations = annotations[annotations.category_id != '']
            annotations["id"] = annotations["id"].astype("str")

            return annotations

        def _open_as_tensor(path):
            """Loads a Picture through its path and converts in to a Tensor object

            Args:
                path (str): The path to the image file
                name (str): The actual file name of the image file

            Returns:
                Tensor: A Tensor Image is returned
            """
            image = Image.open(path)
            img_tf = img_to_array(image).astype('uint8')
            img_tf = tf.convert_to_tensor(img_tf)

            return img_tf

        def _resize_img(img, size, type):
            """This function resizes a pictures with different options of the type of resizing

            Args:
                img (Tensor): Image
                size (int array): The height and width in an array
                type (str): type of resizing (resize, pad, crop)

            Returns:
                Tensor: Resized Tensor Image is returned
            """
            if type == 'resize':
                im_tf = tf.image.resize(img, size=size)
            elif type == 'pad':
                im_tf = tf.image.resize_with_pad(
                    img,
                    target_height=size[0],
                    target_width=size[1],
                )
            elif type == 'crop':
                h, w = img.shape[-3], img.shape[-2]
                if h < w:
                    offset_height = 0
                    offset_width = (w - h) // 2
                    target_height = h
                    target_width = h
                else:
                    offset_height = (h - w) // 2
                    offset_width = 0
                    target_height = w
                    target_width = w
                cropped = tf.image.crop_to_bounding_box(img,
                                                        offset_height=offset_height,
                                                        offset_width=offset_width,
                                                        target_height=target_height,
                                                        target_width=target_width)
                im_tf = tf.image.resize(cropped, (size[0], size[1]))

            return im_tf

        images, labels = [], []
        annotations = _clean_annotations(annotations)
        # walk through all folders
        for index in annotations.index:
            total_path = Path(self.json_directory.parent, annotations.filepath[index])
            # Open pictures as tensor
            image = _open_as_tensor(total_path)
            if self.perf_crop:
                resized_img = _resize_img(image, self.image_size, self.resizing_type)
                images.append(resized_img)
            else:
                images.append(image)
            labels.append(int(annotations.category_id[index]))
        if self.perf_crop:
            dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        else:
            dataset = tf.data.Dataset.from_generator(lambda: images,
                                                     output_signature=tf.TensorSpec(shape=(None),
                                                                                    dtype=tf.uint8))
            dataset = tf.data.Dataset.zip(dataset, labels)

        return dataset

    def _split_data_set(self, dataset):
        """This function splits a Tensor data set into training, test and evaluation data sets.

        Args:
            dataset (Tensorflow data set): One Tensorflow data set is given

        Returns:
            train_ds (Tensorflow data set): Training data set
            test_ds (Tensorflow data set): Testing data set
            eval_ds (Tensorflow data set): Evaluation data set
        """

        def _apply_weighting_and_split(self, dataset):
            """This function should apply the weights and split the data set into training, test and evaluation

            Args:
                dataset (Tensorflow data set): input data set

            Raises:
                IndexError: This function only works when the data set has at least one picture

            Returns:
                Tensorflow data set: weighted data set (right now shuffled)
            """
            if len(list(dataset)) != 0:
                ds_0_image, ds_1_image, ds_2_image, ds_3_image = [], [], [], []
                ds_0_label, ds_1_label, ds_2_label, ds_3_label = [], [], [], []
                ds_0_weights, ds_1_weights, ds_2_weights, ds_3_weights = [], [], [], []
                for image in dataset:
                    if image[1].numpy() == 0:
                        ds_0_image.append(image[0])
                        ds_0_label.append(image[1])
                        ds_0_weights.append(self.class_weights[0])
                    elif image[1].numpy() == 1:
                        ds_1_image.append(image[0])
                        ds_1_label.append(image[1])
                        ds_1_weights.append(self.class_weights[1])
                    elif image[1].numpy() == 2:
                        ds_2_image.append(image[0])
                        ds_2_label.append(image[1])
                        ds_2_weights.append(self.class_weights[2])
                    elif image[1].numpy() == 3:
                        ds_3_image.append(image[0])
                        ds_3_label.append(image[1])
                        ds_3_weights.append(self.class_weights[3])
                    else:
                        ValueError('An unexpected annotation was found!')

                dataset_size = len(list(dataset))
                share_zero = len(ds_0_image) / dataset_size
                share_one = len(ds_1_image) / dataset_size
                share_two = len(ds_2_image) / dataset_size
                share_three = len(ds_3_image) / dataset_size
                random_indices_ds_0 = random.sample(range(0, len(ds_0_image)),
                                                    floor(share_zero * dataset_size))
                random_indices_ds_1 = random.sample(range(0, len(ds_1_image)),
                                                    floor(share_one * dataset_size))
                random_indices_ds_2 = random.sample(range(0, len(ds_2_image)),
                                                    floor(share_two * dataset_size))
                random_indices_ds_3 = random.sample(range(0, len(ds_3_image)),
                                                    floor(share_three * dataset_size))

                def _get_dataset_by_indices(self, images, labels, weights, indices):
                    train_indices = [
                        0, floor((1 - self.test_size - self.val_size) * len(indices)) - 1
                    ]
                    train_images = [
                        images[i] for i in indices[train_indices[0]:train_indices[1] + 1]
                    ]
                    train_labels = [
                        labels[i] for i in indices[train_indices[0]:train_indices[1] + 1]
                    ]
                    train_weights = [
                        weights[i] for i in indices[train_indices[0]:train_indices[1] + 1]
                    ]
                    test_indices = [train_indices[1] + 1, floor((1 - self.val_size) * len(indices))]
                    if test_indices[1] == len(indices) - 1:
                        test_indices[1] = test_indices[1] - 1
                    test_images = [images[i] for i in indices[test_indices[0]:test_indices[1] + 1]]
                    test_labels = [labels[i] for i in indices[test_indices[0]:test_indices[1] + 1]]
                    test_weights = [
                        weights[i] for i in indices[test_indices[0]:test_indices[1] + 1]
                    ]
                    eval_indices = [test_indices[1] + 1, len(indices) - 1]
                    eval_images = [images[i] for i in indices[eval_indices[0]:eval_indices[1] + 1]]
                    eval_labels = [labels[i] for i in indices[eval_indices[0]:eval_indices[1] + 1]]
                    eval_weights = [
                        weights[i] for i in indices[eval_indices[0]:eval_indices[1] + 1]
                    ]

                    return tf.data.Dataset.from_tensor_slices(
                        (train_images, train_labels,
                         train_weights)), tf.data.Dataset.from_tensor_slices(
                             (test_images, test_labels,
                              test_weights)), tf.data.Dataset.from_tensor_slices(
                                  (eval_images, eval_labels, eval_weights))

                train_ds_0, test_ds_0, eval_ds_0 = _get_dataset_by_indices(
                    self, ds_0_image, ds_0_label, ds_0_weights, random_indices_ds_0)

                train_ds_1, test_ds_1, eval_ds_1 = _get_dataset_by_indices(
                    self, ds_1_image, ds_1_label, ds_1_weights, random_indices_ds_1)

                train_ds_2, test_ds_2, eval_ds_2 = _get_dataset_by_indices(
                    self, ds_2_image, ds_2_label, ds_2_weights, random_indices_ds_2)

                train_ds_3, test_ds_3, eval_ds_3 = _get_dataset_by_indices(
                    self, ds_3_image, ds_3_label, ds_3_weights, random_indices_ds_3)

                def _concatenate_four_ds(data1, data2, data3, data4):
                    data12 = data1.concatenate(data2)
                    data34 = data3.concatenate(data4)
                    data = data12.concatenate(data34)
                    return data

                train_ds = _concatenate_four_ds(train_ds_0, train_ds_1, train_ds_2, train_ds_3)
                test_ds = _concatenate_four_ds(test_ds_0, test_ds_1, test_ds_2, test_ds_3)
                eval_ds = _concatenate_four_ds(eval_ds_0, eval_ds_1, eval_ds_2, eval_ds_3)

                train_ds = train_ds.shuffle(len(list(train_ds)))
                test_ds = test_ds.shuffle(len(list(test_ds)))
                eval_ds = eval_ds.shuffle(len(list(eval_ds)))

            else:
                raise IndexError('The length of the the dataset have to be larger than 0.')

            return train_ds, test_ds, eval_ds

        train_ds, test_ds, eval_ds = _apply_weighting_and_split(self, dataset)
        # train_ds = dataset.take(floor((1-test_size-val_size) * dataset_size))
        # test_ds = dataset.take(floor(test_size * dataset_size))
        # eval_ds = dataset.take(floor(val_size * dataset_size))

        return train_ds, test_ds, eval_ds

    def _apply_batch_size(self, train_ds, test_ds, eval_ds):
        """This function divides all data set into batches.

        Args:
            train_ds (Tensorflow data set): Training data set
            test_ds (Tensorflow data set): Testing data set
            eval_ds (Tensorflow data set): Evaluation data set

        Returns:
            Tensorflow batched data set: Batched training data set
            Tensorflow batched data set: Batched testing data set
            Tensorflow batched data set: Batched evaluation data set
        """
        if self.batch_size is not None:
            train_ds = train_ds.batch(self.batch_size)
            test_ds = test_ds.batch(self.batch_size)
            eval_ds = eval_ds.batch(self.batch_size)

        return train_ds, test_ds, eval_ds

    def _transform_datatype(self, train_ds, test_ds, eval_ds):
        """Transforms the all data frames into the wanted format (Tensorflow data set or numpy array)

        Args:
            train_ds (Tensorflow data set): Training data set
            test_ds (Tensorflow data set): Testing data set
            eval_ds (Tensorflow data set): Evaluation data set

        Raises:
            Exception: Only Tensorflow data set or numpy array is allowed

        Returns:
            Tensorflow or numpy array data set: Formatted training data set
            Tensorflow or numpy array data set: Formatted testing data set
            Tensorflow or numpy array data set: Formatted evaluation data set
        """
        if self.data_type == tf.data.Dataset:
            return train_ds, test_ds, eval_ds
        elif self.data_type == np.array:
            return train_ds.as_numpy(), test_ds.as_numpy(), eval_ds.as_numpy()
        else:
            raise Exception('Not excepted data type!')

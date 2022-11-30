import tensorflow as tf
from .operations import Operations
import inspect


class RandAugmenter:
    """Implements RandAugmenter as a standalone class to randomly augment a dataset using the RandAugment algorithm

    Taken mostly from https://github.com/tensorflow/tpu/blob/1ebfc4b5fa6b5fad0bd7d422832ae8826e8059f2/models/official/efficientnet/autoaugment.py

    Todo:
        * make all operations work for 4D tensors as well
    """

    def __init__(self, num_layer: int, magnitude: int, op_list: "list[str]" = None) -> None:
        """Init function

        Args:
            num_layer (int): number of sub-policies
            magnitude (int): strength of the transformations applied
            op_list (list[str], optional): list of str containing operations . Defaults to everything in self.__transforms
        """
        self.__verbose = False
        self.__IMG_WIDTH = 224
        self.__IMG_HEIGHT = 224
        self.__ops = Operations()
        self.__transforms = {
            'AutoContrast': self.__ops.auto_contrast,
            'Equalize': self.__ops.equalize,
            'Invert': self.__ops.invert,
            'Rotate': self.__ops.rotate,
            'Posterize': self.__ops.posterize,
            'Solarize': self.__ops.solarize,
            'SolarizeAdd': self.__ops.solarize_add,
            'Color': self.__ops.color,
            'Contrast': self.__ops.contrast,
            'Brightness': self.__ops.brightness,
            'Sharpness': self.__ops.sharpness,
            'ShearX': self.__ops.shear_x,
            'ShearY': self.__ops.shear_y,
            'TranslateX': self.__ops.translate_x,
            'TranslateY': self.__ops.translate_y,
            'Cutout': self.__ops.cutout,
        }
        #: int: seed for random operations, deprecated.
        self.seed = 0
        #: int: number of sub-policies
        self.num_layer = num_layer
        #: int: transformation magnitude
        self.magnitude = magnitude
        #: list of str: list of operations for RandAugmenter to use
        self.op_list = op_list
        #: float: number used for operation magnitude calculation
        self.max_level = 10.
        if op_list is None:
            self.op_list = self.get_default_operations()

    def get_default_operations(self) -> "list[str]":
        """Returns a list of all available operations

        Returns:
            list[str]: list containing strings of default operations
        """
        return list(self.__transforms.keys())

    def set_verbose(self, verbose: bool) -> None:
        """Set to print the augmentations done when randaugment_transform is called

        Args:
            verbose (bool): To print or not to print
        """
        self.__verbose = verbose

    def summary(self) -> None:
        """Prints the summary of the current RandAugmentor's instance
        """
        print(self.op_list)
        print(
            f"{self.num_layer} layers with magnitude {self.magnitude}, with a total of {len(self.op_list)} operations."
        )

    def _randomly_negate_tensor(self, tensor: tf.Tensor) -> tf.Tensor:
        """Negates given tensor with 50% probability

        Args:
            tensor (tf.Tensor): tensor to negate

        Returns:
            tf.Tensor: negated/identity tensor
        """
        should_flip = tf.cast(tf.floor(tf.random.uniform([]) + 0.5), tf.bool)
        final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
        return final_tensor

    def _rotate_level_to_arg(self, level: float) -> tuple:
        """Converts level to ratio by which we rotate the image content.

        Args:
            level (float): magnitude, defined during initialization

        Returns:
            tuple: tuple containing the magnitude for the transformation function argument
        """
        level = (level / self.max_level) * 30.
        level = self._randomly_negate_tensor(level)
        return (level,)

    def _shrink_level_to_arg(self, level: float) -> tuple:
        """Converts level to ratio by which we shrink the image content.

        Args:
            level (float): magnitude, defined during initialization

        Returns:
            tuple: tuple containing the magnitude for the transformation function argument
        """
        if level == 0:
            # if level is zero, do not shrink the image
            return (1.0,)
        # Maximum shrinking ratio is 2.9.
        level = 2. / (self.max_level / level) + 0.9
        return (level,)

    def _enhance_level_to_arg(self, level: float) -> tuple:
        """Converts level to ratio by which we do enhancement on several transformations
        
        Args:
            level (float): magnitude, defined during initialization
            
        Returns:
            tuple: tuple containing the magnitude for the transformation function argument
        """
        return ((level / self.max_level) * 1.8 + 0.1,)

    def _shear_level_to_arg(self, level: float) -> tuple:
        """Mapping from magnitued to transformation argument

        Args:
            level (float): magnitude, defined during initialization

        Returns:
            tuple: tuple containing the magnitude for the transformation function argument
        """
        level = (level / self.max_level) * 0.3
        # Flip level to negative with 50% chance.
        level = self._randomly_negate_tensor(level)
        return (level,)

    def _translate_level_to_arg(self, level: float, translate_const: int) -> tuple:
        """Converts level to ratio by which we translate an image

        Args:
            level (float): magnitude, defined during initialization
            translate_const (int): translate_const set by the level_to_arg function

        Returns:
            tuple: tuple containing the magnitude for the transformation function argument
        """
        level = (level / self.max_level) * float(translate_const)
        # Flip level to negative with 50% chance.
        level = self._randomly_negate_tensor(level)
        return (level,)

    #
    def level_to_arg(self) -> dict:
        """Helper function defining the magnitude to function arguments for the functions to be called

        Returns:
            dict: Dictionary defining the magnitude to function arguments for the functions to be called
        """
        return {
            'AutoContrast': lambda level: (),
            'Equalize': lambda level: (),
            'Invert': lambda level: (),
            'Rotate': self._rotate_level_to_arg,
            'Posterize': lambda level: (int((level / self.max_level) * 4),),
            'Solarize': lambda level: (int((level / self.max_level) * 256),),
            'SolarizeAdd': lambda level: (int((level / self.max_level) * 110),),
            'Color': self._enhance_level_to_arg,
            'Contrast': self._enhance_level_to_arg,
            'Brightness': self._enhance_level_to_arg,
            'Sharpness': self._enhance_level_to_arg,
            'ShearX': self._shear_level_to_arg,
            'ShearY': self._shear_level_to_arg,
            'Cutout': lambda level: (int((level / self.magnitude) * 10),),
            'TranslateX': lambda level: self._translate_level_to_arg(level, 20),
            'TranslateY': lambda level: self._translate_level_to_arg(level, 20),
        }

    def _parse_policy_info(self, op_name: str, prob: tf.float32, level: float,
                           replace: tf.Tensor) -> tuple:
        """Helper function to parse arguments to call the implemented functions

        Args:
            op_name (str): name of the transformation
            prob (tf.float32): probability of the transformation
            level (float): magnitude, defined during initialization
            replace (tf.Tensor): A one or three value 1D tensor to fill empty pixels.

        Returns:
            tuple: tuple containing the magnitude for the transformation function argument
        """
        func = self.__transforms[op_name]
        args = self.level_to_arg()[op_name](level)

        signature = inspect.getfullargspec(func)[0]
        if 'prob' in signature:
            args = tuple([prob] + list(args))

        if 'replace' in signature:
            assert 'replace' == signature[-1]
            args = tuple(list(args) + [replace])

        return (func, prob, args)

    def randaugment_transform(self, image: tf.Tensor) -> tf.Tensor:
        """Applies the RandAugment policy, will cast the given image to uint8.

        Args:
            image (tf.Tensor): 3D or 4D image tensor

        Returns:
            tf.Tensor: transformed image tensor
        """
        image = tf.cast(image, tf.uint8)
        num_layers = self.num_layer
        magnitude = self.magnitude

        replace_value = [128] * 3
        op_str = ""
        for layer_num in range(num_layers):
            op_to_select = tf.random.uniform([], maxval=len(self.op_list), dtype=tf.int32)
            random_magnitude = float(magnitude)
            # Not quite sure why the TPU Tensorflow efficientnet guys did it this way, doesn't seem very efficient TODO: reconsider reimplementing this block
            with tf.name_scope('randaug_layer_{}'.format(layer_num)):
                for (i, op_name) in enumerate(self.op_list):

                    prob = tf.random.uniform([], minval=0.2, maxval=0.8, dtype=tf.float32)
                    func, _, args = self._parse_policy_info(op_name, prob, random_magnitude,
                                                            replace_value)
                    #if i == op_to_select:
                    #    image = self.__transforms[op_name](image, *args)

                    image = tf.cond(tf.equal(i, op_to_select),
                                    lambda selected_func=func, selected_args=args: selected_func(
                                        image, *selected_args),
                                    lambda: image)
                    op_str = tf.cond(tf.equal(i, op_to_select),
                                     lambda op=op_name: self.__set_op(op),
                                     lambda: op_str)
                if (self.__verbose):
                    tf.print(op_str)

        return image

    def __set_op(self, op_name: str) -> None:
        """Prints the policy, helper function

        Args:
            op_name (str): The policy string
        """
        return op_name

    def set_operations(self, op_list: "list[str]") -> None:
        """Sets the list of operations to be performed
        
        Args:
            op_list (list[str]): list of operations, look at RandAugmenter.get_default_operations for the list of operations
        """
        self.op_list = op_list

    @tf.function
    def internal_transform(self, image_and_label: tuple, seed: int) -> tuple:
        """Helper function to transform an image, unpacks (image,label) tuple into image and label.

        Args:
            image_and_label (tuple): Image tensor to transform and its label
            seed (int): Seed for the transformations (deprecated)

        Returns:
            tuple: tuple containing the transformed image and its label
        """
        image, label, *other = image_and_label
        image = self.randaugment_transform(image)

        return image, label

    def transform(self, data_set: tf.data.Dataset, target_size: int) -> tf.data.Dataset:
        """Main transformation method, prefetches dataset and maps the augmentation.

        Args:
            data_set (tf.data.Dataset): The dataset to be is augmented.
            target_size (int): The target size of the augmented dataset.

        Returns:
            tf.data.Dataset: The augmented dataset
        """
        counter = tf.data.experimental.Counter()
        aug_ds = tf.data.Dataset.zip((data_set, (counter, counter)))
        aug_ds = aug_ds.map(self.internal_transform,
                            num_parallel_calls=tf.data.AUTOTUNE).shuffle(target_size)
        return aug_ds

    def self_test(self) -> None:
        """Quick self test for debug purposes
        """
        self.op_list = self.get_default_operations()
        replace_value = [128] * 3
        failed_set_3d = []
        failed_set_4d = []
        # Test operations on 3D tensors
        arr_3d = tf.random.uniform(shape=[224, 224, 3])
        arr_3d = tf.cast(arr_3d, dtype=tf.uint8)
        for (i, op_name) in enumerate(self.op_list):
            prob = tf.random.uniform([], minval=0.2, maxval=0.8, dtype=tf.float32)
            func, _, args = self._parse_policy_info(op_name, prob, self.magnitude, replace_value)
            try:
                arr_3d = func(arr_3d, *args)
            except:
                failed_set_3d.append(op_name)
        # Test operations on 4D tensors
        arr_4d = tf.random.uniform(shape=[32, 224, 224, 3])
        arr_4d = tf.cast(arr_4d, dtype=tf.uint8)
        for (i, op_name) in enumerate(self.op_list):
            prob = tf.random.uniform([], minval=0.2, maxval=0.8, dtype=tf.float32)
            func, _, args = self._parse_policy_info(op_name, prob, self.magnitude, replace_value)
            try:
                arr_4d = func(arr_4d, *args)
            except:
                failed_set_4d.append(op_name)
        print(f"Set of failed 3D operations : {failed_set_3d}")
        print(f"Set of failed 4D operations : {failed_set_4d}")


class RandAugmenterWrapper(tf.keras.layers.Layer):

    def __init__(self, num_layers, magnitude, op_list, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.magnitude = magnitude
        self.op_list = op_list
        self.aug_obj = RandAugmenter(num_layers, magnitude, op_list)

    def augment(self, images):
        images = self.aug_obj.randaugment_transform(images.numpy())
        return images

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "magnitude": self.magnitude,
            "op_list": self.op_list
        })
        return config


if __name__ == '__main__':
    ra = RandAugmenter(2, 9)

    arr = tf.random.uniform(shape=[224, 224, 3])
    arr = ra.randaugment_transform(arr)
    ra.self_test()
import tensorflow as tf


class DataAugmenter:
    """Class gives methods to randomly augment a dataset to a spesific size.
    """

    def color_transform(self, image: tf.Tensor) -> tf.Tensor:
        """Color transformation of images.
        Args:
            image: The tf.Tensor image that is augmented.

        Returns:
            The augmented tf.Tensor image
        """
        if 'split' in self.method.keys():
            self.seed = tf.random.experimental.stateless_split(self.seed,
                                                               self.method['split']['num'])[0, :]
        #Brightness augmentation
        if 'brightness' in self.method.keys():
            image = tf.image.stateless_random_brightness(image,
                                                         self.method['brightness']['max_delta'],
                                                         self.seed)
        #Contrast augmentation
        if 'contrast' in self.method.keys():
            image = tf.image.stateless_random_contrast(image, self.method['contrast']['lower'],
                                                       self.method['contrast']['upper'], self.seed)
        #Saturation aumentation
        if 'saturation' in self.method.keys():
            image = tf.image.stateless_random_saturation(image, self.method['saturation']['lower'],
                                                         self.method['saturation']['upper'],
                                                         self.seed)
        #Hue augmentation
        if 'hue' in self.method.keys():
            image = tf.image.stateless_random_hue(image, self.method['hue']['max_delta'], self.seed)
        return image

    def orientation_transform(self, image: tf.Tensor) -> tf.Tensor:
        """Orientation transformation of images.
        Args:
            image: The tf.Tensor image that is augmented.

        Returns:
            The augmented tf.Tensor image
        """
        if 'split' in self.method.keys():
            self.seed = tf.random.experimental.stateless_split(self.seed,
                                                               self.method['split']['num'])[0, :]
        if 'flip_lr' in self.method.keys():
            image = tf.image.stateless_random_flip_left_right(image, self.seed)
        if 'flip_ud' in self.method.keys():
            image = tf.image.stateless_random_flip_up_down(image, self.seed)
        return image

    def internal_transform(self, image_label: tuple, seed: int) -> tf.Tensor:
        """Internal method which calls the wanted transformation.

        Args:
            image: The tuple of the image that will be augmented.
            seed: The seed for the augmentation.

        Returns:
            The augmented tf.Tensor image from the augmentation methods and the label of the image
        """
        self.seed = seed
        image, label, *other = image_label
        image = self.color_transform(image)
        image = self.orientation_transform(image)
        return image, label

    def _transform(self, image:tf.Tensor, seed: int,**methods:dict) -> tf.Tensor:
        """method which calls the wanted transformation when it is implemented as a layer.

        Args:
            image: The tf.Tensor image that is augmented.
            seed: The seed for the augmentation.

        Returns:
            The augmented tf.Tensor image from the augmentation methods
        """
        self.method = methods
        self.seed = seed
        image = self.color_transform(image)
        image = self.orientation_transform(image)
        return image

    def transform(self, data_set: tf.data.Dataset, seed: int,
                  **methods: dict) -> tf.data.Dataset:
        """Main transformation method, prefetches dataset and maps the augmentation.

        Args:
            dataset: The dataset that is augmented.
            seed: The seed for the augmentation methods.
            methods: a dictionary of the spesified augmentation methods and their parameter values.

        Returns:
            The augmented dataset
        """

        self.method = methods
        self.seed = seed
        AUTOTUNE = tf.data.AUTOTUNE
        counter = tf.data.experimental.Counter()
        aug_ds = tf.data.Dataset.zip((data_set, (counter, counter)))
        aug_ds = (aug_ds.map(self.internal_transform,
                                                  num_parallel_calls=AUTOTUNE))
        return aug_ds

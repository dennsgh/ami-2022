import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa


class Operations:
    """Helper class with operations implemented using TF ops
    """

    def __init__(self) -> None:
        """Init function
        """

        #: dict: dict containing operations implemented in this class
        self.transforms = {
            'AutoContrast': self.auto_contrast,
            'Equalize': self.equalize,
            'Invert': self.invert,
            'Rotate': self.rotate,
            'Posterize': self.posterize,
            'Solarize': self.solarize,
            'SolarizeAdd': self.solarize_add,
            'Color': self.color,
            'Contrast': self.contrast,
            'Brightness': self.brightness,
            'Sharpness': self.sharpness,
            'ShearX': self.shear_x,
            'ShearY': self.shear_y,
            'TranslateX': self.translate_x,
            'TranslateY': self.translate_y,
            'Cutout': self.cutout,
        }

    def identity(self, image: tf.Tensor) -> tf.Tensor:
        """Returns the original tensor as is

        Args:
            image (tf.Tensor): Any tensor

        Returns:
            tf.Tensor: returns the original image
        """
        return image

    def blend(self, image1: tf.Tensor, image2: tf.Tensor, factor: float) -> tf.Tensor:
        """Helper function to blend images based on a given factor

        Args:
            image1 (tf.Tensor): 3D or 4D uint8 image tensor
            image2 (tf.Tensor): 3D or 4D uint8 image tensor
            factor (float): Number between 0.0 and 1.0, A value of 0.0 means only image1 is used. A value greater than 1.0 "extrapolates" the difference between the two pixel values, and we clip the results to values.

        Returns:
            tf.Tensor: A blended image tensor of type uint8.
        """
        if factor == 0.0:
            return tf.convert_to_tensor(image1)
        if factor == 1.0:
            return tf.convert_to_tensor(image2)

        image1 = tf.cast(image1, dtype=tf.float32)
        image2 = tf.cast(image2, dtype=tf.float32)

        difference = image2 - image1
        scaled = factor * difference

        # Do addition in float.
        temp = tf.cast(image1, dtype=tf.float32) + scaled

        # Interpolate
        if factor > 0.0 and factor < 1.0:
            # Interpolation means we always stay within 0 and 255.
            return tf.cast(temp, tf.uint8)

        # Extrapolate:
        #
        # We need to clip and then cast.
        return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)

    def cutout(self, image: tf.Tensor, pad_size: int, replace_value: int = 0) -> tf.Tensor:
        """Applies cutout to the image

        Args:
            image (tf.Tensor): 3D or 4D uint8 image tensor
            pad_size (int): Specifies how big the zero mask to be generated, mask will be (2*pad_size x 2*pad_size)
            replace_value (int, optional): Pixel value to fill in the cutout mask. Defaults to 0.

        Returns:
            tf.Tensor: Image with cutout applied
        """
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        # Sample the center location in the image where the zero mask will be applied.
        cutout_center_height = tf.random.uniform(shape=[],
                                                 minval=0,
                                                 maxval=image_height,
                                                 dtype=tf.int32)

        cutout_center_width = tf.random.uniform(shape=[],
                                                minval=0,
                                                maxval=image_width,
                                                dtype=tf.int32)

        lower_pad = tf.maximum(0, cutout_center_height - pad_size)
        upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
        left_pad = tf.maximum(0, cutout_center_width - pad_size)
        right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

        cutout_shape = [
            image_height - (lower_pad + upper_pad), image_width - (left_pad + right_pad)
        ]
        padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
        mask = tf.pad(tf.zeros(cutout_shape, dtype=image.dtype), padding_dims, constant_values=1)
        mask = tf.expand_dims(mask, -1)
        mask = tf.tile(mask, [1, 1, 3])
        image = tf.where(tf.equal(mask, 0),
                         tf.ones_like(image, dtype=image.dtype) * replace_value, image)
        return image

    def solarize(self, image: tf.Tensor, threshold: int = 128) -> tf.Tensor:
        """Pixels with values greater than or equal to the threshold will be inverted

        Args:
            image (tf.Tensor): _description_
            threshold (int, optional): _description_. Defaults to 128.

        Returns:
            tf.Tensor: Solarized image tensor
        """

        return tf.where(image < threshold, image, 255 - image)

    def solarize_add(self, image: tf.Tensor, addition: int = 0, threshold: int = 128) -> tf.Tensor:
        """For each pixel in the image less than threshold 'addition' amount is added to it and then clip the pixel value to be between 0 and 255.
        
        Args:
            image (tf.Tensor): 3D or 4D uint8 image tensor
            addition (int, optional): delta to add to pixels less than threshold. The value of 'addition' should be between -128 and 128. Defaults to 0.
            threshold (int, optional): operation threshold. Defaults to 128.

        Returns:
            tf.Tensor: Solarized image Tensor
        """

        added_image = tf.cast(image, tf.int64) + addition
        added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)
        return tf.where(image < threshold, added_image, image)

    def color(self, image: tf.Tensor, factor: float) -> tf.Tensor:
        """Equivalent of PIL Color implemented with TF ops

        Args:
            image (tf.Tensor): 3D or 4D uint8 image tensor
            factor (float): number between 0 and 1 reflecting the strength of the transformation

        Returns:
            tf.Tensor: An image tensor with its grayscale image blended with the original image based on the set magnitude.
        """
        degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
        return self.blend(degenerate, image, factor)

    def contrast(self, image: tf.Tensor, factor: float) -> tf.Tensor:
        """Equivalent of PIL Contrast

        Args:
            image (tf.Tensor): 3D or 4D uint8 image tensor
            factor (float): number between 0 and 1 reflecting the strength of the transformation

        Returns:
            tf.Tensor: An image tensor with modified contrast blended with the original image based on the set magnitude.
        """
        degenerate = tf.image.rgb_to_grayscale(image)
        # Cast before calling tf.histogram.
        degenerate = tf.cast(degenerate, tf.int32)

        # Compute the grayscale histogram, then compute the mean pixel value,
        # and create a constant image size of that value.  Use that as the
        # blending degenerate target of the original image.
        hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
        mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
        degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
        degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
        degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
        return self.blend(degenerate, image, factor)

    def brightness(self, image: tf.Tensor, factor: float) -> tf.Tensor:
        """Equivalent of PIL brightness

        Args:
            image (tf.Tensor): 3D or 4D image tensor
            factor (float): number between 0 and 1 reflecting the strength of the transformation

        Returns:
            tf.Tensor: image tensor with brightness modified
        """
        degenerate = tf.zeros_like(image)
        return self.blend(degenerate, image, factor)

    def posterize(self, image: tf.Tensor, bits: int) -> tf.Tensor:
        """Equivalent of PIL Posterize.

        Args:
            image (tf.Tensor): 3D or 4D image tensor
            bits (int): number of bits reduced for each color channel

        Returns:
            tf.Tensor: posterized image tensor
        """
        shift = 8 - bits
        return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)

    def rotate(self, image: tf.Tensor, degrees: float) -> tf.Tensor:
        """Rotates an image tensor

        Args:
            image (tf.Tensor): 3D or 4D tensors
            degrees (float): amount of degrees to rotate

        Returns:
            tf.Tensor: Rotated image tensor
        """
        # Convert from degrees to radians.
        radians = degrees * np.pi / 180.0
        image = tfa.image.rotate(image, radians)
        return image

    def translate_x(self, image: tf.Tensor, pixels: int) -> tf.Tensor:
        """Translates an image in the X axis
        
        Args:
            image (tf.Tensor): 3D or 4D uint8 image tensor
            pixels (int): number of pixels to translate in the x direction
            
        Returns:
            tf.Tensor: The translated image tensor
        """
        image = tfa.image.translate(image,
                                    translations=[pixels, 0],
                                    interpolation='bilinear',
                                    fill_mode='constant',
                                    fill_value=0)

        return image

    def translate_y(self, image: tf.Tensor, pixels: int) -> tf.Tensor:
        """Translates an image in the Y axis
        
        Args:
            image (tf.Tensor): 3D or 4D uint8 image tensor
            pixels (int): number of pixels to translate in the y direction
            
        Returns:
            tf.Tensor: The translated image tensor
        """
        image = tfa.image.translate(image,
                                    translations=[0, pixels],
                                    interpolation='bilinear',
                                    fill_mode='constant',
                                    fill_value=0)
        return image

    def shear_x(self, image: tf.Tensor, pixels: int, replace: tf.Tensor) -> tf.Tensor:
        """Shears an image in the x direction

        Args:
            image (tf.Tensor): 3D uint8 image tensors
            pixels (int): number of pixels to shear
            replace (tf.Tensor): A one or three value 1D tensor to fill empty pixels.

        Returns:
            tf.Tensor: Sheared image tensor
        """
        R = -0.3 + 2 * pixels * 0.3
        image = tfa.image.shear_x(image, level=R, replace=replace)
        return image

    def shear_y(self, image: tf.Tensor, pixels: int, replace: tf.Tensor) -> tf.Tensor:
        """Shears an image in the y direction

        Args:
            image (tf.Tensor): 3D uint8 image tensors
            pixels (int): number of pixels to shear
            replace (tf.Tensor): A one or three value 1D tensor to fill empty pixels.

        Returns:
            tf.Tensor: Sheared image tensor
        """
        R = -0.3 + 2 * pixels * 0.3
        image = tfa.image.shear_y(image, level=R, replace=replace)
        return image

    def auto_contrast(self, image: tf.Tensor) -> tf.Tensor:
        """AutoContrast algorithm taken from efficientnet/autoaugment.py

        Args:
            image (tf.Tensor):3D uint8 image tensor

        Returns:
            tf.Tensor: Image tensor with autocontrast applied
        """

        def scale_channel(image):
            """Scale the 2D image using the autocontrast rule."""
            # A possibly cheaper version can be done using cumsum/unique_with_counts
            # over the histogram values, rather than iterating over the entire image.
            # to compute mins and maxes.
            lo = tf.cast(tf.reduce_min(image), dtype=tf.float32)
            hi = tf.cast(tf.reduce_max(image), dtype=tf.float32)

            # Scale the image, making the lowest value 0 and the highest value 255.
            def scale_values(im):
                scale = 255.0 / (hi - lo)
                offset = -lo * scale
                im = tf.cast(im, dtype=tf.float32) * scale + offset
                im = tf.clip_by_value(im, 0.0, 255.0)
                return tf.cast(im, tf.uint8)

            result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
            return result

        # Assumes RGB for now.  Scales each channel independently
        # and then stacks the result.
        s1 = scale_channel(image[:, :, 0])
        s2 = scale_channel(image[:, :, 1])
        s3 = scale_channel(image[:, :, 2])
        image = tf.stack([s1, s2, s3], 2)
        return image

    def sharpness(self, image: tf.Tensor, factor: float) -> tf.Tensor:
        """Equivalent of sharpness transform from PIL using TF ops.

        Args:
            image (tf.Tensor): 3D uint8 image tensor
            factor (float): number between 0 and 1 reflecting the strength of the transformation

        Returns:
            tf.Tensor: Image tensor with sharpness raised.
        """
        orig_image = image
        image = tf.cast(image, tf.float32)
        # Make image 4D for conv operation.
        image = tf.expand_dims(image, 0)
        # SMOOTH PIL Kernel.
        kernel = tf.constant(
            [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32, shape=[3, 3, 1, 1]) / 13.
        # Tile across channel dimension.
        kernel = tf.tile(kernel, [1, 1, 3, 1])
        strides = [1, 1, 1, 1]
        with tf.device('/cpu:0'):
            # Some augmentation that uses depth-wise conv will cause crashing when
            # training on GPU. See (b/156242594) for details.
            degenerate = tf.nn.depthwise_conv2d(image, kernel, strides, padding='VALID')
        degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
        degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

        # For the borders of the resulting image, fill in the values of the
        # original image.
        mask = tf.ones_like(degenerate)
        padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
        padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
        result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

        # Blend the final result.
        return self.blend(result, orig_image, factor)

    def equalize(self, image: tf.Tensor) -> tf.Tensor:
        """Equivalent of equalize function from PIL using TF ops.

        Args:
            image (tf.Tensor): 3D or 4D uint8 image tensor.

        Returns:
            tf.Tensor: Equalized image tensor
        """

        def scale_channel(im, c):
            """Scale the data in the channel to implement equalize."""
            im = tf.cast(im[:, :, c], tf.int32)
            # Compute the histogram of the image channel.
            histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

            # For the purposes of computing the step, filter out the nonzeros.
            nonzero = tf.where(tf.not_equal(histo, 0))
            nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
            step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

            def build_lut(histo, step):
                # Compute the cumulative sum, shifting by step // 2
                # and then normalization by step.
                lut = (tf.cumsum(histo) + (step // 2)) // step
                # Shift lut, prepending with 0.
                lut = tf.concat([[0], lut[:-1]], 0)
                # Clip the counts to be in range.  This is done
                # in the C code for image.point.
                return tf.clip_by_value(lut, 0, 255)

            # If step is zero, return the original image.  Otherwise, build
            # lut from the full histogram and step and then index from it.
            result = tf.cond(tf.equal(step, 0), lambda: im,
                             lambda: tf.gather(build_lut(histo, step), im))

            return tf.cast(result, tf.uint8)

        # Assumes RGB for now.  Scales each channel independently
        # and then stacks the result.
        s1 = scale_channel(image, 0)
        s2 = scale_channel(image, 1)
        s3 = scale_channel(image, 2)
        image = tf.stack([s1, s2, s3], 2)
        return tf.cast(image, tf.uint8)

    def invert(self, image: tf.Tensor) -> tf.Tensor:
        """Inverts the image pixels.

        Args:
            image (tf.Tensor): 3D or 4D uint8 image tensor

        Returns:
            tf.Tensor: Returns an inverted image tensor 
        """
        image = tf.convert_to_tensor(image)
        return 255 - image

    def wrap(self, image: tf.Tensor) -> tf.Tensor:
        """Pads an image with an extra channel with set to all 1s

        Args:
            image (tf.Tensor): 3D uint8 image tensor

        Returns:
            tf.Tensor: Image tensor with an extra channel
        """
        shape = tf.shape(image)
        extended_channel = tf.ones([shape[0], shape[1], 1], image.dtype)
        extended = tf.concat([image, extended_channel], 2)
        return extended

    def unwrap(self, image: tf.Tensor, replace: tf.Tensor) -> tf.Tensor:
        """Unwraps an image produced by wrap. (deprecated)

        Args:
            image (tf.Tensor): 3D image tensor with 4 channels produced by wrap
            replace (tf.Tensor): A one or three value 1D tensor to fill empty pixels.

        Returns:
            tf.Tensor: A 3D image tensor with 3 channels.
        """
        image_shape = tf.shape(image)
        # Flatten the spatial dimensions.
        flattened_image = tf.reshape(image, [-1, image_shape[2]])

        # Find all pixels where the last channel is zero.
        alpha_channel = flattened_image[:, 3]

        replace = tf.concat([replace, tf.ones([1], image.dtype)], 0)

        # Where they are zero, fill them in with 'replace'.
        flattened_image = tf.where(tf.equal(alpha_channel, 0),
                                   tf.ones_like(flattened_image, dtype=image.dtype) * replace,
                                   flattened_image)
        image = tf.reshape(flattened_image, image_shape)
        image = tf.slice(image, [0, 0, 0], [image_shape[0], image_shape[1], 3])

        return image
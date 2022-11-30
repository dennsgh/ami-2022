import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy


class PiecewiseConstantDecayWithLinearWarmup(tf.keras.optimizers.schedules.PiecewiseConstantDecay):

    def __init__(self, warmup_learning_rate, warmup_steps, boundaries, values, **kwargs):
        super(PiecewiseConstantDecayWithLinearWarmup, self).__init__(boundaries=boundaries,
                                                                     values=values,
                                                                     **kwargs)

        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self._step_size = self.values[0] - self.warmup_learning_rate

    def __call__(self, step):
        with tf.name_scope(self.name or 'PiecewiseConstantDecayWithLinearWarmup'):
            learning_rate = tf.cond(
                pred=tf.less(step, self.warmup_steps),
                true_fn=lambda: (self.warmup_learning_rate + tf.cast(step, dtype=tf.float32) / self.
                                 warmup_steps * self._step_size),
                false_fn=lambda:
                (super(PiecewiseConstantDecayWithLinearWarmup, self).__call__(step)))
        return learning_rate

    def get_config(self):
        config = {
            "warmup_learning_rate": self.warmup_learning_rate,
            "warmup_steps": self.warmup_steps,
        }
        base_config = super(PiecewiseConstantDecayWithLinearWarmup, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def custom_crossentropy(y, y_hat):
    """Custom function for adding label smoothing to sparse categorical crossentropy
    
    Args:
        y (float): True label as sparse value.
        y_hat (float): Predicted labels as #classes-dimensional vector.

    Returns:
        loss: Return categorical crossentropy with label smoothing.  
    
    """
    n_classes = 4

    y = tf.one_hot(tf.cast(y, tf.int32), n_classes)
    y_hat = tf.expand_dims(y_hat, axis=1)

    return categorical_crossentropy(y, y_hat, label_smoothing=0.05)


class RandomColorAffine(tf.keras.layers.Layer):

    def __init__(self, brightness=0, jitter=0, **kwargs):
        super().__init__(**kwargs)

        self.brightness = brightness
        self.jitter = jitter

    def get_config(self):
        config = super().get_config()
        config.update({"brightness": self.brightness, "jitter": self.jitter})
        return config

    def call(self, images, training=True):
        if training:
            batch_size = tf.shape(images)[0]

            # Same for all colors
            brightness_scales = 1 + tf.random.uniform(
                (batch_size, 1, 1, 1), minval=-self.brightness, maxval=self.brightness)
            # Different for all colors
            jitter_matrices = tf.random.uniform((batch_size, 1, 3, 3),
                                                minval=-self.jitter,
                                                maxval=self.jitter)

            color_transforms = (tf.eye(3, batch_shape=[batch_size, 1]) * brightness_scales +
                                jitter_matrices)
            images = tf.clip_by_value(tf.matmul(images, color_transforms), 0, 1)
        return images


# Image augmentation module
def get_augmenter(min_area, brightness, jitter):

    image_size = 80
    image_channels = 3
    zoom_factor = 1.0 - np.sqrt(min_area)
    return tf.keras.Sequential([
        tf.keras.Input(shape=(image_size, image_size, image_channels)),
        tf.keras.layers.Rescaling(1 / 255),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomTranslation(zoom_factor / 2, zoom_factor / 2),
        tf.keras.layers.RandomZoom((-zoom_factor, 0.0), (-zoom_factor, 0.0)),
        RandomColorAffine(brightness, jitter),
    ])

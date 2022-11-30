import resnet_cifar
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

N = 2
DEPTH = N * 9 + 2
NUM_BLOCKS = ((DEPTH - 2) // 9) - 1
PROJECT_DIM = 2048
WEIGHT_DECAY = 0.0005
LATENT_DIM = 4
CROP_TO = 80

#Set up Self-supervised model (encoder + predictor), Use Encoder as backbone later
def get_encoder():
    """Generates encoder for the Self-supervised learning model based on ResNet, a part of the encoder will be used for feature extraction in the final classifier

    Returns:
        TF Model: returns encoder model
    """
    # Input and backbone.
    inputs = layers.Input((CROP_TO, CROP_TO, 3))
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1)(inputs)
    x = resnet_cifar.stem(x)
    x = resnet_cifar.learner(x, NUM_BLOCKS)
    x = layers.GlobalAveragePooling2D(name="backbone_pool")(x)

    # Projection head.
    x = layers.Dense(PROJECT_DIM, use_bias=False,
                     kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(PROJECT_DIM, use_bias=False,
                     kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x)
    outputs = layers.BatchNormalization()(x)
    return tf.keras.Model(inputs, outputs, name="encoder")


def get_predictor():
    """Generates predictor for the Self-supervised learning model, this part won't be used for the backbone during the final classification

    Returns:
        TF Model: returns predictor model
    """
    model = tf.keras.Sequential(
        [
            # Note the AutoEncoder-like structure.
            layers.Input((PROJECT_DIM,)),
            layers.Dense(
                LATENT_DIM,
                use_bias=False,
                kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
            ),
            layers.ReLU(),
            layers.BatchNormalization(),
            layers.Dense(PROJECT_DIM),
        ],
        name="predictor",
    )
    return model


def compute_loss(p, z):
    """The authors of SimSiam emphasize the impact of
    the `stop_gradient` operator in the paper as it
    has an important role in the overall optimization.

    Args:
        p (tensorflow.python.framework.ops.Tensor): encoder output
        z (tensorflow.python.framework.ops.Tensor): prediction output

    Returns:
        tensorflow.python.framework.ops.Tensor: Negative cosine similarity (minimizing this is equivalent to maximizing the similarity).
    """
    z = tf.stop_gradient(z)
    p = tf.math.l2_normalize(p, axis=1)
    z = tf.math.l2_normalize(z, axis=1)
    return -tf.reduce_mean(tf.reduce_sum((p * z), axis=1))


class SimSiam(tf.keras.Model):
    """Generate the SimSiam Setup

    Args:
        tf (tf.keras.Model): SimSiam model input (encoder+predictor)
    """

    def __init__(self, encoder, predictor):
        """initialize SimSiam

        Args:
            encoder (tf.keras.Model): encoder model
            predictor (tf.keras.Model): predictor model
        """
        super(SimSiam, self).__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        """Returns tracked metric
        Returns:
            loss: returns loss tracker
        """
        return [self.loss_tracker]

    def train_step(self, data):
        """Run one training step with two augmented images

        Args:
            data : zipped data of two augmented images

        Returns:
            loss: negative cosine similarity loss
        """
        # Unpack the data.
        ds_one, ds_two = data

        # Forward pass through the encoder and predictor.
        with tf.GradientTape() as tape:
            z1, z2 = self.encoder(ds_one), self.encoder(ds_two)
            p1, p2 = self.predictor(z1), self.predictor(z2)
            # Note that here we are enforcing the network to match
            # the representations of two differently augmented batches
            # of data.
            loss = compute_loss(p1, z2) / 2 + compute_loss(p2, z1) / 2
            #std = compute_std(p1, z2) / 2 + compute_std(p2, z1) / 2
        # Compute gradients and update the parameters.
        learnable_params = (self.encoder.trainable_variables + self.predictor.trainable_variables)
        gradients = tape.gradient(loss, learnable_params)
        self.optimizer.apply_gradients(zip(gradients, learnable_params))

        # Monitor loss.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}  #, "std": std}


def train_test_split_array(array, labels, train_split):
    """splits dataset in train and test

    Args:
        array : data array
        labels : label array
        train_split (float): split, e.g. 0.2 --> 20% test, 80% train

    Returns:
        splitted array in train image, test image, train label and test label
    """
    train_samples = int(array.shape[0] * train_split)
    train_images = array[:train_samples, :, :, :]
    test_images = array[train_samples:, :, :, :]
    train_labels = labels[:train_samples,]
    test_labels = labels[train_samples:,]
    return train_images, test_images, train_labels, test_labels

import numpy as np
import tensorflow as tf


class AbstractPipeline():
    """Interface class for pipelines.
    """

    def __init__(self,
                 batch_size: int,
                 compiler_config: dict,
                 epochs: int,
                 model: object,
                 model_name: str,
                 callbacks: list = [],
                 custom_objects: dict = {},
                 patience: int = 10,
                 test_fraction_split: float = 0.,
                 validation_fraction_split: float = 0.) -> None:
        """Initialize the hyperparameters of your pipeline

        Args:
            batch_size (int): 
            compiler_config (dict): Kwargs for compiler
            epochs (int): 
            model (object): Uncompiled model
            model_name (str): Model name to differentiate between runs
            callbacks (list, optional): Additional callbacks
            custom_objects (dict, optional): For restoring saved model. Defaults to {}.
            patience (int, optional): For early stopping. Defaults to 10.
            test_fraction_split (float, optional): Fraction of data for test. Defaults to 0..
            validation_fraction_split (float, optional): Fraction fo data for validation. Defaults to 0..
        """

        self.batch_size = batch_size
        self.compiler_config = compiler_config
        self.epochs = epochs
        self.model = model
        self.model_name = model_name
        self.callbacks = callbacks
        self.custom_objects = custom_objects
        self.patience = patience
        self.test_fraction_split = test_fraction_split
        self.validation_fraction_split = validation_fraction_split

    def train(self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset) -> dict:
        """Train the model with the provided pipeline hyperparameters and return the training
           history.

        Args:
            train_ds (tf.data.Dataset): Training dataset
            val_ds (tf.data.Dataset): Validation dataset

        Raises:
            NotImplementedError: Implementation mandatory

        Returns:
            dict: history.history dictionary
        """

        raise NotImplementedError
        return history

    def fit(self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset) -> dict:
        """Fit the pipeline, store the fitted model as attribute and return the training history.

        Args:
            train_ds (tf.data.Dataset): Training dataset
            val_ds (tf.data.Dataset): Validation dataset

        Returns:
            dict: Dictionary with training history
        """
        train_ds = self.augment(train_ds)
        train_ds = self.transfrom(train_ds)
        history = self.train(train_ds, val_ds)

        return history

    def transfrom(self, train_ds: tf.data.Dataset) -> tf.data.Dataset:
        """Transform the data to fit model input, e.g. rescaling image size or pixel intensity.

        Args:
            train_ds (tf.data.Dataset): Training dataset

        Raises:
            NotImplementedError: Implementation mandatory.

        Returns:
            tf.data.Dataset: Transformed dataset as fed into the model.
        """
        raise NotImplementedError
        return transfromed_ds

    def augment(self, train_ds: tf.data.Dataset) -> tf.data.Dataset:
        """Perform data augmentation on provided dataset.

        Args:
            train_ds (tf.data.Dataset): Training dataset

        Raises:
            NotImplementedError: Implementation optional.

        Returns:
            tf.data.Dataset: Augmented dataset
        """
        raise NotImplementedError
        return augmented_ds

    def prepend_layer(self, layer: object):
        """Prepend a layer to model. Done for preprocessing or data augmentation.

        Args:
            layer (object): Layer to be prepended.
        """
        inputs = tf.keras.Input(shape=(*self.model.input_shape, 3))
        x = layer(inputs)
        outputs = self.model.call(x)
        self.model = tf.keras.Model(inputs, outputs)
        return

    def score(self, test_ds: tf.data.Dataset) -> list:
        """Evaluate the model and return the model scores.

        Args:
            test_ds (tf.data.Dataset): Test data set

        Returns:
            list: Scores on validation loss and metrics
        """
        test_ds = self.transfrom(test_ds)
        score = self.model.evaluate(test_ds)

        return score

    def predict(self, X: tf.Tensor) -> np.ndarray:
        """ Predict samples and return prediction probabilities.

        Args:
            X (np.array | tfds | tf.Tensor): sample data

        Raises:
            NotImplementedError:
        """
        return self.model.predict(X)

    def predict_class(self, X: tf.Tensor) -> np.ndarray:
        """ Predict samples and return predicted classes.

        Args:
            X (np.array | tfds | tf.Tensor): sample data
        """
        return np.argmax(self.predict(X), axis=1)

import traceback

import cv2
import numpy as np
from flask import request

from model_containers import SelfSupModelContainer, TransferModelContainer


class ModelManager():
    """Model handler class for the backend.
    """

    def __init__(self, mode: str = None) -> None:
        """Init function

        Args:
            mode (str, optional): _description_. Defaults to the first loaded model if not specified.

        Raises:
            ValueError: Invalid mode string specified
        """
        self.__load_models()  # Raises a RuntimeError exception if 0 models are loaded.
        self.model_dict = {}
        self.implemented_predicts = {
            "selfsup": self.__predict_selfsup,
            "transfer": self.__predict_transfer
        }
        self.valid_modes = list(self.implemented_predicts.keys())

        if mode not in self.valid_modes:
            print("Invalid/No mode string specified, will default to first loaded model...")

        for m, container_obj in self.__loaded_models.items():
            # load first model which is not none
            if (container_obj):
                self.mode = m
                break
        # if user specified something then set it to that instead
        if (mode in self.valid_modes and self.__loaded_models[mode]):
            self.mode = mode

        print("Mode set to {}.".format(self.mode))

    def __load_models(self) -> None:
        """Internal loading function for loading models from the containers

        Raises:
            RuntimeError: If no models at all are loaded
        """
        self.__loaded_models = {}
        err = ""
        try:
            selfsup_container = SelfSupModelContainer()
            print("[LOAD] Successfully loaded Self-supervised model")
            self.__loaded_models["selfsup"] = selfsup_container
            self.label_dict = {0: "scratch", 1: "dent", 2: "other", 3: "rim"}
        except Exception:
            print("Failed to load self-supervised model...")
            self.__loaded_models["selfsup"] = None
            err = err + traceback.format_exc()

        try:
            transfer_container = TransferModelContainer()
            print("[LOAD] Successfully loaded Transfer Learning model")
            self.transfer_label_dict = {0: "dent", 1: "other", 2: "rim", 3: "scratch"}
            self.__loaded_models["transfer"] = transfer_container
        except Exception:
            print("Failed to load Transfer Learning model...")
            self.__loaded_models["transfer"] = None
            err = err + traceback.format_exc()

        if all(i is None for i in list(self.__loaded_models.values())):
            print(err)
            raise RuntimeError('Model loading failure has occured')

    def get_valid_modes(self) -> list:
        """returns list of valid modes

        Returns:
            list: List of valid modes
        """
        return self.valid_modes

    def set_mode(self, mode: str) -> bool:
        """sets the model to use for prediction

        Args:
            mode (str): mode of prediction i.e. which model should be used
        """
        if mode in self.valid_modes:
            self.mode = mode
            return True

        return False

    def predict(self, image) -> list:
        label = ""
        if (self.__loaded_models[self.mode]):
            label = self.implemented_predicts[self.mode](image)
        else:
            label = f"Model container is {self.__loaded_models[self.mode]} , likely not loaded!"
            print(label)
        return label

    def __predict_selfsup(self, image) -> str:
        """function to predict using self supervised learning

        Args:
            image (_type_): numpy array to predict

        Returns:
            str: label
        """
        image = cv2.resize(image, (80, 80))
        image = np.expand_dims(np.array(image), 0).tolist()
        prediction = ""
        prediction = self.__loaded_models["selfsup"].model.predict(image)
        class_id = np.argmax(prediction)
        label = self.label_dict[class_id]

        return label

    def __predict_transfer(self, image) -> str:
        """function to predict using transfer learning

        Args:
            image (_type_): numpy array to predict

        Returns:
            str: label
        """
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(np.array(image), 0).tolist()
        prediction = ""
        prediction = self.__loaded_models["transfer"].predict(image)[0]
        class_id = np.argmax(prediction)
        label = self.transfer_label_dict[class_id]

        return label

    def get_loaded_models(self) -> dict:
        """Returns the dictionary pointing to the container objects

        Returns:
            dict: dictionary with container objects
        """
        return self.__loaded_models

    def summary(self) -> None:
        """prints the loaded models by the ModelManager instance
        """
        for mode, container_obj in self.__loaded_models.items():
            if (container_obj):
                print(mode)

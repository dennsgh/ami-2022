import json
import math
import os
from random import random, seed

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from flask import Flask, flash, jsonify, request, session
from model_manager import ModelManager
from utils.file_system import load_json, update_json

data_set_chunk = []
labels_chunk = []
label_dict = {0: "dent", 1: "other", 2: "rim", 3: "scratch"}
reverse_label_dict = {"dent": 0, "other": 1, "rim": 2, "scratch": 3}
labels = ["Dent", "Scratch", "Other", "Rim"]


def create_app(test_config=None):
    annotations = []
    reverse_label_dict = {"dent": 0, "other": 1, "rim": 2, "scratch": 3}

    model_manager = ModelManager(mode="transfer")
    model_manager.summary()

    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )
    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/api/predict', methods=['GET', 'POST'])
    def upload():
        """
            This function is a service that the server provides for prediction.+

            Args: POST request

            Returns:
                  label as json response.
        """
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file part')
            print("processing request...")
            filestr = request.files['file'].read()
            npimg = np.fromstring(filestr, np.uint8)
            image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            print("performing inference...")
            # pass image to model_manager
            print(f'Usind model : {model_manager.mode}...')
            label = model_manager.predict(image)
            response = {"file_name": request.files['file'].filename, "label": f'{label}'}
            print("sending results to client...")
            return response

    @app.route('/api/mode', methods=['GET', 'POST'])
    def mode_switch():
        """
            Mode switch

            Args: POST request

            Returns:
                OK on success
        """
        response = ""
        if request.method == 'POST':
            if 'mode' not in request.files:
                flash('No mode was specified')
            print("processing request...")
            mode = request.get_json()

            if (model_manager.set_mode(mode)):
                response = jsonify("OK")

        return response

    @app.route('/api/export', methods=['GET', 'POST'])
    def export():
        """
            This function is a service that the server provides for exporting annotations

            Args: GET request

            Returns:
                  annotations as json response.
        """
        if request.method == 'GET':
            return jsonify(json.dumps(annotations))

    @app.route('/api/label', methods=['GET', 'POST'])
    def label():
        """
            This function is a service that the server provides for keeping track of the labels chosed by the user
            in the frontend. the result is saved in a list that can be exported

            Args: POST request

            Returns:
                  annotations as json response.
        """
        if request.method == 'POST':
            ## take request content and append it to a live labels object
            print(request.headers.get('Content-Type'))
            image_label = request.get_json()
            image_exists = False
            index = 0
            print(image_label['file_name'])
            for index, annotation in enumerate(annotations):
                if annotation['file_name'] == image_label['file_name']:
                    image_exists = True
                    index = index
            if (not image_exists):
                annotations.append(image_label)
            else:
                annotations[index] = image_label
        return jsonify("OK")

    @app.route('/api/append_data', methods=['GET', 'POST'])
    def append_data():
        """
                    something
        """
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file part')
            image_dict = {}
            print(request)
            image_dict['name'] = request.files['file'].filename
            print(request.files['file'].filename)
            filestr = request.files['file'].read()
            npimg = np.fromstring(filestr, np.uint8)
            image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            image_dict['data'] = image
            data_set_chunk.append(image_dict)

        return jsonify("OK")

    @app.route('/api/append_label', methods=['GET', 'POST'])
    def append_label():
        """
        something

        """
        if request.method == 'POST':
            ## take request content and append it to a live labels object
            image_label = request.get_json()
            labels_chunk.append(image_label)

        return jsonify("OK")

    @app.route('/api/retrain', methods=['GET', 'POST'])
    def retrain():
        """
        something
        """

        data_config = Path(os.getenv("CONFIG"), "data_config.json")
        anotation_location = load_json(data_config)["anotation_location"]
        anotation_location = Path(os.getenv("DATA"), anotation_location)
        image_location = Path(anotation_location.parent, "cropped")

        flag = False
        if request.method == 'GET':
            ## take request content and append it to a live labels object
            if (len(data_set_chunk) != 0):
                try:
                    annotations_file = open(anotation_location)
                except Exception as e:
                    print("Counld Not open Annotatinos file!")
                    print(e)
                annotation_entry = {}
                annotations_json = json.load(annotations_file)

                for image in data_set_chunk:
                    random_id = random() * 10000000
                    random_id = math.trunc(random_id)
                    annotation_entry['id'] = random_id
                    image_name = image['name']
                    file_name = 'file_name'
                    label = ""
                    for file in labels_chunk:
                        if file['file_name'] == image_name:
                            try:
                                label = file['label']
                                annotation_entry['category_id'] = reverse_label_dict[label.lower()]
                            except KeyError:
                                print("Error while readong label key. Moving on...")
                                pass

                    #file_path = f'{IMAGE_SAVE_PATH}{label}/{image_name}'
                    file_path = Path(image_location, str(random_id)).with_suffix(".png")
                    print(file_path)
                    annotation_entry['filepath'] = str(Path(image_location.name,
                                                        str(random_id)).with_suffix(".png"))
                    print(annotation_entry)
                    annotations_json['annotations'].append(annotation_entry)
                    cv2.imwrite(str(file_path), image['data'])
                update_json(anotation_location, annotations_json)
                #
                # Here goes the training stuff............
                #
                print(model_manager.mode)
                model_manager.set_mode("transfer")
                container = model_manager.get_loaded_models()["transfer"]
                container.retrain()
            else:
                flag = True
        if flag:
            return jsonify("Empty Image Cache")
        else:
            return jsonify("OK")

    app.logger.info("App loaded successfully.")

    return app


app = create_app()

import tensorflow as tf
import pdb
import os
from preprocessing.data_loader import DataLoader
from pathlib import Path
from pipelines.ilsvrc import EfficientNetPipeline
from models.ilsvrc import EfficientNetV2S, EfficientNetV2M
from optimization.ilsvrc import ILSVRCOptunaSearch

model_config = {
    "layer_size": {
        "type": "int",
        "low": 16,
        "high": 4096,
        "step": 16
    },
    "dropout": {
        "type": "float",
        "low": 0.0,
        "high": 0.6,
    },
    "pooling_type": {
        "type": "categorical",
        "choices": ["max", "avg"]
    },
    "depth": {
        "type": "int",
        "low": 0,
        "high": 4,
    },
    "learning_rate": {
        "feature_extraction": {
            "type": "float",
            "low": 1e-6,
            "high": 1e-3,
        },
        "fine_tuning": {
            "type": "float",
            "low": 1e-10,
            "high": 1e-3,
        },
    }
}

pipeline_config = {
    "batch_size": {
        "feature_extraction": {
            "type": "int",
            "low": 8,
            "high": 32,
            "step": 8
        },
        "fine_tuning": {
            "type": "int",
            "low": 8,
            "high": 128,
            "step": 16
        },
    },
    "epochs": 100,
    "model_name": "trial",
    "store_model": False,
    "save_weights_only": False,
    "patience": 6,
    "compiler_config": {
        "optimizer": "adam",
        "loss": "sparse_categorical_crossentropy",
        "metrics": ['accuracy'],
        "weighted_metrics": []
    },
    "lr_decay": {
        "feature_extraction": {
            "type": "float",
            "low": 0.9,
            "high": 1,
        },
        "fine_tuning": {
            "type": "float",
            "low": 0.9,
            "high": 1,
        },
    }
}

image_path = Path(os.getenv("DATA"), 'sort')
data_loader = DataLoader(json_directory=Path(os.getenv("DATA"),
                                             'restructured_w_original_labels.json'),
                         val_size=0.2,
                         test_size=0.2,
                         perf_crop=True,
                         image_size=[224, 224],
                         resizing_type='pad',
                         batch_size=None,
                         class_weights=[0.25, 0.25, 0.25, 0.25],
                         data_type=tf.data.Dataset)

train_ds, test_ds, val_ds = data_loader.run()

opt = ILSVRCOptunaSearch(name="new_setup",
                         n_trials=150,
                         fine_tuning_iterations=4,
                         model=EfficientNetV2M,
                         pipeline=EfficientNetPipeline,
                         model_config=model_config,
                         pipeline_config=pipeline_config,
                         objective="val_loss",
                         pruners=True)

opt.run(train_ds, val_ds)

pdb.set_trace()
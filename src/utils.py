import tensorflow as tf
from datetime import datetime
import json
import os
import pandas as pd

from tensorboard.plugins.hparams import api as hp
from tensorflow import keras

from src.classes.TestingCallback import TestingCallback


def get_base_model(name, trainable=False, weights='imagenet'):
    pre_trained_models = {
        'Xception': tf.keras.applications.Xception,
        'VGG16': tf.keras.applications.vgg16.VGG16,
        'ResNet152V2': tf.keras.applications.resnet_v2.ResNet152V2,
        'EfficientNetB0': tf.keras.applications.efficientnet.EfficientNetB0,
        'EfficientNetB1': tf.keras.applications.efficientnet.EfficientNetB1,
        'EfficientNetB2': tf.keras.applications.efficientnet.EfficientNetB2,
        'EfficientNetB3': tf.keras.applications.efficientnet.EfficientNetB3,
        'EfficientNetB4': tf.keras.applications.efficientnet.EfficientNetB4,
        'EfficientNetB5': tf.keras.applications.efficientnet.EfficientNetB5,
        'EfficientNetB6': tf.keras.applications.efficientnet.EfficientNetB6,
        'EfficientNetB7': tf.keras.applications.efficientnet.EfficientNetB7,
    }

    pre_trained_model = pre_trained_models[name]

    pre_trained_base_model = pre_trained_model(
        weights=weights,  # Load weights pre-trained on ImageNet.
        input_shape=(150, 150, 3),
        include_top=False)  # Do not include the ImageNet classifier at the top.

    pre_trained_base_model.trainable = trainable

    return pre_trained_base_model


def save_results_to_file(results):
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    path = f'training_results/{dt_string}'
    os.mkdir(path)

    with open(f'{path}/results.json', 'w') as outfile:
        json.dump(results, outfile)


def get_files_by_labels(labels: list) -> dict:
    df = pd.read_csv('label_lookup.csv')
    filtered_rows = df[df['label'].isin(labels)]
    return filtered_rows


def get_config_file(file_name='../config.json'):
    f = open(file_name, )
    return json.load(f)


def save_model(model, model_name, path='src/models/saved/'):
    if path[-1] != '/':
        path = path + '/'
    model.save(path + model_name)


def load_model(model_name, path='models/saved/'):
    if path[-1] != '/':
        path = path + '/'
    return tf.keras.models.load_model(path + model_name + '/')


def save_weights(model, model_name, path='src/models/saved/'):
    if path[-1] != '/':
        path = path + '/'
    model.save_weights(path + model_name)


def load_weights(model, model_name, path='models/saved/'):
    if path[-1] != '/':
        path = path + '/'
    return model.load_weights(path + model_name, by_name=True)


def init_callbacks(test_generator, config, hyper_parameters):
    log_dir = f'logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)
    hyper_parameter_callback = hp.KerasCallback(log_dir, hyper_parameters)
    testing_callbacks = TestingCallback(test_generator, config, log_dir)

    return [tensorboard_callback, early_stopping_callback, hyper_parameter_callback, testing_callbacks]

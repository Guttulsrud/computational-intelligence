import random
from datetime import datetime

import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Dense, Dropout
from tensorboard.plugins.hparams import api as hp

from src.generator import get_data_generators
from src.models.utils import get_base_model


def get_files_by_labels(labels: list) -> dict:
    df = pd.read_csv('label_lookup.csv')
    filtered_rows = df[df['label'].isin(labels)]
    return filtered_rows


config = {
    'batch_size': 32,
    'number_of_runs': 100,
    'images_per_class': 2000,
    'number_of_epochs': 30,
    'validation_split': 0.2,
    'test_split': 0.2,
    'labels': [0, 1, 23],
    'image_shape': (150, 150, 3),
    'model': {
        # if config is changed to JSON, this has to be done somewhere else
        'architecture': hp.HParam('architecture', hp.Discrete(['Xception', 'VGG16', 'ResNet152V2', 'EfficientNetB7'])),
        'activation': hp.HParam('activation', hp.Discrete(['relu'])),
        'dropout': hp.HParam('dropout', hp.Discrete([0.1, 0.2])),
    },
    'metric': 'accuracy'
}

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[
            config['model']['architecture'],
            config['model']['activation'],
            config['model']['dropout'],
        ],
        metrics=[hp.Metric(config['metric'], display_name='Accuracy')],
    )

log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)

train_generator, validation_generator, test_generator = get_data_generators(config)

mixed_precision.set_global_policy('mixed_float16')


def get_random_hyper_parameters(config_model: dict) -> dict:
    hyper_parameters = {
        'architecture': random.choice(config_model['architecture'].domain.values),
        'activation': random.choice(config_model['activation'].domain.values),
        'dropout': random.choice(config_model['dropout'].domain.values),
    }
    print(hyper_parameters)

    return hyper_parameters


def create_model(parameters: dict):
    number_of_classes = len(config['labels'])

    base_model = get_base_model(name=parameters['architecture'])
    inputs = Input(shape=config['image_shape'])
    base_model = base_model(inputs, training=False)
    pooling_layer = keras.layers.GlobalAveragePooling2D()(base_model)
    dense_layer = Dense(number_of_classes * 8, activation=parameters['activation'])(pooling_layer)
    dropout_layer = Dropout(rate=parameters['dropout'])(dense_layer)
    output = Dense(number_of_classes, activation='softmax')(dropout_layer)
    model = keras.Model(inputs, output)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


for _ in range(config['number_of_runs']):
    hyper_parameters = get_random_hyper_parameters(config['model'])

    print('Running model ' + hyper_parameters['architecture'])
    model = create_model(hyper_parameters)

    hyper_parameter_callback = hp.KerasCallback(log_dir, hyper_parameters)

    model.fit(
        train_generator,
        batch_size=config['batch_size'],
        validation_data=validation_generator,
        epochs=config['number_of_epochs'],
        callbacks=[early_stopping_callback, tensorboard_callback, hyper_parameter_callback],
    )

import random
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import regularizers
from tensorboard.plugins.hparams import api as hp

from src.generator import get_data_generators
from src.models.keras import create_model
from models.utils import get_config_file, save_model

config = get_config_file()
# todo: put a mapping of me in config
config['model']['keras']['dense_regularization'] = regularizers.l2(0.01)
config['number_of_classes'] = len(config['labels'])

train_generator, validation_generator, test_generator = get_data_generators(config)


def get_random_hyper_parameters(config_model: dict) -> dict:
    hyper_parameters = {
        'architecture': random.choice(config_model['architecture'].domain.values),
        'activation': random.choice(config_model['activation'].domain.values),
        'dropout': random.choice(config_model['dropout'].domain.values),
    }
    print(hyper_parameters)

    return hyper_parameters


for _ in range(100):
    hyper_parameters = get_random_hyper_parameters(config['model'])

    print('Running model ' + hyper_parameters['architecture'])
    model = create_model(name=hyper_parameters['architecture'], config=config)

    log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)
    hyper_parameter_callback = hp.KerasCallback(log_dir, hyper_parameters)

    model.fit(
        train_generator,
        batch_size=config['batch_size'],
        validation_data=validation_generator,
        epochs=config['number_of_epochs'],
        callbacks=[early_stopping_callback, tensorboard_callback, hyper_parameter_callback],
    )

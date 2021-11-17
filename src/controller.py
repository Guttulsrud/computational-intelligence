from tensorboard.plugins.hparams import api as hp
import random
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

from src.TestingCallback import TestingCallback


class Controller:
    def __init__(self, config, test_gen):
        self.test_gen = test_gen
        self.config = config

        self.hyper_parameters = {key: hp.HParam(key, hp.Discrete(value)) for key, value in config['model'].items()}

    def get_random_hyper_parameters(self) -> dict:
        hyper_parameters = {key: random.choice(value.domain.values) for key, value in self.hyper_parameters.items()}
        return hyper_parameters


def init_callbacks(test_generator, config, hyper_parameters):
    log_dir = f'logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)
    hyper_parameter_callback = hp.KerasCallback(log_dir, hyper_parameters)
    testing_callbacks = TestingCallback(test_generator, config, log_dir)

    return [tensorboard_callback, early_stopping_callback, hyper_parameter_callback, testing_callbacks]

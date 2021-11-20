from tensorboard.plugins.hparams import api as hp
import random
import tensorflow as tf


class Controller:
    def __init__(self, config, test_gen):
        self.test_gen = test_gen
        self.config = config

        self.hyper_parameters = {key: hp.HParam(key, hp.Discrete(value)) for key, value in config['model'].items()}
        with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
            hp.hparams_config(
                hparams=[*self.hyper_parameters.values()],
                metrics=[hp.Metric('accuracy', display_name='Accuracy')],
            )

    def get_random_hyper_parameters(self) -> dict:
        hyper_parameters = {}
        for key, value in self.hyper_parameters.items():
            decided_value = random.choice(value.domain.values)

            hyper_parameters[key] = decided_value
        return hyper_parameters

    def update_hparams(self, run_dir, hyper_parameters, accuracy):
        with tf.summary.create_file_writer(f'logs/hparam_tuning/{run_dir}').as_default():
            hp.hparams(hyper_parameters)  # record the values used in this trial
            tf.summary.scalar('accuracy', accuracy, step=1)

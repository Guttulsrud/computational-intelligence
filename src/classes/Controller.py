from tensorboard.plugins.hparams import api as hp
import random


class Controller:
    def __init__(self, config, test_gen):
        self.test_gen = test_gen
        self.config = config

        self.hyper_parameters = {key: hp.HParam(key, hp.Discrete(value)) for key, value in config['model'].items()}

    def get_random_hyper_parameters(self) -> dict:
        hyper_parameters = {}
        for key, value in self.hyper_parameters.items():
            decided_value = random.choice(value.domain.values)

            hyper_parameters[key] = decided_value
        return hyper_parameters



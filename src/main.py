from generator import get_data_generators
from models.model import create_model
from models.utils import get_config_file
from src.controller import Controller, init_callbacks

config = get_config_file()
config['number_of_classes'] = len(config['classes'])

train_generator, validation_generator, test_generator = get_data_generators(config)

c = Controller(config, test_generator)

for _ in range(100):
    hyper_parameters = c.get_random_hyper_parameters()
    print(f'Running model {hyper_parameters}')
    model = create_model(name=hyper_parameters['convolutional_base'], config=config, hyper_parameters=hyper_parameters)

    callbacks = init_callbacks(test_generator, config, hyper_parameters)

    model.fit(
        train_generator,
        batch_size=config['batch_size'],
        validation_data=validation_generator,
        epochs=config['number_of_epochs'],
        callbacks=callbacks
    )

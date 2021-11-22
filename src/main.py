from generator import get_data_generators
from src.model import create_model
from src.utils import get_config_file
from src.classes.Controller import Controller
from src.utils import init_callbacks
import tensorflow as tf
from datetime import datetime

config = get_config_file()
config['number_of_classes'] = len(config['classes'])

train_generator, validation_generator, test_generator = get_data_generators(config)

controller = Controller(config, test_generator)

for count in range(100):
    try:
        hyper_parameters = controller.get_random_hyper_parameters()
        print(f'Running model number {count} \n'
              f'with hyper parameters: {hyper_parameters}')
        model = create_model(config=config, hyper_parameters=hyper_parameters)

        run_dir = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        callbacks = init_callbacks(test_generator, config, hyper_parameters, f'logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}')

        model.fit(
            train_generator,
            batch_size=config['batch_size'],
            validation_data=validation_generator,
            epochs=config['number_of_epochs'],
            callbacks=callbacks
        )
        _, accuracy = model.evaluate(test_generator)

        controller.update_hparams(run_dir, hyper_parameters, accuracy)
    except:
        pass

from tensorflow import keras
from datetime import datetime
from generator import get_data_generators
from models.utils import get_config_file, save_model
import tensorflow as tf
from src.models.keras import create_model
from tensorflow.python.keras import regularizers

log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

config = get_config_file()
# todo: put a mapping of me in config
config['model']['keras']['dense_regularization'] = regularizers.l2(0.01)

train_generator, validation_generator, test_generator = get_data_generators(config)

config['number_of_classes'] = len(config['labels'])

for model_name in config['model']['convolutional_base']:
    print(f'Running model {model_name}')
    model = create_model(name=model_name, config=config)

    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)

    model.fit(
        train_generator,
        batch_size=config['batch_size'],
        validation_data=validation_generator,
        epochs=30,
        callbacks=[early_stopping_callback, tensorboard_callback],
    )

    save_model(model, model_name)

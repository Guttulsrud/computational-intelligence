import pandas as pd
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
from datetime import datetime
from generator import get_data_generators
from models.utils import get_base_model
from tensorflow.keras import Input
import tensorflow as tf

from src.models.keras import create_model

config = {
    'batch_size': 32,
    'images_per_class': 2000,
    'validation_split': 0.2,
    'test_split': 0.2,
    'labels': [0, 1, 23],
    'image_shape': (150, 150, 3),
    'model': {
        'keras': {
            'optimizer': 'adam',
            'loss': 'categorical_crossentropy',
            'metrics': ['accuracy']
        },
        'convolutional_base': [
            'Xception',
            'VGG16',
            'ResNet152V2',
            'EfficientNetB7'
        ]
    }

}

log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

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

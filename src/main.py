import pandas as pd
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras import mixed_precision

from src.generator import get_data_generators
from src.models.utils import get_base_model, save_results_to_file
from tensorflow.keras import Input
import tensorflow as tf

def get_files_by_labels(labels: list) -> dict:
    df = pd.read_csv('label_lookup.csv')
    filtered_rows = df[df['label'].isin(labels)]
    return filtered_rows


config = {
    'batch_size': 32,
    'images_per_class': 2000,
    'validation_split': 0.2,
    'test_split': 0.2,
    'labels': [0, 1, 23],
    'image_shape': (150, 150, 3)
}

log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

train_generator, validation_generator, test_generator = get_data_generators(config)

models = [
    'Xception',
    'VGG16',
    'ResNet152V2',
    'EfficientNetB7'
]

results = {
    'info': {},
    'models': {},
}
number_of_classes = len(config['labels'])
trained_models = []

start_time = datetime.now()
mixed_precision.set_global_policy('mixed_float16')


def create_model(model_name):
    base_model = get_base_model(name=model_name)
    inputs = Input(shape=config['image_shape'])
    model = base_model(inputs, training=False)
    model = keras.layers.GlobalAveragePooling2D()(model)
    dense_layer = Dense(number_of_classes * 8, activation='relu')(model)
    dropout_lLayer = Dropout(rate=0.2)(dense_layer)
    output = Dense(number_of_classes, activation='softmax')(dropout_lLayer)
    model = keras.Model(inputs, output)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


for model_name in models:
    print(f'Running model {model_name}')
    model = create_model(model_name)

    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)

    model.fit(
        train_generator,
        batch_size=config['batch_size'],
        validation_data=validation_generator,
        epochs=30,
        callbacks=[early_stopping_callback, tensorboard_callback],
    )

    trained_models.append(model)



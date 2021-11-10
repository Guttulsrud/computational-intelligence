from src.models.utils import get_base_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
from tensorflow.keras import Input


def create_model(name, config):
    number_of_classes = config['number_of_classes']

    base_model = get_base_model(name=name)
    inputs = Input(shape=config['image_shape'])
    model = base_model(inputs, training=False)
    model = keras.layers.GlobalAveragePooling2D()(model)
    dense_layer = Dense(number_of_classes * 8, activation='relu')(model)
    dropout_layer = Dropout(rate=0.2)(dense_layer)
    output = Dense(number_of_classes, activation='softmax')(dropout_layer)
    model = keras.Model(inputs, output)

    model.compile(optimizer=config['model']['keras']['optimizer'],
                  loss=config['model']['keras']['loss'],
                  metrics=config['model']['keras']['metrics'])
    return model

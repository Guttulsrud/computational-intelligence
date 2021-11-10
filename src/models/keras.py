
from src.models.utils import get_base_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
from tensorflow.keras import Input


def create_model(name, config):
    number_of_classes = config['number_of_classes']
    image_shape = config['image_shape']
    config = config['model']['keras']

    dense_activation = config['dense_activation']
    dense_regularization = config['dense_regularization']
    dropout_rate = config['dropout_rate']
    dense_nodes = number_of_classes * 8

    base_model = get_base_model(name=name)
    inputs = Input(shape=image_shape)
    model = base_model(inputs, training=False)
    # Average pooling
    model = keras.layers.GlobalAveragePooling2D()(model)

    dense_layer = Dense(dense_nodes, kernel_regularizer=dense_regularization, activation=dense_activation)(model)
    # Regularization by dropout
    dropout_layer = Dropout(rate=dropout_rate)(dense_layer)
    output = Dense(number_of_classes, activation='softmax')(dropout_layer)
    model = keras.Model(inputs, output)

    model.compile(optimizer=config['optimizer'],
                  loss=config['loss'],
                  metrics=config['metrics'])
    return model


from src.models.utils import get_base_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.python.keras import regularizers


def create_model(name, config, hyper_parameters):
    number_of_classes = config['number_of_classes']
    image_shape = config['image_shape']

    dense_activation = hyper_parameters['dense_activation']
    dense_regularization = regularizers.l2(hyper_parameters['dense_regularization'])
    dropout_rate = hyper_parameters['dropout_rate']
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

    model.compile(optimizer=hyper_parameters['optimizer'],
                  loss=hyper_parameters['loss'],
                  metrics=hyper_parameters['metrics'])
    return model

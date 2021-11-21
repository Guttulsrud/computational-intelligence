from src.utils import get_base_model
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.python.keras import regularizers


def create_model(config, hyper_parameters):
    number_of_classes = config['number_of_classes']

    dense_activation = hyper_parameters['dense_activation']
    dense_regularization = regularizers.l1(hyper_parameters['dense_regularization'])
    dropout_rate = hyper_parameters['dropout_rate']
    dense_nodes = hyper_parameters['nodes_in_layer']
    pooling_type = hyper_parameters['pooling']

    pooling_map = {'average': keras.layers.GlobalAveragePooling2D(),
                   'max': keras.layers.GlobalMaxPooling2D()}

    model = tf.keras.Sequential()

    model.add(get_base_model(name=hyper_parameters['convolutional_base']))

    model.add(pooling_map[pooling_type])

    for _ in range(hyper_parameters['number_of_layers']):
        model.add(Dense(dense_nodes, kernel_regularizer=dense_regularization, activation=dense_activation))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(number_of_classes, activation='softmax'))

    model.compile(optimizer=hyper_parameters['optimizer'],
                  loss=hyper_parameters['loss_function'],
                  metrics=config['metric'])
    return model

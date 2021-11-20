
from src.models.utils import get_base_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.python.keras import regularizers


def create_model(config, hyper_parameters):
    number_of_classes = config['number_of_classes']
    image_shape = config['image_shape']

    dense_activation = hyper_parameters['dense_activation']
    dense_regularization = regularizers.l1(hyper_parameters['dense_regularization'])
    dropout_rate = hyper_parameters['dropout_rate']
    dense_nodes = hyper_parameters['nodes_in_layer']

    base_model = get_base_model(name=hyper_parameters['convolutional_base'])
    inputs = Input(shape=image_shape)
    model = base_model(inputs, training=False)

    pooling = keras.layers.GlobalAveragePooling2D()(model)

    dense_layer1 = Dense(dense_nodes, kernel_regularizer=dense_regularization, activation=dense_activation)(pooling)
    dropout_layer1 = Dropout(rate=dropout_rate)(dense_layer1)

    dense_layer2 = Dense(dense_nodes, kernel_regularizer=dense_regularization, activation=dense_activation)(dropout_layer1)
    dropout_layer2 = Dropout(rate=dropout_rate)(dense_layer2)

    output = Dense(number_of_classes, activation='softmax')(dropout_layer2)
    model = keras.Model(inputs, output)

    model.compile(optimizer=hyper_parameters['optimizer'],
                  loss=hyper_parameters['loss'],
                  metrics=hyper_parameters['metrics'])
    return model


from src.models.utils import get_base_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
from tensorflow.keras import Input
import numpy as np
#import tensorflow_model_optimization as tfmot


def create_model(name, config, pruning=False):
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

    dense_layer = Dense(dense_nodes, kernel_regularizer=dense_regularization,
                        activation=dense_activation)(model)
    # Regularization by dropout
    dropout_layer = Dropout(rate=dropout_rate)(dense_layer)
    output = Dense(number_of_classes, activation='softmax')(dropout_layer)
    model = keras.Model(inputs, output)

    """
    if pruning:
        prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
        num_images = config['images_per_class'] * len(config['labels'])
        end_step = np.ceil(num_images / config['batch_size']).astype(np.int32) * config['number_of_epochs']
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                                     final_sparsity=0.80,
                                                                     begin_step=0,
                                                                     end_step=end_step)
        }
    
        model = prune_low_magnitude(model, **pruning_params)
    """
    
    model.compile(optimizer=config['optimizer'],
                  loss=config['loss'],
                  metrics=config['metrics'])
    return model

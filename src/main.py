import random
from datetime import datetime
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import regularizers
from tensorboard.plugins.hparams import api as hp
import matplotlib.pyplot as plt
import numpy as np
import itertools
import io

from generator import get_data_generators
from models.keras import create_model
from models.utils import get_config_file, save_model

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# TODO:
# - Fairness Indicators
# - Displaying Confusion Matrix  in TensorBoard
# - Move Tensorboard stuff our of main
# - Make hparam bounderies in config

config = get_config_file()
config['model']['keras']['dense_regularization'] = regularizers.l2(0.01)
config['number_of_classes'] = len(config['classes'])
config['model']['architecture'] = hp.HParam('architecture',
                                            hp.Discrete(['Xception', 'VGG16', 'ResNet152V2', 'EfficientNetB7']))
config['model']['activation'] = hp.HParam('activation', hp.Discrete(['relu']))
config['model']['dropout'] = hp.HParam('dropout', hp.Discrete([0.1, 0.2]))

train_generator, validation_generator, test_generator = get_data_generators(config)


def get_random_hyper_parameters(config_model: dict) -> dict:
    hyper_parameters = {
        'architecture': random.choice(config_model['architecture'].domain.values),
        'activation': random.choice(config_model['activation'].domain.values),
        'dropout': random.choice(config_model['dropout'].domain.values),
    }
    print(hyper_parameters)

    return hyper_parameters


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


label_list = []
for data in test_generator:
    for label in data[1]:
        label_list.append(list(label))


def log_confusion_matrix(epoch, log):
    global log_dir

    test_pred_raw = model.predict(test_generator)
    test_pred = np.argmax(test_pred_raw, axis=1)
    test_labels = np.argmax(label_list, axis=1)

    # Calculate the confusion matrix.
    cm = confusion_matrix(test_labels, test_pred)
    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=config["classes"])
    cm_image = plot_to_image(figure)

    # Log the confusion matrix as an image summary.
    file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)


for _ in range(100):
    hyper_parameters = get_random_hyper_parameters(config['model'])

    print('Running model ' + hyper_parameters['architecture'])
    model = create_model(name=hyper_parameters['architecture'], config=config)

    log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    file_writer = tf.summary.create_file_writer(log_dir)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)
    hyper_parameter_callback = hp.KerasCallback(log_dir, hyper_parameters)
    cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

    model.fit(
        train_generator,
        batch_size=config['batch_size'],
        validation_data=validation_generator,
        epochs=config['number_of_epochs'],
        callbacks=[
            early_stopping_callback,
            tensorboard_callback,
            hyper_parameter_callback,
            cm_callback
        ],
    )

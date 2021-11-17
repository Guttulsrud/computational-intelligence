import io
import itertools
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras


class ConfusionMatrixCallback(keras.callbacks.Callback):
    def __init__(self, test_gen, config, log_dir):
        self.test_gen = test_gen
        self.config = config
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs):

        label_list = []
        for data in self.test_gen:
            for label in data[1]:
                label_list.append(list(label))

        test_pred_raw = self.model.predict(self.test_gen)
        test_pred = np.argmax(test_pred_raw, axis=1)
        test_labels = np.argmax(label_list, axis=1)

        cm = confusion_matrix(test_labels, test_pred)

        figure = plot_confusion_matrix(cm, class_names=self.config["classes"])
        cm_image = plot_to_image(figure)

        file_writer_cm = tf.summary.create_file_writer(self.log_dir + '/cm')
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)


def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib as mpl
import io
import itertools
from matplotlib import pyplot as plt


class TestingCallback(keras.callbacks.Callback):
    def __init__(self, test_gen, config, log_dir):
        self.test_gen = test_gen
        self.config = config
        self.log_dir = log_dir

    def on_epoch_end(self, epoch, logs):
        test_pred_raw = self.model.predict(self.test_gen)
        test_pred = np.argmax(test_pred_raw, axis=1)

        test_labels_raw = [list(label) for data in self.test_gen for label in data[1]]
        test_labels = np.argmax(test_labels_raw, axis=1)

        self.save_confusion_matrix(epoch, test_labels, test_pred)
        self.save_metrics(epoch, test_labels, test_pred)

    def save_metrics(self, epoch, test_labels, test_pred):
        data = classification_report(test_labels, test_pred, output_dict=True)
        del data['accuracy']

        mpl.style.use('seaborn')

        df = pd.DataFrame(data).transpose().reset_index()
        for metric in ['precision', 'recall', 'f1-score']:
            fig = plt.figure()
            plt.bar(df['index'], df[metric])

            plt.ylim(0, 1)
            plt.xticks(rotation=30)
            plt.ylabel(metric)
            plt.xlabel('class')
            plt.title(metric)

            metrics_image = self.plot_to_image(fig)

            file_writer = tf.summary.create_file_writer(f'{self.log_dir}/{metric}')
            with file_writer.as_default():
                tf.summary.image(metric, metrics_image, step=epoch)

    def save_confusion_matrix(self, epoch, test_labels, test_pred):
        cm = confusion_matrix(test_labels, test_pred)
        figure = self.plot_confusion_matrix(cm, class_names=self.config["classes"])
        cm_image = self.plot_to_image(figure)

        file_writer_cm = tf.summary.create_file_writer(f'{self.log_dir}/confusion_matrix')
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)

    @staticmethod
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

    @staticmethod
    def plot_to_image(figure):
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)

        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image

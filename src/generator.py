import pandas as pd
import numpy as np
import cv2
import tensorflow as tf

lookup_table = 'label_lookup.csv'
data_path = '../images/'


def get_data_generators(config: dict):
    df = get_data_from_labels(config['labels'])
    df['file_name'] = data_path + df['file_name']

    validation_data = df.sample(frac=config['validation_split'])
    temp = df.drop(validation_data.index)
    test_data = temp.sample(frac=config['test_split'])
    train_data = temp.drop(test_data.index)

    train_data_generator = Generator(train_data, config)
    validation_data_generator = Generator(validation_data, config)
    test_data_generator = Generator(test_data, config)

    return train_data_generator, validation_data_generator, test_data_generator


def get_data_from_labels(labels: list) -> pd.DataFrame:
    df = pd.read_csv(lookup_table)
    df = df[df['label'].isin(labels)]
    # Add extraction of self.config['images_per_class']
    return df


class Generator(tf.keras.utils.Sequence):
    def __init__(self, df, config=None):
        self.config = config
        self.df = df
        self.data_length = len(self.df)
        self.batch_size = config['batch_size']

    def on_epoch_start(self):
        # def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __get_input(self, path):
        image = tf.keras.preprocessing.image.load_img(path)
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = self.__augment_image(image)

        return image / 255.

    def __get_output(self, label):
        one_hot_label = np.zeros(len(self.config['labels']))
        one_hot_label[self.config['labels'].index(label)] = 1
        return one_hot_label

    def __get_data(self, batches):
        path_batch = batches['file_name']
        label_batch = batches['label']

        x_batch = np.asarray([self.__get_input(path) for path in path_batch])
        y_batch = np.asarray([self.__get_output(label) for label in label_batch])

        return x_batch, y_batch

    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        x, y = self.__get_data(batches)
        return x, y

    def __augment_image(self, image):
        image = cv2.resize(image, (150, 150), interpolation=cv2.INTER_AREA)
        return image

    def __len__(self):
        return self.data_length // self.batch_size

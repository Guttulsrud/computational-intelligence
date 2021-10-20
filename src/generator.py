import pandas as pd
import numpy as np
import cv2


class Generator:
    def __init__(self, config=None):
        self.data_path = '../images/'
        self.lookup_table = '../label_lookup.csv'
        self.config = config

    def get_data(self):
        df = self.__get_data_from_labels(self.config['labels'])
        df['file_name'] = self.data_path + df['file_name']

        image_shape = self.config['image_shape']
        batch_size = self.config['batch_size']

        batch_of_images = np.zeros((batch_size, image_shape[0], image_shape[1], image_shape[2]))
        batch_of_labels = np.zeros((batch_size, len(self.config['labels'])))

        while True:

            df = df.sample(frac=1)

            for count, row in enumerate(df.iterrows()):
                row = row[1]
                file_path = row['file_name']
                label = row['label']

                image = cv2.imread(file_path)
                image = self.__augment_image(image)

                one_hot_label = np.zeros(len(self.config['labels']))
                one_hot_label[self.config['labels'].index(label)] = 1

                batch_of_images[count % batch_size] = image
                batch_of_labels[count % batch_size] = one_hot_label

                if count % batch_size == batch_size - 1:
                    yield batch_of_images, batch_of_labels

    def __get_data_from_labels(self, labels: list) -> dict:
        df = pd.read_csv(self.lookup_table)
        df = df[df['label'].isin(labels)]
        # Add extraction of self.config['images_per_class']
        return df

    def __augment_image(self, image):
        image = cv2.resize(image, (150, 150))
        return image


def view_image(img_name):
    img = cv2.imread(f'../images/{img_name}')
    img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    config = {
        'batch_size': 32,
        'images_per_class': 2000,
        'validation_split': 0.8,
        'labels': [0, 1, 23],
        'image_shape': (150, 150, 3)
    }
    g = Generator(config)
    generator = g.get_data()
    next(generator)

import pandas as pd
import numpy as np
import cv2


class Generator:
    def __init__(self, config=None):
        self.data_path = '../images/'
        self.lookup_table = 'label_lookup.csv'
        self.config = config

    def get_data(self, labels: list):
        df = self._get_data_from_labels(labels)
        df['file_name'] = self.data_path + df['file_name']

        image_shape = (32, 32, 3)
        batch_size = self.config['batch_size']

        batch_of_images = np.zeros((batch_size, image_shape[0], image_shape[1], image_shape[2]))
        batch_of_labels = np.zeros((batch_size, 1))

        while True:

            df = df.sample(frac=1)

            for count, row in enumerate(df.iterrows()):
                row = row[1]
                file_path = row['file_name']
                label = row['label']

                image = cv2.imread(file_path)
                image = self._augment_image(image)

                batch_of_images[count % batch_size] = image
                batch_of_labels[count % batch_size] = label

                if count % batch_size == batch_size - 1:
                    yield batch_of_images, batch_of_labels

    def _get_data_from_labels(self, labels: list) -> dict:
        df = pd.read_csv(self.lookup_table)
        df = df[df['label'].isin(labels)]
        return df

    def _augment_image(self, image):
        # Here we will add augmentation to images
        return image


def view_image(img_name):
    img = cv2.imread(f'../images/{img_name}')
    img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    gen_config = {
        'batch_size': 32
    }
    g = Generator(gen_config)
    generator = g.get_data([0, 1, 23])

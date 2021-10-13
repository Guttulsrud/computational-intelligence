import tensorflow as tf
import pandas as pd
import numpy as np
import cv2

def view_image(img_name):
    img = cv2.imread(f'../images/{img_name}')
    img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)
    cv2.imshow('', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_files_by_labels(labels: list) -> dict:
    df = pd.read_csv('label_lookup.csv')
    filtered_rows = df[df['label'].isin(labels)]
    return filtered_rows


df = get_files_by_labels([25, 37])

file_names = df['file_name'].to_list()[:10]
labels = df['label'].to_list()[0:10]

map = {
    37: 0,
    25: 1
}

labels = [map[x] for x in labels]
file_names = [f'../ee/{file_name}' for file_name in file_names]
files = np.array([cv2.imread(f'../images/{file_name}') for file_name in file_names])

y = np.array(labels)
x = files

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
    tf.keras.layers.Dense(2)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x, y, epochs=10)

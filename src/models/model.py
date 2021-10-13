import tensorflow as tf
import pandas as pd
import numpy as np
from os import listdir
import cv2
from pandas import DataFrame
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

from src.models.utils import get_base_model


def get_files_by_labels(labels: list) -> dict:
    df = pd.read_csv('../label_lookup.csv')
    filtered_rows = df[df['label'].isin(labels)]
    return filtered_rows


df = get_files_by_labels([25, 37])

file_names = df['file_name'].to_list()[:30]
labels = df['label'].to_list()[0:30]

map = {
    37: 0,
    25: 1,
}

labels = [map[x] for x in labels]

file_names = [f'../images/{file_name}' for file_name in file_names]

files = []

for file_name in file_names:
    img = cv2.imread(f'../../images/{file_name}')
    img = cv2.resize(img, (150, 150))

    files.append(img)

x = np.array(files)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

models = [
    'Xception',
    'VGG16',
    'ResNet152V2',
    'EfficientNetB0',
    'EfficientNetB1',
    'EfficientNetB2',
    'EfficientNetB3',
    'EfficientNetB4',
    'EfficientNetB5',
    'EfficientNetB6',
    'EfficientNetB7'

]

scores = {}
for pre_trained_model in models:
    print(f'Running model {pre_trained_model}')
    base_model = get_base_model(name=pre_trained_model)

    inputs = tf.keras.Input(shape=(150, 150, 3))
    # We make sure that the base_model is running in inference mode here,
    # by passing `training=False`. This is important for fine-tuning, as you will
    # learn in a few paragraphs.

    x = base_model(inputs, training=False)
    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # A Dense classifier with a single unit (binary classification)
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    # print(len(X_train), len(X_test))

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])

    history = model.fit(X_train, y_train, epochs=20, callbacks=[], validation_data=(X_test, y_test))

    score = f'{str(max(history.history["val_binary_accuracy"]) * 100)}%'
    print(score)
    scores[pre_trained_model] = score

print(scores)

import pandas as pd
import numpy as np
import cv2
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import matplotlib.pyplot as plt
from datetime import datetime

from src.models.utils import get_base_model, save_results_to_file
from tensorflow.keras import Input


def get_files_by_labels(labels: list) -> dict:
    df = pd.read_csv('../label_lookup.csv')
    filtered_rows = df[df['label'].isin(labels)]
    return filtered_rows


number_of_images = 2000
classes = [x for x in range(0, 10)]

df = get_files_by_labels(classes)

file_names = df['file_name'].to_list()[:number_of_images]
labels = df['label'].to_list()[:number_of_images]

file_names = [f'../images/{file_name}' for file_name in file_names]

files = []

for file_name in file_names:
    img = cv2.imread(f'../../images/{file_name}')
    img = cv2.resize(img, (150, 150))

    files.append(img)

x = np.array(files)
y = np.array(labels)

encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
dummy_y = np_utils.to_categorical(encoded_Y)

X_train, X_validation, y_train, y_validation = train_test_split(x, dummy_y, test_size=0.35)

models = [
    'Xception',
    'VGG16',
    'ResNet152V2',
    'EfficientNetB7'
]

results = {
    'info': {},
    'models': {},
}
number_of_classes = len(classes)
trained_models = []

start_time = datetime.now()

for pre_trained_model in models:

    print(f'Running model {pre_trained_model}')
    base_model = get_base_model(name=pre_trained_model)

    inputs = Input(shape=(150, 150, 3))
    model = base_model(inputs, training=False)
    model = keras.layers.GlobalAveragePooling2D()(model)

    dense_layer = Dense(number_of_classes * 8, activation='relu')(model)
    dropout_lLayer = Dropout(rate=0.2)(dense_layer)
    output = Dense(number_of_classes, activation='softmax')(dropout_lLayer)

    model = keras.Model(inputs, output)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)

    history = model.fit(
        X_train,
        y_train,
        epochs=30,
        callbacks=[early_stopping_callback],
        validation_data=(X_validation, y_validation)
    )

    trained_models.append(pre_trained_model)
    plt.plot(history.history["val_accuracy"])

    results['models'][pre_trained_model] = {
        'val_accuracy': max(history.history["val_accuracy"]) * 100,
    }


results['info'] = {
    'number_of_images': number_of_images,
    'number_of_classes': number_of_classes,
    'start_time': start_time.strftime("%Y-%m-%d-%H-%M-%S"),
    'end_time': datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
}
save_results_to_file(results)

plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(trained_models, loc='upper left')
plt.show()

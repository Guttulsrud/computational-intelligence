import pandas as pd
from keras.layers import Dense, Dropout
from tensorflow import keras
import matplotlib.pyplot as plt
from datetime import datetime

from src.generator import Generator
from src.models.utils import get_base_model, save_results_to_file
from tensorflow.keras import Input


def get_files_by_labels(labels: list) -> dict:
    df = pd.read_csv('label_lookup.csv')
    filtered_rows = df[df['label'].isin(labels)]
    return filtered_rows


config = {
    'batch_size': 32,
    'images_per_class': 2000,
    'validation_split': 0.8,
    'labels': [0, 1, 23],
    'image_shape': (150, 150, 3)
}
g = Generator(config)
generator = g.get_data()

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
number_of_classes = len(config['labels'])
trained_models = []

start_time = datetime.now()

for pre_trained_model in models:
    print(f'Running model {pre_trained_model}')
    base_model = get_base_model(name=pre_trained_model)

    inputs = Input(shape=config['image_shape'])
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
        generator,
        epochs=30,
        callbacks=[early_stopping_callback],
    )

    trained_models.append(pre_trained_model)
    plt.plot(history.history["val_accuracy"])

    results['models'][pre_trained_model] = {
        'val_accuracy': max(history.history["val_accuracy"]) * 100,
    }

results['info'] = {
    'number_of_images': config['images_per_class'],
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

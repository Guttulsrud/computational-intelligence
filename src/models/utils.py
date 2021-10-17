import tensorflow as tf
from datetime import datetime
import json
import os


def get_base_model(name, trainable=False, weights='imagenet'):
    pre_trained_models = {
        'Xception': tf.keras.applications.Xception,
        'VGG16': tf.keras.applications.vgg16.VGG16,
        'ResNet152V2': tf.keras.applications.resnet_v2.ResNet152V2,
        'EfficientNetB0': tf.keras.applications.efficientnet.EfficientNetB0,
        'EfficientNetB1': tf.keras.applications.efficientnet.EfficientNetB1,
        'EfficientNetB2': tf.keras.applications.efficientnet.EfficientNetB2,
        'EfficientNetB3': tf.keras.applications.efficientnet.EfficientNetB3,
        'EfficientNetB4': tf.keras.applications.efficientnet.EfficientNetB4,
        'EfficientNetB5': tf.keras.applications.efficientnet.EfficientNetB5,
        'EfficientNetB6': tf.keras.applications.efficientnet.EfficientNetB6,
        'EfficientNetB7': tf.keras.applications.efficientnet.EfficientNetB7,
    }

    pre_trained_model = pre_trained_models[name]

    pre_trained_base_model = pre_trained_model(
        weights=weights,  # Load weights pre-trained on ImageNet.
        input_shape=(150, 150, 3),
        include_top=False)  # Do not include the ImageNet classifier at the top.

    pre_trained_base_model.trainable = trainable

    return pre_trained_base_model


def save_results_to_file(results):
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    path = f'training_results/{dt_string}'
    os.mkdir(path)

    with open(f'{path}/results.json', 'w') as outfile:
        json.dump(results, outfile)

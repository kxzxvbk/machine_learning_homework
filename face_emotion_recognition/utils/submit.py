import os
import pandas as pd
from PIL import Image
import tensorflow as tf
import numpy as np


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image /= 255  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


def submit(model, class_dict):
    test_set = pd.read_csv('./data/submission.csv', sep=',')
    test_path = './data/test'
    total = len(test_set)
    y = []
    for i in range(total):
        image_path = os.path.join(test_path, test_set['file_name'][i])
        y.append(image_path)

    ds = tf.data.Dataset.from_tensor_slices(y)
    image_ds = ds.map(load_and_preprocess_image)
    ds = image_ds.batch(32)

    preds = model.predict(ds)
    answer = preds.argmax(axis=1)
    sub = []
    for i in range(answer.shape[0]):
        sub.append(class_dict[answer[i]])
    test_set['class'] = sub
    test_set[['file_name', 'class']].to_csv('submit.csv', sep=',', index=False)

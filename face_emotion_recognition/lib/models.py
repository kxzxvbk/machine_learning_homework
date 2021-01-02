import tensorflow as tf
import os
from utils.submit import submit


class TopModel:
    def __init__(self, name, class_num, train_set, test_set, img_size):
        self.img_size = img_size
        self.train_set = train_set
        self.test_set = test_set
        self.name = name
        self.class_dict = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
        if name == "resnet101":
            self.model = create_model_resnet101(class_num, self.img_size)
        elif name == 'resnet50':
            self.model = create_model_resnet50(class_num, self.img_size)

    def train(self):
        if self.name == "resnet101":
            train_resnet101(self.model, self.train_set, self.test_set)
        if self.name == "resnet50":
            train_resnet50(self.model, self.train_set, self.test_set)

    def submit(self):
        if self.name == "resnet101":
            if os.path.exists('./.model/save_model_resnet101.h5'):
                print("Load existing weights from:  ./.model/save_model_resnet101.h5")
                self.model.load_weights('./.model/save_model_resnet101.h5')
        if self.name == "resnet50":
            if os.path.exists('./.model/save_model_resnet50.h5'):
                print("Load existing weights from:  ./.model/save_model_resnet50.h5")
                self.model.load_weights('./.model/save_model_resnet50.h5')
        submit(self.model, self.class_dict)


def create_model_resnet101(class_num, img_size):
    res_net = tf.keras.applications.ResNet101V2(input_shape=(img_size, img_size, 3), include_top=False)
    res_net.trainable = True
    image = tf.keras.layers.Input((img_size, img_size, 3), dtype=tf.float32)
    features = res_net(image)
    o = tf.keras.layers.GlobalAveragePooling2D()(features)
    o1 = tf.keras.layers.Dense(200, activation='relu')(o)
    o1 = tf.keras.layers.Dropout(0.1)(o1)
    o2 = tf.keras.layers.Dense(class_num, activation='softmax')(o1)

    model = tf.keras.models.Model(inputs=image, outputs=o2)
    return model


def train_resnet101(model, train_set, val_set):
    lr = 0.001
    for i in range(3):
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                      loss='sparse_categorical_crossentropy',
                      metrics=["accuracy"])
        if os.path.exists('./.model/save_model_resnet101.h5'):
            print("Load existing weights from:  ./.model/save_model_resnet101.h5")
            model.load_weights('./.model/save_model_resnet101.h5')
        model.fit(train_set, epochs=20, steps_per_epoch=200)
        model.evaluate(val_set)
        model.save('./.model/save_model_resnet101.h5')
        lr /= 10


def create_model_resnet50(class_num, img_size):

    res_net = tf.keras.applications.ResNet50V2(input_shape=(img_size, img_size, 3), include_top=False)
    res_net.trainable = True
    image = tf.keras.layers.Input((img_size, img_size, 3), dtype=tf.float32)
    features = res_net(image)
    o = tf.keras.layers.GlobalAveragePooling2D()(features)
    o1 = tf.keras.layers.Dense(200, activation='relu')(o)
    o1 = tf.keras.layers.Dropout(0.1)(o1)
    o2 = tf.keras.layers.Dense(class_num, activation='softmax')(o1)

    model = tf.keras.models.Model(inputs=image, outputs=o2)
    return model


def train_resnet50(model, train_set, val_set):
    lr = 0.001
    for i in range(3):
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                      loss='sparse_categorical_crossentropy',
                      metrics=["accuracy"])
        if os.path.exists('./.model/save_model_resnet50.h5'):
            print("Load existing weights from:  ./.model/save_model_resnet50.h5")
            model.load_weights('./.model/save_model_resnet50.h5')
        model.fit(train_set, epochs=10, steps_per_epoch=200)
        model.evaluate(val_set)
        model.save('./.model/save_model_resnet50.h5')
        lr /= 10

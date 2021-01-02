import pathlib
import random
import tensorflow as tf
from lib.models import TopModel


img_size = 224
data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_size, img_size])
    return image


def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

data_root = pathlib.Path('data/train')
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
image_count = len(all_image_paths)

train_image_paths = all_image_paths[:20000]
test_image_paths = all_image_paths[20000:]
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
label_to_index = dict((name, index) for index, name in enumerate(label_names))
all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]
train_image_labels = all_image_labels[:20000]
test_image_labels = all_image_labels[20000:]


train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_image_labels))
train_image_label_ds = train_ds.map(load_and_preprocess_from_path_label)
print(train_image_label_ds)

test_ds = tf.data.Dataset.from_tensor_slices((test_image_paths, test_image_labels))
test_image_label_ds = test_ds.map(load_and_preprocess_from_path_label)
print(test_image_label_ds)

BATCH_SIZE = 32

train_ds = train_image_label_ds.shuffle(buffer_size=20000)
train_ds = train_ds.repeat()
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(buffer_size=128)
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
print(train_ds)

test_ds = test_image_label_ds.shuffle(buffer_size=image_count-20000)
# test_ds = test_ds.repeat()
test_ds = test_ds.batch(BATCH_SIZE)
# test_ds = test_ds.prefetch(buffer_size=128)
test_ds = test_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
print(test_ds)

model = TopModel('resnet101', len(label_names), train_ds, test_ds, img_size)
model.train()
model.submit()

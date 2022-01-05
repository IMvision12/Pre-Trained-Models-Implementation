from VGG16 import VGG16
import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def normalization(train_images, test_images):
    mean = np.mean(train_images, axis=(0, 1, 2, 3))
    std = np.std(train_images, axis=(0, 1, 2, 3))
    train_images = (train_images - mean) / (std + 1e-7)
    test_images = (test_images - mean) / (std + 1e-7)
    return train_images, test_images

def load_images():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    train_images = train_images.astype(np.float32)
    test_images = test_images.astype(np.float32)

    (train_images, test_images) = normalization(train_images, test_images)

    train_labels = tf.keras.utils.to_categorical(train_labels, 10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    return train_images, train_labels, test_images, test_labels

X_train, y_train, X_test, y_test = load_images()

datagen = ImageDataGenerator(rotation_range=15,
                             width_shift_range=0.1,
                            height_shift_range=0.1,
                            horizontal_flip=True,
                            vertical_flip=True)

datagen.fit(X_train)
model = VGG16(input_shape=(32,32,3),num_classes=10)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1,decay=1e-6, momentum=0.9, nesterov=True)

learning_rate = 0.1
momentum = 0.9
lr_drop = 20

def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))

reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.fit_generator(datagen.flow(X_train, y_train,batch_size=16), epochs=10, callbacks=[reduce_lr],
                    steps_per_epoch=X_train.shape[0] // 16, validation_data=(X_test, y_test))


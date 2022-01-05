import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ResNet50 import ResNet50

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping


(X_train, y_train), (X_test, y_test) = cifar10.load_data()
Y_train = tf.keras.utils.to_categorical(y_train, 10)
Y_test = tf.keras.utils.to_categorical(y_test, 10)

X_train = tf.cast(X_train,tf.float32)
X_test = tf.cast(X_test,tf.float32)

mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
X_train /= 128.
X_test /= 128.

data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.1,0.2],
    shear_range=0.2,
    zoom_range=0.2)

data_gen.fit(X_train)

model = ResNet50(input_shape=(32,32,3),classes=10)
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(data_gen.flow(X_train, Y_train, batch_size=16),
          validation_data=(X_test,Y_test), epochs=10, steps_per_epoch=200)

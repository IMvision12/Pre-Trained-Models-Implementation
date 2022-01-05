import tensorflow as tf
from tensorflow import keras
import os

def conv_block(filters,inputs,flag=False):

    if not flag:
        x = tf.keras.layers.Conv2D(filters, kernel_size=(3,3), strides=1,
                               activation='relu', padding ='same')(inputs)
        x = tf.keras.layers.Conv2D(filters, kernel_size=(3,3), strides=1,
                               activation='relu', padding ='same')(x)
        x = tf.keras.layers.MaxPooling2D((5, 5), strides=2, padding='same')(x)

        return x

    if flag:
        x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=1, activation='relu',
                                   padding='same')(inputs)
        x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=1, activation='relu',
                                   padding='same')(x)
        x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=1, activation='relu',
                                   padding='same')(x)
        x = tf.keras.layers.MaxPooling2D((5,5), strides=2,padding ='same')(x)

        return x


def VGG16(input_shape,num_classes):

    input_layer = tf.keras.layers.Input(input_shape)
    x = conv_block(64,input_layer,False)
    x = conv_block(128, x, False)
    x = conv_block(256, x, True)
    x = conv_block(512, x, True)
    x = conv_block(512, x, True)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096,activation='relu')(x)
    x = tf.keras.layers.Dense(4096,activation='relu')(x)
    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    vgg_model = tf.keras.models.Model(inputs=input_layer,outputs=output_layer)
    return vgg_model


#model = VGG16((224,224,3),1000)
#model.summary()
#tf.keras.utils.plot_model(model,to_file="vgg16.png",
                          #show_shapes=True,show_dtype=True,
                          #show_layer_names=True)
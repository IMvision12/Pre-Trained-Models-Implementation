import tensorflow as tf
from tensorflow import keras

def conv_block(inputs, filters, strides=1):
    x = tf.keras.layers.Conv2D(filters, kernel_size=(3,3), strides=2)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def separable_conv_block(x, filters, activation=None):
    if activation:
        x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.SeparableConv2D(filters, (3,3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if not activation:
        x = tf.keras.layers.Activation('relu')(x)
    return x

def middle_block(inputs,filters):
    x_in = inputs

    x = separable_conv_block(inputs, filters, activation=True)
    x = separable_conv_block(x, filters, activation=True)
    x = separable_conv_block(x, filters, activation=True)

    output = tf.keras.layers.add([x, x_in])
    return output

def entry_block(inputs,filters,activation=True,first=False):
    conv1_1 = tf.keras.layers.Conv2D(filters,kernel_size=(1,1),strides=2,padding='same')(inputs)
    conv1_1 = tf.keras.layers.BatchNormalization()(conv1_1)

    x = separable_conv_block(inputs,filters,activation=activation)
    x = separable_conv_block(x, filters,activation=True)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=2,padding='same')(x)

    add_layer = tf.keras.layers.Add()([x,conv1_1])
    return add_layer

def Xception(num_classes,input_shape=(299,299,3)):
    input_layer = tf.keras.Input(input_shape)

    x = conv_block(input_layer,32,strides=2)
    x = conv_block(x,64)
    x = entry_block(x,128,activation=False)
    x = entry_block(x,256)
    x = entry_block(x,728)

    for i in range(8):
        x = middle_block(x,728)

    x_new = tf.keras.layers.Conv2D(1024,kernel_size=(1,1),strides=2,padding='same')(x)
    x_new = tf.keras.layers.BatchNormalization()(x_new)

    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.SeparableConv2D(728,kernel_size=(3,3),padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.SeparableConv2D(1024,kernel_size=(3,3),padding='same')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
    add_layer = tf.keras.layers.Add()([x, x_new])

    add_layer = separable_conv_block(add_layer,1536,activation=False)
    add_layer = separable_conv_block(add_layer,2048,activation=False)
    add_layer = tf.keras.layers.GlobalAveragePooling2D()(add_layer)

    output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(add_layer)
    model = tf.keras.models.Model(inputs=input_layer,outputs=output_layer)
    return model

model = Xception(1000)
model.summary()

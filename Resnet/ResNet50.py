import tensorflow as tf

def identity_block(x, filters, strides=1):
    x_in = x

    x = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='valid')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=strides, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters*4, kernel_size=(1, 1), strides=strides, padding='valid')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)

    x = tf.keras.layers.Add()([x, x_in])
    x = tf.keras.layers.Activation('relu')(x)

    return x


def convolutional_block(x, filters,strides=1):
    x_in = x

    x = tf.keras.layers.Conv2D(filters, (1, 1), strides=2)(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=strides, padding='same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters*4, kernel_size=(1, 1), strides=strides, padding='valid')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)

    x_in = tf.keras.layers.Conv2D(filters*4, kernel_size=(1, 1), strides=2, padding='valid')(x_in)
    x_in = tf.keras.layers.BatchNormalization(axis=3)(x_in)

    x = tf.keras.layers.Add()([x, x_in])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def ResNet50(input_shape=(64, 64, 3), classes=3):
    input_layer = tf.keras.Input(input_shape)
    x = tf.keras.layers.ZeroPadding2D((3, 3))(input_layer)

    x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = convolutional_block(x, 64)
    x = identity_block(x,64)
    x = identity_block(x,64)

    x = convolutional_block(x, 128)
    for i in range(3):
        x = identity_block(x, 128)

    x = convolutional_block(x, 256)
    for i in range(5):
        x = identity_block(x, 256)

    x = convolutional_block(x,512)
    x = identity_block(x, 512)
    x = identity_block(x,512)


    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output_layer = tf.keras.layers.Dense(classes, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name='ResNet50')

    return model

#resnet_model = ResNet50(input_shape = (224, 224, 3), classes = 1000)
#resnet_model.summary()



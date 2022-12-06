import tensorflow as tf
from tensorflow.keras import layers


class StochasticDepth(layers.Layer):
    def __init__(self, drop_path, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_path = drop_path

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_path
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


class Block(tf.keras.Model):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super(Block, self).__init__()
        self.dim = dim
        if layer_scale_init_value > 0:
            self.gamma = tf.Variable(layer_scale_init_value * tf.ones((dim,)))
        else:
            self.gamma = None
        self.dconv = layers.Conv2D(dim, (7,7), padding='same', groups = dim)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.pwconv = layers.Dense(dim*4)
        self.act =  layers.Activation("gelu")
        self.pwconv2 = layers.Dense(dim)
        self.drop_path = (
            StochasticDepth(drop_path)
            if drop_path > 0.0
            else layers.Activation("linear")
        )

    def call(self, inputs):
        x = inputs

        x = self.dw_conv_1(x)
        x = self.layer_norm(x)
        x = self.pw_conv_1(x)
        x = self.act_fn(x)
        x = self.pw_conv_2(x)

        if self.gamma is not None:
            x = self.gamma * x

        return inputs + self.drop_path(x)



def DownSample(dims):
    downsample_layers = []
    stem = keras.Sequential(
        [
            layers.Conv2D(dims[0], kernel_size=4, strides=4),
            layers.LayerNormalization(epsilon=1e-6),
        ],
        name="stem",
    )
    downsample_layers.append(stem)
    for i in range(3):
        downsample_layer = keras.Sequential(
            [
                layers.LayerNormalization(epsilon=1e-6),
                layers.Conv2D(dims[i + 1], kernel_size=2, strides=2),
            ],
            name=f"downsampling_block_{i}",
        )
        downsample_layers.append(downsample_layer)

    return downsample_layers
import tensorflow as tf

#
# ResNet 20, 36, 44, 56, 110, 1202 for CIFAR10
#
# He, Kaiming, et al. "Deep residual learning for image recognition." (2016)
#  - https://arxiv.org/abs/1512.03385
#


def resnet_basic_block(inputs, num_filters, strides=1, name=None):
    x = tf.keras.layers.Conv2D(
        num_filters,
        kernel_size=3,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name=name + "_1_conv2d",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(
        epsilon=1e-5, momentum=0.1, name=name + "_1_bn"
    )(x)
    x = tf.keras.layers.Activation("relu", name=name + "_1_relu")(x)

    x = tf.keras.layers.Conv2D(
        num_filters,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name=name + "_2_conv2d",
    )(x)
    x = tf.keras.layers.BatchNormalization(
        epsilon=1e-5, momentum=0.1, name=name + "_2_bn"
    )(x)

    if strides != 1:
        y = tf.keras.layers.Conv2D(
            num_filters,
            kernel_size=1,
            strides=strides,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name=name + "_0_conv2d",
        )(inputs)
    else:
        y = inputs

    out = tf.keras.layers.add([x, y], name=name + "_add")
    out = tf.keras.layers.Activation("relu", name=name + "_out")(out)

    return out


def resnet(input_shape, num_blocks=3, num_classes=10):
    inputs = tf.keras.layers.Input(shape=input_shape)

    name = "conv1"
    x = tf.keras.layers.Conv2D(
        16,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name=name + "_conv2d",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(
        epsilon=1e-5, momentum=0.1, name=name + "_bn"
    )(x)
    x = tf.keras.layers.Activation("relu", name=name + "_relu")(x)

    name = "conv2"
    x = resnet_basic_block(inputs=x, num_filters=16, strides=1, name=name + "_block1")
    for blocks in range(num_blocks - 1):
        x = resnet_basic_block(
            inputs=x, num_filters=16, strides=1, name=name + "_block" + str(blocks + 2)
        )

    name = "conv3"
    x = resnet_basic_block(inputs=x, num_filters=32, strides=2, name=name + "_block1")
    for blocks in range(num_blocks - 1):
        x = resnet_basic_block(
            inputs=x, num_filters=32, strides=1, name=name + "_block" + str(blocks + 2)
        )

    name = "conv4"
    x = resnet_basic_block(inputs=x, num_filters=64, strides=2, name=name + "_block1")
    for blocks in range(num_blocks - 1):
        x = resnet_basic_block(
            inputs=x, num_filters=64, strides=1, name=name + "_block" + str(blocks + 2)
        )

    # Add the classifier
    x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", kernel_initializer="he_normal"
    )(x)

    # Build the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def resnet20(input_shape, num_classes=10):
    return resnet(input_shape, 3, num_classes=num_classes)


def resnet32(input_shape, num_classes=10):
    return resnet(input_shape, 5, num_classes=num_classes)


def resnet44(input_shape, num_classes=10):
    return resnet(input_shape, 7, num_classes=num_classes)


def resnet56(input_shape, num_classes=10):
    return resnet(input_shape, 9, num_classes=num_classes)


def resnet110(input_shape, num_classes=10):
    return resnet(input_shape, 18, num_classes=num_classes)


def resnet1202(input_shape, num_classes=10):
    return resnet(input_shape, 200, num_classes=num_classes)


if __name__ == "__main__":
    model = resnet20(input_shape=(32, 32, 3))

    model.summary()

    plot_model_filename = "resnet20_plot.png"
    tf.keras.utils.plot_model(model, to_file=plot_model_filename, show_shapes=True)

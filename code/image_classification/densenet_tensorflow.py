import tensorflow as tf

#
# DenseNet (k12d40, k12d100, k24d100) & DenseNet-BC (k12d100, k24d250, k40d190) for CIFAR10
#
# Huang, Gao, et al. "Densely connected convolutional networks." (2017) [1]
#  - https://arxiv.org/pdf/1608.06993.pdf
#
# Code from paper:
#  https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua [2]
#


def densenet_basic_block(inputs, growth_rate, bottleneck, name=None):
    x = inputs

    # A 1×1 convolution can be introduced as bottleneck layer before each 3×3 convolution
    # to reduce the number of input feature-maps, and thus to improve computational
    # efficiency. Section 3 "Bottleneck layers" [1]
    if bottleneck:
        x = tf.keras.layers.BatchNormalization(name=name + "_0_bn")(x)
        x = tf.keras.layers.Activation("relu", name=name + "_0_relu")(x)
        # We let each 1×1 convolution produce 4k (4 * growth_rate) feature-maps.
        # Section 3 "Bottleneck layers" [1]
        x = tf.keras.layers.Conv2D(
            4 * growth_rate,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name=name + "_0_conv2d",
        )(x)

    # For convolutional layers with kernel size 3×3, each side of the inputs is zero-padded by one pixel
    # to keep the feature-map size fixed. Section 3 "Implementation Details" [1]
    x = tf.keras.layers.BatchNormalization(name=name + "_1_bn")(x)
    x = tf.keras.layers.Activation("relu", name=name + "_1_relu")(x)
    x = tf.keras.layers.Conv2D(
        growth_rate,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name=name + "_1_conv2d",
    )(x)

    out = tf.keras.layers.Concatenate(name=name + "_concat")([inputs, x])

    return out


def densenet_transition_block(inputs, n_channels, compression_reduction, name=None):
    #
    # We use 1×1 convolution followed by 2×2 average pooling as transition layers
    # between two contiguous dense blocks. Section 3 "Implementation Details" [1]
    #
    x = tf.keras.layers.BatchNormalization(name=name + "_bn")(inputs)
    x = tf.keras.layers.Activation("relu", name=name + "_relu")(x)
    x = tf.keras.layers.Conv2D(
        int(n_channels * compression_reduction),
        kernel_size=1,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name=name + "_conv",
    )(x)

    out = tf.keras.layers.AveragePooling2D(
        2, strides=2, padding="same", name=name + "_pool"
    )(x)

    return out


def densenet(
    input_shape, depth, growth_rate, compression_reduction, bottleneck, num_classes=10
):
    # Dense connected layers in each denseblock (DenseNet) [2]
    n = int((depth - 4) / 3)

    # See Section 3 "Implementation Details" [1]
    input_filter = 16

    if bottleneck:
        # See Section 3 "Implementation Details" [1]
        input_filter = growth_rate * 2
        # Dense connected layers in each denseblock (DenseNet-BC) [2]
        n = int(n / 2)

    # Track the growth after each dense block
    n_channels = input_filter

    inputs = tf.keras.layers.Input(shape=input_shape)

    # Before entering the first dense block, a convolution with 16 (or twice the growth
    # rate for DenseNet-BC). Section 3 "Implementation Details" [1] & [2]
    x = tf.keras.layers.Conv2D(
        input_filter,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="conv1_conv2d",
    )(inputs)

    #
    # DenseNet used in our experiments has three dense blocks that each has
    # an equal number of layers. Section 3 "Implementation Details" [1]
    #

    # First dense block
    for i in range(n):
        x = densenet_basic_block(
            x, growth_rate, bottleneck, name="conv2_block" + str(i + 1)
        )
        # Each layer adds k feature-maps of its own to this state. The growth rate
        # regulates how much new information each layer contributes to the global state.
        # Section 3 "Growth rate" [1]
        n_channels += growth_rate

    # First transition block
    x = densenet_transition_block(x, n_channels, compression_reduction, name="trans1")
    # To further improve model compactness, we can reduce the number of feature-maps at
    # transition layers. When θ = 1, the number of feature-maps across transition layers
    # remains unchanged. We refer the DenseNet with θ < 1 as DenseNet-C, and we set θ = 0.5
    # in our experiment. Section 3 "Compression" [1]
    n_channels = int(n_channels * compression_reduction)

    # Second dense block
    for i in range(n):
        x = densenet_basic_block(
            x, growth_rate, bottleneck, name="conv3_block" + str(i + 1)
        )
        # Each layer adds k feature-maps of its own to this state. The growth rate
        # regulates how much new information each layer contributes to the global state.
        # Section 3 "Growth rate" [1]
        n_channels += growth_rate

    # Second transition block
    x = densenet_transition_block(x, n_channels, compression_reduction, name="trans2")
    # To further improve model compactness, we can reduce the number of feature-maps at
    # transition layers. When θ = 1, the number of feature-maps across transition layers
    # remains unchanged. We refer the DenseNet with θ < 1 as DenseNet-C, and we set θ = 0.5
    # in our experiment. Section 3 "Compression" [1]
    n_channels = int(n_channels * compression_reduction)

    # Third dense block
    for i in range(n):
        x = densenet_basic_block(
            x, growth_rate, bottleneck, name="conv4_block" + str(i + 1)
        )
        # Each layer adds k feature-maps of its own to this state. The growth rate
        # regulates how much new information each layer contributes to the global state.
        # Section 3 "Growth rate" [1]
        n_channels += growth_rate

    # Last transition block before the classifier. Only use BN-ReLU after the last
    # dense block. See addTransition() [2]
    x = tf.keras.layers.BatchNormalization(name="bn")(x)
    x = tf.keras.layers.Activation("relu", name="relu")(x)

    # At the end of the last dense block, a global average pooling is performed and
    # then a softmax classifier is attached. Section 3 "Implementation Details" [1]
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="predictions",
    )(x)

    # Build the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def densenet_k12d40(input_shape, num_classes=10):
    return densenet(
        input_shape,
        depth=40,
        growth_rate=12,
        compression_reduction=1,
        bottleneck=False,
        num_classes=num_classes,
    )


def densenet_k12d100(input_shape, num_classes=10):
    return densenet(
        input_shape,
        depth=100,
        growth_rate=12,
        compression_reduction=1,
        bottleneck=False,
        num_classes=num_classes,
    )


def densenet_k24d100(input_shape, num_classes=10):
    return densenet(
        input_shape,
        depth=100,
        growth_rate=24,
        compression_reduction=1,
        bottleneck=False,
        num_classes=num_classes,
    )


def densenet_bc_k12d100(input_shape, num_classes=10):
    return densenet(
        input_shape,
        depth=100,
        growth_rate=12,
        compression_reduction=0.5,
        bottleneck=True,
        num_classes=num_classes,
    )


def densenet_bc_k24d250(input_shape, num_classes=10):
    return densenet(
        input_shape,
        depth=250,
        growth_rate=24,
        compression_reduction=0.5,
        bottleneck=True,
        num_classes=num_classes,
    )


def densenet_bc_k40d190(input_shape, num_classes=10):
    return densenet(
        input_shape,
        depth=190,
        growth_rate=40,
        compression_reduction=0.5,
        bottleneck=True,
        num_classes=num_classes,
    )


if __name__ == "__main__":
    model = densenet_k12d40((32, 32, 3), num_classes=10)

    model.summary()

    plot_model_filename = "densenet_bc_k12d40_plot.png"
    tf.keras.utils.plot_model(model, to_file=plot_model_filename, show_shapes=True)

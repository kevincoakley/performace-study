import tensorflow as tf

num_classes = None
image_shape = None


def preprocessing(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, image_shape[:2])
    return image, tf.squeeze(tf.one_hot(label, depth=num_classes))


def augmentation(image, label):
    image = tf.image.resize_with_crop_or_pad(
        image, image_shape[0] + 15, image_shape[1] + 15
    )
    image = tf.image.random_crop(image, image_shape, seed=1)
    image = tf.image.random_flip_left_right(image, seed=1)
    return image, label


def get_input(X_train, y_train, X_val, y_val, batch_size, shuffle_seed):
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train))
        .map(preprocessing)
        .repeat()
        .map(augmentation)
        .shuffle(10000, seed=shuffle_seed)
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    val_dataset = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .map(preprocessing)
        .batch(200)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return train_dataset, val_dataset


def get_dataset(dataset_name, batch_size, shuffle_seed, shape=[32, 32, 3]):
    global num_classes
    global image_shape
    image_shape = shape

    if dataset_name == "cifar10":
        num_classes = 10
        (X_train, y_train), (X_val, y_val) = tf.keras.datasets.cifar10.load_data()
        return get_input(X_train, y_train, X_val, y_val, batch_size, shuffle_seed)
    elif dataset_name == "cifar100":
        num_classes = 100
        (X_train, y_train), (X_val, y_val) = tf.keras.datasets.cifar100.load_data()
        return get_input(X_train, y_train, X_val, y_val, batch_size, shuffle_seed)
    elif dataset_name == "fashion_mnist":
        num_classes = 10
        (X_train, y_train), (X_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()
        X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
        X_val = X_val.reshape((X_val.shape[0], 28, 28, 1))
        return get_input(X_train, y_train, X_val, y_val, batch_size, shuffle_seed)

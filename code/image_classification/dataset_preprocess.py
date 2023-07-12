import tensorflow as tf
import tensorflow_datasets as tfds

num_classes = None
image_shape = None


def preprocessing(image, label):
    image = tf.cast(image, tf.float32)
    # Normalize the pixel values ((input[channel] - mean[channel]) / std[channel])
    image = tf.divide(image, (255.0, 255.0, 255.0))  # divide by 255 to match pytorch
    image = tf.subtract(image, (0.5, 0.5, 0.5))
    image = tf.divide(image, (0.5, 0.5, 0.5))
    image = tf.image.resize(image, image_shape[:2], antialias=False, method="nearest")
    return image, tf.squeeze(tf.one_hot(label, depth=num_classes))


def augmentation(image, label):
    image = tf.image.resize_with_crop_or_pad(
        image, image_shape[0] + 20, image_shape[1] + 20
    )
    image = tf.image.random_crop(image, image_shape)
    image = tf.image.random_flip_left_right(image)
    return image, label


def get_input(train, val, batch_size, shuffle_seed):
    train_dataset = (
        train.map(preprocessing)
        .map(augmentation)
        .shuffle(10000, seed=shuffle_seed)
        .batch(batch_size, drop_remainder=True)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = val.map(preprocessing).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_dataset, val_dataset, len(list(train)), len(list(val))


def get_dataset(dataset_name, batch_size, shuffle_seed, shape=[32, 32, 3]):
    global num_classes
    global image_shape
    image_shape = shape

    if dataset_name == "cifar10":
        num_classes = 10
        train_dataset, val_dataset = tfds.load(
            name="cifar10", split=["train", "test"], as_supervised=True
        )
        print(train_dataset)
        return get_input(train_dataset, val_dataset, batch_size, shuffle_seed)
    elif dataset_name == "cifar100":
        num_classes = 100
        train_dataset, val_dataset = tfds.load(
            name="cifar100", split=["train", "test"], as_supervised=True
        )
        return get_input(train_dataset, val_dataset, batch_size, shuffle_seed)
    elif dataset_name == "fashion_mnist":
        num_classes = 10
        train_dataset, val_dataset = tfds.load(
            name="fashion_mnist", split=["train", "test"], as_supervised=True
        )
        print(train_dataset)
        return get_input(train_dataset, val_dataset, batch_size, shuffle_seed)
    elif dataset_name == "cats_vs_dogs":
        num_classes = 2
        split = ["train[:70%]", "train[70%:]"]
        train_dataset, val_dataset = tfds.load(
            name="cats_vs_dogs", split=split, as_supervised=True
        )
        return get_input(train_dataset, val_dataset, batch_size, shuffle_seed)

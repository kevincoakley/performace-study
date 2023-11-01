import tensorflow as tf

import csv, math, random
import numpy as np
from sklearn.metrics import accuracy_score

import resnet_tensorflow as resnet
import densenet_tensorflow as densenet


class Tensorflow:
    def __init__(self):
        self.version = tf.version.VERSION
        self.epochs = 0
        self.lr_warmup = False

    def deterministic(self, seed_val):
        """
        ## Configure Tensorflow for fixed seed runs
        """
        major, minor, revision = tf.version.VERSION.split(".")

        if int(major) >= 2 and int(minor) >= 7:
            # Sets all random seeds for the program (Python, NumPy, and TensorFlow).
            # Supported in TF 2.7.0+
            tf.keras.utils.set_random_seed(seed_val)
            print("Setting random seed using tf.keras.utils.set_random_seed()")
        else:
            # for TF < 2.7
            random.seed(seed_val)
            np.random.seed(seed_val)
            tf.random.set_seed(seed_val)
            print("Setting random seeds manually")
        # Configures TensorFlow ops to run deterministically to enable reproducible
        # results with GPUs (Supported in TF 2.8.0+)
        if int(major) >= 2 and int(minor) >= 8:
            tf.config.experimental.enable_op_determinism()
            print("Enabled op determinism")

    def load_dataset(
        self,
        model_details,
        dataset_details,
        dataset_seed_val,
    ):
        batch_size = model_details["batch_size"]
        train_path = dataset_details["train_path"]
        val_path = dataset_details["val_path"]
        test_path = dataset_details["test_path"]
        dataset_shape = dataset_details["dataset_shape"]
        normalization_mean = dataset_details["normalization"]["mean"]
        normalization_std = dataset_details["normalization"]["std"]

        def preprocessing(image, label):
            image = tf.cast(image, tf.float32)
            # Normalize the pixel values ((input[channel] - mean[channel]) / std[channel])
            image = tf.divide(
                image, (255.0, 255.0, 255.0)
            )  # divide by 255 to match pytorch
            image = tf.subtract(image, normalization_mean)
            image = tf.divide(image, normalization_std)
            return image, label

        def augmentation(image, label):
            image = tf.image.resize_with_crop_or_pad(
                image, dataset_shape[0] + 8, dataset_shape[1] + 8
            )
            image = tf.image.random_crop(image, dataset_shape)
            image = tf.image.random_flip_left_right(image)
            return image, label

        # Get the training and validation datasets from the directory
        train = tf.keras.utils.image_dataset_from_directory(
            train_path,
            shuffle=True,
            image_size=dataset_shape[:2],
            interpolation="nearest",
            batch_size=None,
        )
        val = tf.keras.utils.image_dataset_from_directory(
            val_path,
            shuffle=False,
            image_size=dataset_shape[:2],
            interpolation="nearest",
            batch_size=None,
        )
        test = tf.keras.utils.image_dataset_from_directory(
            test_path,
            shuffle=False,
            image_size=dataset_shape[:2],
            interpolation="nearest",
            batch_size=None,
        )

        # Batch and prefetch the dataset
        train_dataset = (
            train.map(preprocessing)
            .map(augmentation)
            .shuffle(1000, seed=dataset_seed_val)
            .batch(batch_size, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_dataset = (
            val.map(preprocessing).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        )
        test_dataset = (
            test.map(preprocessing).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        )

        return train_dataset, val_dataset, test_dataset

    def load_model(self, model_name, dataset_details):
        num_classes = dataset_details["num_classes"]
        dataset_shape = dataset_details["dataset_shape"]

        model_functions = {
            "ResNet20": resnet.resnet20,
            "ResNet32": resnet.resnet32,
            "ResNet44": resnet.resnet44,
            "ResNet56": resnet.resnet56,
            "ResNet110": resnet.resnet110,
            "ResNet1202": resnet.resnet1202,
            "DenseNet_k12d40": densenet.densenet_k12d40,
            "DenseNet_k12d100": densenet.densenet_k12d100,
            "DenseNet_k24d100": densenet.densenet_k24d100,
            "DenseNet_bc_k12d100": densenet.densenet_bc_k12d100,
            "DenseNet_bc_k24d250": densenet.densenet_bc_k24d250,
            "DenseNet_bc_k40d190": densenet.densenet_bc_k40d190,
        }

        model = model_functions[model_name](
            input_shape=dataset_shape, num_classes=num_classes
        )

        model.build(
            input_shape=(None, dataset_shape[0], dataset_shape[1], dataset_shape[2])
        )

        # Print the model summary
        # model.summary()

        return model

    def train(
        self,
        model,
        train_dataset,
        val_dataset,
        model_details,
        epochs,
        save_epoch_logs=False,
        csv_train_log_file=None,
    ):
        nesterov = model_details["nesterov"]

        """
        ## Define the learning rate schedule
        """

        def lr_schedule(epoch):
            if self.lr_warmup and epoch < 5:
                return 0.01
            elif epoch < math.ceil(self.epochs * 0.5):
                return 0.1
            elif epoch < math.ceil(self.epochs * 0.75):
                return 0.01
            else:
                return 0.001

        model.compile(
            optimizer=tf.keras.optimizers.experimental.SGD(
                weight_decay=0.0001,
                momentum=0.9,
                learning_rate=lr_schedule(0),
                nesterov=nesterov,
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

        ## Define csv logger callback
        csv_logger = tf.keras.callbacks.CSVLogger(csv_train_log_file)

        # Define callbacks
        if save_epoch_logs:
            callbacks = [csv_logger, lr_scheduler]
        else:
            callbacks = [lr_scheduler]

        # Train the model
        model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
        )

        return model

    def evaluate(
        self, model, val_dataset, save_predictions=False, predictions_csv_file=None
    ):
        # Get the predictions
        predictions = model.predict(val_dataset)

        # Get the labels of the validation dataset
        val_dataset = val_dataset.unbatch()
        labels = np.asarray(list(val_dataset.map(lambda x, y: y)))

        # Get the index to the highest probability
        y_true = labels
        y_pred = np.argmax(predictions, axis=1)

        if save_predictions:
            # Add the true values to the first column and the predicted values to the second column
            true_and_pred = np.vstack((y_true, y_pred)).T

            # Add each label predictions to the true and predicted values
            csv_output_array = np.concatenate((true_and_pred, predictions), axis=1)

            # Save the predictions to a csv file
            with open(predictions_csv_file, "w") as csvfile:
                writer = csv.writer(csvfile)

                csv_columns = ["true_value", "predicted_value"]
                for i in range(predictions.shape[1]):
                    csv_columns.append("label_" + str(i))

                writer.writerow(csv_columns)
                writer.writerows(csv_output_array.tolist())

        # Calucate the validation loss
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        validation_loss = loss(labels, predictions).numpy()

        # Use sklearn to calculate the validation accuracy
        validation_accuracy = accuracy_score(y_true, y_pred)

        return [validation_loss, validation_accuracy]

    def save(self, model, model_path):
        model.save(model_path)

    def load(self, model_path):
        model = tf.keras.models.load_model(model_path)

        return model

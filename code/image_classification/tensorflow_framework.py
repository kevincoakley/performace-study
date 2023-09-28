import tensorflow as tf

import csv, random
import numpy as np
from sklearn.metrics import accuracy_score


class Tensorflow:
    def __init__(self):
        self.version = tf.version.VERSION

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
        dataset_details,
        dataset_seed_val,
    ):
        train_path = dataset_details["train_path"]
        val_path = dataset_details["val_path"]
        dataset_shape = dataset_details["dataset_shape"]
        batch_size = dataset_details["batch_size"]

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

        # Batch and prefetch the dataset
        train_dataset = train.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        return train_dataset, val_dataset

    def load_model(self, model_name, dataset_details):
        num_classes = dataset_details["num_classes"]
        training_shape = dataset_details["training_shape"]
        normalization_mean = dataset_details["normalization"]["mean"]
        normalization_std = dataset_details["normalization"]["std"]

        model_dictionary = {
            "EfficientNetB4": {
                "application": tf.keras.applications.EfficientNetB4,
                "args": {
                    "include_top": True,
                    "weights": None,
                    "input_shape": training_shape,
                    "classes": num_classes,
                    "classifier_activation": "softmax",
                },
            },
            "InceptionV3": {
                "application": tf.keras.applications.InceptionV3,
                "args": {
                    "include_top": True,
                    "weights": None,
                    "input_shape": training_shape,
                    "classes": num_classes,
                    "classifier_activation": "softmax",
                },
            },
            "ResNet50": {
                "application": tf.keras.applications.resnet_v2.ResNet50V2,
                "args": {
                    "include_top": True,
                    "weights": None,
                    "input_shape": training_shape,
                    "classes": num_classes,
                    "classifier_activation": "softmax",
                },
            },
            "ResNet101": {
                "application": tf.keras.applications.resnet_v2.ResNet101V2,
                "args": {
                    "include_top": True,
                    "weights": None,
                    "input_shape": training_shape,
                    "classes": num_classes,
                    "classifier_activation": "softmax",
                },
            },
            "ResNet152": {
                "application": tf.keras.applications.resnet_v2.ResNet152V2,
                "args": {
                    "include_top": True,
                    "weights": None,
                    "input_shape": training_shape,
                    "classes": num_classes,
                    "classifier_activation": "softmax",
                },
            },
            "DenseNet121": {
                "application": tf.keras.applications.DenseNet121,
                "args": {
                    "include_top": True,
                    "weights": None,
                    "input_shape": training_shape,
                    "classes": num_classes,
                    "classifier_activation": "softmax",
                },
            },
            "DenseNet169": {
                "application": tf.keras.applications.DenseNet169,
                "args": {
                    "include_top": True,
                    "weights": None,
                    "input_shape": training_shape,
                    "classes": num_classes,
                    "classifier_activation": "softmax",
                },
            },
            "DenseNet201": {
                "application": tf.keras.applications.DenseNet201,
                "args": {
                    "include_top": True,
                    "weights": None,
                    "input_shape": training_shape,
                    "classes": num_classes,
                    "classifier_activation": "softmax",
                },
            },
        }

        preprocessing = tf.keras.Sequential(
            [
                tf.keras.layers.Rescaling(1.0 / 255),
                tf.keras.layers.Normalization(
                    mean=normalization_mean, variance=np.square(normalization_std)
                ),
                tf.keras.layers.Resizing(
                    training_shape[0], training_shape[1], interpolation="nearest"
                ),
            ],
            name="preprocessing",
        )

        data_augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.ZeroPadding2D(10),
                tf.keras.layers.RandomCrop(training_shape[0], training_shape[1]),
                tf.keras.layers.RandomFlip("horizontal"),
            ],
            name="data_augmentation",
        )

        application = model_dictionary[model_name]["application"](
            **model_dictionary[model_name]["args"]
        )

        model = tf.keras.Sequential()
        model.add(preprocessing)
        model.add(data_augmentation)
        model.add(application)

        model.build(
            input_shape=(None, training_shape[0], training_shape[1], training_shape[2])
        )

        return model

    def train(
        self,
        model,
        train_dataset,
        val_dataset,
        epochs,
        learning_rate,
        save_epoch_logs=False,
        csv_train_log_file=None,
    ):
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        # Print the model summary
        model.summary()

        ## Define csv logger callback
        csv_logger = tf.keras.callbacks.CSVLogger(csv_train_log_file)

        # Define callbacks
        if save_epoch_logs:
            callbacks = [csv_logger]
        else:
            callbacks = []

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

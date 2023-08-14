import tensorflow as tf
import tensorflow_dataset_preprocess

import csv, random
import numpy as np
from sklearn.metrics import accuracy_score


class Tensorflow():
    def __init__(self):
        self.version = tf.version.VERSION
        self.train_steps_per_epoch = None
        self.val_steps_per_epoch = None
        
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

    def load_dataset(self, dataset_name, batch_size, input_shape, dataset_seed_val):     
        train_dataset, val_dataset, train_size, val_size = tensorflow_dataset_preprocess.get_dataset(
            dataset_name, batch_size, dataset_seed_val=dataset_seed_val, shape=input_shape
        )   

        self.train_steps_per_epoch = train_size // batch_size
        self.val_steps_per_epoch = val_size // batch_size

        return train_dataset, val_dataset

    def load_model(self, model_name, input_shape, num_classes):
        model_dictionary = {
            "EfficientNetB4": {
                "application": tf.keras.applications.EfficientNetB4,
                "args": {
                    "include_top": True,
                    "weights": None,
                    "input_shape": input_shape,
                    "classes": num_classes,
                    "classifier_activation": "softmax",
                },
            },
            "InceptionV3": {
                "application": tf.keras.applications.InceptionV3,
                "args": {
                    "include_top": True,
                    "weights": None,
                    "input_shape": input_shape,
                    "classes": num_classes,
                    "classifier_activation": "softmax",
                },
            },
            "ResNet50": {
                "application": tf.keras.applications.resnet_v2.ResNet50V2,
                "args": {
                    "include_top": True,
                    "weights": None,
                    "input_shape": input_shape,
                    "classes": num_classes,
                    "classifier_activation": "softmax",
                },
            },
            "ResNet101": {
                "application": tf.keras.applications.resnet_v2.ResNet101V2,
                "args": {
                    "include_top": True,
                    "weights": None,
                    "input_shape": input_shape,
                    "classes": num_classes,
                    "classifier_activation": "softmax",
                },
            },
            "ResNet152": {
                "application": tf.keras.applications.resnet_v2.ResNet152V2,
                "args": {
                    "include_top": True,
                    "weights": None,
                    "input_shape": input_shape,
                    "classes": num_classes,
                    "classifier_activation": "softmax",
                },
            },
            "DenseNet121": {
                "application": tf.keras.applications.DenseNet121,
                "args": {
                    "include_top": True,
                    "weights": None,
                    "input_shape": input_shape,
                    "classes": num_classes,
                    "classifier_activation": "softmax",
                },
            },
            "DenseNet169": {
                "application": tf.keras.applications.DenseNet169,
                "args": {
                    "include_top": True,
                    "weights": None,
                    "input_shape": input_shape,
                    "classes": num_classes,
                    "classifier_activation": "softmax",
                },
            },
            "DenseNet201": {
                "application": tf.keras.applications.DenseNet201,
                "args": {
                    "include_top": True,
                    "weights": None,
                    "input_shape": input_shape,
                    "classes": num_classes,
                    "classifier_activation": "softmax",
                },
            },
        }

        model = tf.keras.Sequential()
        model.add(model_dictionary[model_name]["application"](**model_dictionary[model_name]["args"]))

        return model

    def train(self, model, train_dataset, val_dataset, epochs, learning_rate):
 
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Print the model summary
        model.summary()

        """
        ## Define csv logger callback
        """
        #csv_train_log_file = base_name + "_log_" + str(run_number) + ".csv"

        #csv_logger = tf.keras.callbacks.CSVLogger(csv_train_log_file)

        # Define callbacks
        #callbacks = [csv_logger]

        # Train the model
        model.fit(
            train_dataset,
            epochs=epochs,
            steps_per_epoch=self.train_steps_per_epoch,
            validation_data=val_dataset,
            #callbacks=callbacks,
        )

        return model

    def evaluate(self, model, val_dataset, save_predictions=False, predictions_csv_file=None):
        # Get the predictions
        predictions = model.predict(val_dataset)

        # Get the labels of the validation dataset
        val_dataset = val_dataset.unbatch()
        labels = np.asarray(list(val_dataset.map(lambda x, y: y)))

        # Get the index to the highest probability
        y_true = np.argmax(labels, axis=1)
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

        # Calucate the validation loss by averaging the loss of all the samples
        loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
        validation_loss = np.sum(loss.numpy()) / len(loss.numpy())

        # Use sklearn to calculate the validation score
        validation_score = accuracy_score(y_true, y_pred)

        return [validation_loss, validation_score]

    def save(self, model, model_path):
        model.save(model_path)

    def load(self, model_path):
        pass
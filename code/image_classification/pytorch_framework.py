import torch
import torchvision

import csv, random
import numpy as np
from sklearn.metrics import accuracy_score


class Pytorch:
    def __init__(self):
        self.version = torch.__version__
        self.device = torch.device("cuda:0")

    def deterministic(self, seed_val):
        torch.manual_seed(seed_val)
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.use_deterministic_algorithms(True)

    def load_dataset(
        self,
        train_path,
        val_path,
        num_classes,
        batch_size,
        image_shape,
        dataset_seed_val,
    ):
        data_generator = torch.Generator()
        data_generator.manual_seed(dataset_seed_val)

        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                torchvision.transforms.Resize(
                    image_shape[:2],
                    antialias=False,
                    interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                ),
            ]
        )

        transform_augment = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                torchvision.transforms.Resize(
                    image_shape[:2],
                    antialias=False,
                    interpolation=torchvision.transforms.InterpolationMode.NEAREST,
                ),
                torchvision.transforms.Pad(10),
                torchvision.transforms.RandomCrop((128, 128)),
                torchvision.transforms.RandomHorizontalFlip(),
            ]
        )

        train_dataset = torchvision.datasets.ImageFolder(
            root=train_path, transform=transform_augment
        )
        val_dataset = torchvision.datasets.ImageFolder(
            root=val_path, transform=transform
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=1,
            generator=data_generator,
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            generator=data_generator,
        )

        return train_dataloader, val_dataloader

    def load_model(self, model_name, input_shape, num_classes):
        model_dictionary = {
            "EfficientNetB4": {
                "application": torchvision.models.efficientnet_b4,
                "args": {
                    "weights": None,
                    "num_classes": num_classes,
                },
            },
            "InceptionV3": {
                "application": torchvision.models.inception_v3,
                "args": {
                    "weights": None,
                    "num_classes": num_classes,
                },
            },
            "ResNet50": {
                "application": torchvision.models.resnet50,
                "args": {
                    "weights": None,
                    "num_classes": num_classes,
                },
            },
            "ResNet101": {
                "application": torchvision.models.resnet101,
                "args": {
                    "weights": None,
                    "num_classes": num_classes,
                },
            },
            "ResNet152": {
                "application": torchvision.models.resnet152,
                "args": {
                    "weights": None,
                    "num_classes": num_classes,
                },
            },
            "DenseNet121": {
                "application": torchvision.models.densenet121,
                "args": {
                    "weights": None,
                    "num_classes": num_classes,
                },
            },
            "DenseNet169": {
                "application": torchvision.models.densenet169,
                "args": {
                    "weights": None,
                    "num_classes": num_classes,
                },
            },
            "DenseNet201": {
                "application": torchvision.models.densenet201,
                "args": {
                    "weights": None,
                    "num_classes": num_classes,
                },
            },
        }

        model = model_dictionary[model_name]["application"](
            **model_dictionary[model_name]["args"]
        )
        model.to(self.device)

        return model

    def train(
        self,
        model,
        train_dataloader,
        val_dataloader,
        epochs,
        learning_rate,
        save_epoch_logs=False,
        csv_train_log_file=None,
    ):
        epoch_logs = []

        loss_function = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        def train_one_epoch():
            running_loss = 0
            running_predicted = []
            running_labels = []

            batch_epoch_loss = 0.0

            # Loop through the training dataset by batches
            for i, data in enumerate(train_dataloader, 0):
                # Every data instance is an input + label pair
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # Zero your gradients every batch
                optimizer.zero_grad()

                # Make predictions for this batch
                outputs = model(inputs)
                # Compute the loss and its gradients
                loss = loss_function(outputs, labels)
                loss.backward()
                # Adjust learning weights
                optimizer.step()

                # Store the running loss for the training dataset
                running_loss += loss

                # Store the batch predicted values for the training dataset
                _, predicted = torch.max(outputs.data, 1)
                running_predicted.extend(predicted.tolist())
                running_labels.extend(labels.tolist())

                # Report the accuracy and loss every 250 batches
                if i % 250 == 249:
                    # loss per batch
                    batch_epoch_loss = running_loss.item() / i
                    print(
                        "     batch %s loss: %s accuracy: %s"
                        % (
                            i + 1,
                            batch_epoch_loss,
                            accuracy_score(running_labels, running_predicted),
                        )
                    )

            # Calucate the training loss by dividing the total loss by number of batches
            epoch_loss = running_loss.item() / len(train_dataloader)

            # Use sklearn to calculate the training accuracy
            epoch_accuracy = accuracy_score(running_labels, running_predicted)

            return epoch_loss, epoch_accuracy

        for epoch in range(epochs):
            print("EPOCH %s/%s:" % (epoch + 1, epochs))

            # Make sure gradient tracking is on, and do a pass over the data
            model.train(True)
            train_loss, train_accuracy = train_one_epoch()

            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            model.eval()

            running_val_loss = 0.0
            running_val_predicted = []
            running_val_labels = []

            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                for i, val_data in enumerate(val_dataloader, 0):
                    val_inputs, val_labels = val_data[0].to(self.device), val_data[
                        1
                    ].to(self.device)

                    val_outputs = model(val_inputs)

                    # Calculate the batch loss for the val dataset
                    running_val_loss += loss_function(val_outputs, val_labels)

                    # Calculate the batch predicted values for the val dataset
                    _, val_predicted = torch.max(val_outputs.data, 1)
                    running_val_predicted.extend(val_predicted.tolist())
                    running_val_labels.extend(val_labels.tolist())

            # Calucate the validation loss by dividing the total loss by number of batches
            validation_loss = running_val_loss.item() / len(val_dataloader)
            # Use sklearn to calculate the validation accuracy
            validation_accuracy = accuracy_score(
                running_val_labels, running_val_predicted
            )

            print(
                "EPOCH %s/%s END: TRAIN loss: %s acc: %s VAL loss: %s acc: %s"
                % (
                    epoch + 1,
                    epochs,
                    train_loss,
                    train_accuracy,
                    validation_loss,
                    validation_accuracy,
                )
            )

            epoch_logs.append(
                [
                    epoch,
                    train_accuracy,
                    train_loss,
                    validation_accuracy,
                    validation_loss,
                ]
            )

        if save_epoch_logs:
            # Save the epoch log to a csv file
            with open(csv_train_log_file, "w") as csvfile:
                writer = csv.writer(csvfile)

                csv_columns = ["epoch", "accuracy", "loss", "val_accuracy", "val_loss"]

                writer.writerow(csv_columns)
                writer.writerows(epoch_logs)

        return model

    def evaluate(
        self, model, dataloader, save_predictions=False, predictions_csv_file=None
    ):
        loss_function = torch.nn.CrossEntropyLoss()
        loss = 0

        running_predicted = []
        running_labels = []
        predictions = []

        # Forward pass only. Get the predictions and calculate the loss and accuarcy
        with torch.no_grad():
            for data in dataloader:
                # Get the images and labels from the dataloader
                images, labels = data[0].to(self.device), data[1].to(self.device)

                # Calculate the prediction values for each image
                outputs = model(images)

                # Calculate the loss for the batch
                loss += loss_function(outputs, labels)

                # Select the largest prediction value for each class
                _, predicted = torch.max(outputs.data, 1)

                running_predicted.extend(predicted.tolist())
                running_labels.extend(labels.tolist())

                if save_predictions:
                    # loop through the batch and add each prediction to the predictions list
                    for output in outputs:
                        predictions.append(output.tolist())

        if save_predictions:
            # Add the true values to the first column and the predicted values to the second column
            true_and_pred = np.vstack((running_labels, running_predicted)).T

            # Add each label predictions to the true and predicted values
            csv_output_array = np.concatenate((true_and_pred, predictions), axis=1)

            # Save the predictions to a csv file
            with open(predictions_csv_file, "w") as csvfile:
                writer = csv.writer(csvfile)

                csv_columns = ["true_value", "predicted_value"]
                for i in range(len(predictions[0])):
                    csv_columns.append("label_" + str(i))

                writer.writerow(csv_columns)
                writer.writerows(csv_output_array.tolist())

        # Calucate the validation loss by dividing the total loss by number of batches
        validation_loss = loss.item() / len(dataloader)
        # Use sklearn to calculate the validation accuracy
        validation_accuracy = accuracy_score(running_labels, running_predicted)

        return [validation_loss, validation_accuracy]

    def save(self, model, model_path):
        torch.save(model, model_path)

    def load(self, model_path):
        model = torch.load(model_path)
        model.eval()

        return model

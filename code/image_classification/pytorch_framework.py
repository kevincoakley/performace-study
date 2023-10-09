import torch
import torchvision

torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2

import csv, random
import numpy as np
from sklearn.metrics import accuracy_score
from datetime import datetime


class Pytorch:
    def __init__(self):
        self.version = torch.__version__
        self.device = torch.device("cuda:0")

    def deterministic(self, seed_val):
        torch.manual_seed(seed_val)
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.use_deterministic_algorithms(True)

    def load_dataset(self, dataset_details, dataset_seed_val):
        train_path = dataset_details["train_path"]
        val_path = dataset_details["val_path"]
        batch_size = dataset_details["batch_size"]

        data_generator = torch.Generator()
        data_generator.manual_seed(dataset_seed_val)

        transform_to_tensor = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        )

        train = torchvision.datasets.ImageFolder(
            root=train_path, transform=transform_to_tensor
        )
        val = torchvision.datasets.ImageFolder(
            root=val_path, transform=transform_to_tensor
        )

        train_dataset = torch.utils.data.DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            generator=data_generator,
            pin_memory=True,
        )

        val_dataset = torch.utils.data.DataLoader(
            val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            generator=data_generator,
            pin_memory=True,
        )

        return train_dataset, val_dataset

    def load_model(self, model_name, dataset_details):
        num_classes = dataset_details["num_classes"]
        training_shape = dataset_details["training_shape"]
        normalization_mean = dataset_details["normalization"]["mean"]
        normalization_std = dataset_details["normalization"]["std"]

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

        preprocessing = torch.nn.Sequential(
            torchvision.transforms.v2.Normalize(normalization_mean, normalization_std),
            torchvision.transforms.v2.Resize(
                training_shape[:2],
                antialias=False,
                interpolation=torchvision.transforms.InterpolationMode.NEAREST,
            ),
        )
        augmentation = torch.nn.Sequential(
            torchvision.transforms.v2.Pad(10),
            torchvision.transforms.v2.RandomCrop(training_shape[:2]),
            torchvision.transforms.v2.RandomHorizontalFlip(),
        )

        application = model_dictionary[model_name]["application"](
            **model_dictionary[model_name]["args"]
        )

        model = torch.nn.Sequential()
        model.add_module("preprocessing", preprocessing)
        model.add_module("augmentation", augmentation)
        model.add_module("application", application)

        model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {total_params}")

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

        criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        def train_one_epoch():
            running_loss = 0
            running_predicted = []
            running_labels = []

            model.train()

            # Loop through the training dataset and train the model
            for i, (input, label) in enumerate(train_dataloader, 0):
                input, label = input.to(self.device), label.to(self.device)

                output = model(input)
                loss = criterion(output, label)

                # Zero your gradients every batch
                optimizer.zero_grad()

                # Compute the loss and its gradients
                loss.backward()
                # Adjust learning weights
                optimizer.step()

                # Store the running loss for the training dataset
                running_loss += loss

                # Select the largest prediction value for each class
                _, predicted = torch.max(output.data, 1)

                # Store the batch predicted values for the training dataset
                running_predicted.extend(predicted.tolist())
                running_labels.extend(label.tolist())

            # Calucate the training loss by dividing the total loss by number of batches
            epoch_loss = running_loss.item() / len(train_dataloader)

            # Use sklearn to calculate the training accuracy
            epoch_accuracy = accuracy_score(running_labels, running_predicted)

            return epoch_loss, epoch_accuracy

        for epoch in range(epochs):
            print("EPOCH %s/%s:" % (epoch + 1, epochs))

            start_time = datetime.now()

            # Train the model for one epoch
            train_loss, train_accuracy = train_one_epoch()

            # Evaluate the model on the validation dataset after each epoch
            validation_loss, validation_accuracy = self.evaluate(
                model, val_dataloader, save_predictions=False, predictions_csv_file=None
            )

            # Calculate the epoch time
            end_time = datetime.now()
            epoch_time = end_time - start_time

            print(
                "EPOCH %s/%s END: TRAIN loss: %s acc: %s VAL loss: %s acc: %s (time: %s)"
                % (
                    epoch + 1,
                    epochs,
                    train_loss,
                    train_accuracy,
                    validation_loss,
                    validation_accuracy,
                    epoch_time,
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
        criterion = torch.nn.CrossEntropyLoss()
        loss = 0

        running_predicted = []
        running_labels = []
        predictions = []

        model.eval()

        with torch.inference_mode():
            for data in dataloader:
                # Get the images and labels from the dataloader
                input = data[0].to(self.device, non_blocking=True)
                labels = data[1].to(self.device, non_blocking=True)

                # Calculate the prediction values for each image
                outputs = model(input)

                # Calculate the loss for the batch
                loss += criterion(outputs, labels)

                # Select the largest prediction value for each class
                _, predicted = torch.max(outputs.data, 1)

                # Store the batch predicted values for the dataset
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

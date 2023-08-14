import torch
import torchvision

import csv, random
import numpy as np
from sklearn.metrics import accuracy_score


class Pytorch():
    def __init__(self):
        self.version = torch.__version__
        self.device = torch.device("cuda:0")
        
    def deterministic(self, seed_val):
        torch.manual_seed(seed_val)
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.use_deterministic_algorithms(True)

    def load_dataset(self, dataset_name, batch_size, input_shape, dataset_seed_val):     
        data_generator = torch.Generator()
        data_generator.manual_seed(dataset_seed_val)

        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                torchvision.transforms.Resize((128, 128), antialias=False, interpolation=torchvision.transforms.InterpolationMode.NEAREST),
            ]
        )

        transform_augment = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                torchvision.transforms.Resize((128, 128), antialias=False, interpolation=torchvision.transforms.InterpolationMode.NEAREST),
                torchvision.transforms.Pad(10),
                torchvision.transforms.RandomCrop((128, 128)),
                torchvision.transforms.RandomHorizontalFlip(),
            ]
        )

        if dataset_name == "cifar10":
            train_dataset = torchvision.datasets.CIFAR10(
                root="./data", train=True, download=True, transform=transform_augment
            )
            val_dataset = torchvision.datasets.CIFAR10(
                root="./data", train=False, download=True, transform=transform
            )
        elif dataset_name == "cifar100":
            train_dataset = torchvision.datasets.CIFAR100(
                root="./data", train=True, download=True, transform=transform_augment
            )
            val_dataset = torchvision.datasets.CIFAR100(
                root="./data", train=False, download=True, transform=transform
            )
        elif dataset_name == "fashion_mnist":
            train_dataset = torchvision.datasets.FashionMNIST(
                root="./data", train=True, download=True, transform=transform_augment
            )
            val_dataset = torchvision.datasets.FashionMNIST(
                root="./data", train=False, download=True, transform=transform
            )
        elif dataset_name == "cats_vs_dogs":
            pass

        train_dataloader  = torch.utils.data.DataLoader(
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

        model = model_dictionary[model_name]["application"](**model_dictionary[model_name]["args"])
        model.to(self.device)

        return model

    def train(self, model, train_dataloader, val_dataloader, epochs, learning_rate):
        criterion = torch.nn.CrossEntropyLoss()
        
        # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                # inputs, labels = data
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 200 == 199:  # print every 2000 mini-batches
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}")
                    running_loss = 0.0

        return model

    def evaluate(self, model, dataloader, save_predictions=False, predictions_csv_file=None):
        criterion = torch.nn.CrossEntropyLoss()
        loss = 0

        sklearn_pred = []
        sklearn_labels = []
        predictions = []

        # Forward pass only. Get the predictions and calculate the loss and accuarcy
        with torch.no_grad():
            for data in dataloader:
                # Get the images and labels from the dataloader
                images, labels = data[0].to(self.device), data[1].to(self.device)

                # Calculate the prediction values for each image
                outputs = model(images)
               
                # Calculate the loss for the batch
                loss += criterion(outputs, labels)

                # Select the largest prediction value for each class
                _, predicted = torch.max(outputs.data, 1)

                sklearn_pred.extend(predicted.tolist())
                sklearn_labels.extend(labels.tolist())

                if save_predictions:
                    # loop through the batch and add each prediction to the predictions list
                    for output in outputs:
                        predictions.append(output.tolist())
        
                    
        if save_predictions:
            # Add the true values to the first column and the predicted values to the second column
            true_and_pred = np.vstack((sklearn_labels, sklearn_pred)).T

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
        # Use sklearn to calculate the validation score
        validation_score = accuracy_score(sklearn_labels, sklearn_pred)

        return [validation_loss, validation_score]

    def save(self, model, model_path):
        torch.save(model, model_path)

    def load(self, model_path):
        model = torch.load(model_path)
        model.eval()

        return model
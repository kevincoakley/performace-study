import argparse, pathlib, sys

script_version = "2.0.0"


def parse_arguments(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        dest="model",
        help="Path to the model",
        default="",
        required=True,
    )

    parser.add_argument(
        "--dataset-name",
        dest="dataset_name",
        help="cifar10, cifar100, fashion_mnist",
        default="cifar10",
        choices=["cifar10", "cifar100", "cats_vs_dogs"],
        required=True,
    )

    parser.add_argument(
        "--save-predictions",
        dest="save_predictions",
        help="Save the predictions",
        action="store_true",
    )

    parser.add_argument(
        "--csv-file",
        dest="predictions_csv_file",
        help="Path to the csv file to save the predictions",
        default="",
        required=True,
    )

    parser.add_argument(
        "--seed-val", dest="seed_val", help="Set the seed value", type=int, default=1
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])

    model_path = args.model
    dataset_name = args.dataset_name
    save_predictions = args.save_predictions
    predictions_csv_file = args.predictions_csv_file
    seed_val = args.seed_val

    model_extension = pathlib.Path(model_path).suffix

    """
    ## Framework definition based on model extension
    """
    if model_extension == ".h5":
        from tensorflow_framework import Tensorflow

        framework = Tensorflow()

    elif model_extension == ".pth":
        from pytorch_framework import Pytorch

        framework = Pytorch()

    """
    ## Configure framework for fixed seed runs
    """
    framework.deterministic(seed_val)

    """
    ## Datasets definition dictionary
    """
    datasets = {
        "cifar100": {
            "train_path": "./cifar100/train/",
            "val_path": "./cifar100/test/",
            "num_classes": 100,
            "dataset_shape": (32, 32, 3),
            "training_shape": (128, 128, 3),
            "batch_size": 32,
        },
        "cifar10": {
            "train_path": "./cifar10/train/",
            "val_path": "./cifar10/test/",
            "num_classes": 10,
            "dataset_shape": (32, 32, 3),
            "training_shape": (128, 128, 3),
            "batch_size": 32,
        },
        "cats_vs_dogs": {
            "train_path": "./cats_vs_dogs/train/",
            "val_path": "./cats_vs_dogs/test/",
            "num_classes": 2,
            "dataset_shape": (128, 128, 3),
            "training_shape": (128, 128, 3),
            "batch_size": 32,
        },
    }

    train_path = datasets[dataset_name]["train_path"]
    val_path = datasets[dataset_name]["val_path"]
    dataset_shape = datasets[dataset_name]["dataset_shape"]
    training_shape = datasets[dataset_name]["training_shape"]
    num_classes = datasets[dataset_name]["num_classes"]
    batch_size = datasets[dataset_name]["batch_size"]

    """
    ## Load the dataset
    """
    # Always use the same random seed for the dataset
    train_dataset, val_dataset = framework.load_dataset(
        train_path, val_path, num_classes, batch_size, dataset_shape, training_shape, 42
    )

    """
    ## Load the model
    """
    model = framework.load(model_path)

    """
    ## Evaluate the model
    """
    score = framework.evaluate(
        model, val_dataset, save_predictions, predictions_csv_file
    )

    print("Test loss: ", score[0])
    print("Test accuracy: ", score[1])

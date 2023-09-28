import argparse, pathlib, sys
import image_classification

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
    ## Get the dataset details
    """
    dataset_detail = image_classification.get_dataset_details(dataset_name)

    """
    ## Load the dataset
    """
    # Always use the same random seed for the dataset
    train_dataset, val_dataset = framework.load_dataset(dataset_detail, 42)

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

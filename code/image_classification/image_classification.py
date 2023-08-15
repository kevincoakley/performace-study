import argparse, csv, math, os, random, sys, yaml
import numpy as np
from datetime import datetime

script_version = "2.0.0"


def image_classification(
    run_number,
    machine_learning_framework="TensorFlow",
    model_name="Densenet",
    dataset_name="cifar10",
    deterministic=False,
    epochs=50,
    learning_rate=4e-4,
    seed_val=1,
    run_name="",
    save_model=False,
    save_predictions=False,
):
    if machine_learning_framework == "TensorFlow":
        from tensorflow_framework import Tensorflow

        framework = Tensorflow()
    elif machine_learning_framework == "PyTorch":
        from pytorch_framework import Pytorch

        framework = Pytorch()

    if deterministic or seed_val != 1:
        """
        ## Configure framework for fixed seed runs
        """
        framework.deterministic(seed_val)

    """
    ## Datasets definition dictionary
    """
    datasets = {
        "cifar100": {
            "num_classes": 100,
            "input_shape": (128, 128, 3),
            "batch_size": 32,
        },
        "cifar10": {
            "num_classes": 10,
            "input_shape": (128, 128, 3),
            "batch_size": 32,
        },
        "fashion_mnist": {
            "num_classes": 10,
            "input_shape": (128, 128, 1),
            "batch_size": 32,
        },
        "cats_vs_dogs": {
            "num_classes": 2,
            "input_shape": (128, 128, 3),
            "batch_size": 32,
        },
    }

    input_shape = datasets[dataset_name]["input_shape"]
    num_classes = datasets[dataset_name]["num_classes"]
    batch_size = datasets[dataset_name]["batch_size"]

    """
    ## Load the dataset
    """
    # Always use the same random seed for the dataset
    train_dataset, val_dataset = framework.load_dataset(
        dataset_name, batch_size, input_shape, 42
    )

    """
    ## Create the model
    """
    model = framework.load_model(model_name, input_shape, num_classes)

    """
    ## Create the base name for the log and model files
    """
    base_name = os.path.basename(sys.argv[0]).split(".")[0]

    if len(run_name) >= 1:
        base_name = base_name + "_" + run_name

    """
    ## Train the model
    """
    # Time the training
    start_time = datetime.now()

    # Train the model
    trained_model = framework.train(
        model, train_dataset, val_dataset, epochs, learning_rate
    )

    # Calculate the training time
    end_time = datetime.now()
    training_time = end_time - start_time

    """
    ## Evaluate the trained model and save the predictions
    """
    predictions_csv_file = base_name + "_predictions_seed_" + str(seed_val) + ".csv"

    score = framework.evaluate(
        trained_model, val_dataset, save_predictions, predictions_csv_file
    )

    print("Test loss: ", score[0])
    print("Test accuracy: ", score[1])
    print("Training time: ", training_time)

    """
    ## Save the model
    """
    if save_model:
        if os.path.exists("models/") == False:
            os.mkdir("models/")

        model_path = "models/"

        if run_name != "":
            if os.path.exists("models/" + run_name + "/") == False:
                os.mkdir("models/" + run_name + "/")
            model_path = "models/" + run_name + "/"

        if deterministic or seed_val != 1:
            model_path = model_path + base_name + "_seed_" + str(seed_val)

        # Append the file extension based on the machine learning framework
        if machine_learning_framework == "TensorFlow":
            model_path = model_path + ".h5"
        elif machine_learning_framework == "PyTorch":
            model_path = model_path + ".pth"

        framework.save(model, model_path)

    return score[0], score[1], training_time


def get_system_info():
    if os.path.exists("system_info.py"):
        base_name = os.path.basename(sys.argv[0]).split(".")[0]

        import system_info

        sysinfo = system_info.get_system_info()

        with open("%s.yaml" % base_name, "w") as system_info_file:
            yaml.dump(sysinfo, system_info_file, default_flow_style=False)

        return sysinfo
    else:
        return None


def save_score(
    test_loss,
    test_accuracy,
    machine_learning_framework,
    epochs,
    learning_rate,
    training_time,
    model_name,
    dataset_name,
    deterministic,
    seed_val,
    run_name="",
):
    if machine_learning_framework == "TensorFlow":
        from tensorflow_framework import Tensorflow

        framework = Tensorflow()
    elif machine_learning_framework == "PyTorch":
        from pytorch_framework import Pytorch

        framework = Pytorch()

    csv_file = os.path.basename(sys.argv[0]).split(".")[0] + ".csv"
    write_header = False

    # If determistic is false and the seed value is 1 then the
    # seed value is totally random and we don't know what it is.
    if deterministic == False and seed_val == 1:
        seed_val = "random"

    if not os.path.isfile(csv_file):
        write_header = True

    with open(csv_file, "a") as csvfile:
        fieldnames = [
            "run_name",
            "script_version",
            "date_time",
            "fit_time",
            "python_version",
            "machine_learning_framework",
            "framework_version",
            "epochs",
            "learning_rate",
            "model_name",
            "dataset_name",
            "random_seed",
            "test_loss",
            "test_accuracy",
        ]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow(
            {
                "run_name": run_name,
                "script_version": script_version,
                "date_time": datetime.now(),
                "fit_time": training_time,
                "python_version": sys.version.replace("\n", ""),
                "machine_learning_framework": machine_learning_framework,
                "framework_version": framework.version,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "model_name": model_name,
                "dataset_name": dataset_name,
                "random_seed": seed_val,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )


def parse_arguments(args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        help="Run in deterministic mode",
        action="store_true",
    )

    parser.add_argument(
        "--random-seed-val",
        dest="random_seed_val",
        help="Pick a random int for the seed value every run and record it in the csv file",
        action="store_true",
    )

    parser.add_argument(
        "--seed-val", dest="seed_val", help="Set the seed value", type=int, default=1
    )

    parser.add_argument(
        "--num-runs",
        dest="num_runs",
        help="Number of training runs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--epochs",
        dest="epochs",
        help="Number of epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--learning-rate",
        dest="learning_rate",
        help="Set the learning rate",
        type=float,
        default=4e-4,
    )

    parser.add_argument(
        "--run-name",
        dest="run_name",
        help="Name of training run",
        default="",
    )

    parser.add_argument(
        "--save-model", dest="save_model", help="Save the model", action="store_true"
    )

    parser.add_argument(
        "--save-predictions",
        dest="save_predictions",
        help="Save the predictions",
        action="store_true",
    )

    parser.add_argument(
        "--ml-framework",
        dest="machine_learning_framework",
        help="Name of Machine Learning framework",
        default="TensorFlow",
        choices=[
            "TensorFlow",
            "PyTorch",
        ],
        required=True,
    )

    parser.add_argument(
        "--model-name",
        dest="model_name",
        help="Name of model to train",
        default="DenseNet",
        choices=[
            "EfficientNetB4",
            "InceptionV3",
            "ResNet50",
            "ResNet101",
            "ResNet152",
            "DenseNet121",
            "DenseNet169",
            "DenseNet201",
        ],
        required=True,
    )

    parser.add_argument(
        "--dataset-name",
        dest="dataset_name",
        help="cifar10, cifar100, fashion_mnist",
        default="cifar10",
        choices=["cifar10", "cifar100", "fashion_mnist", "cats_vs_dogs"],
        required=True,
    )

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])

    system_info = get_system_info()
    seed_val = args.seed_val

    for x in range(args.num_runs):
        if args.random_seed_val:
            seed_val = random.randint(0, 2**32 - 1)

        print(
            "\nImage Classification (%s - %s - %s): %s of %s [%s]\n======================\n"
            % (
                args.machine_learning_framework,
                args.model_name,
                args.dataset_name,
                str(x + 1),
                args.num_runs,
                seed_val,
            )
        )
        test_loss, test_accuracy, training_time = image_classification(
            x + 1,
            machine_learning_framework=args.machine_learning_framework,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            deterministic=args.deterministic,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed_val=seed_val,
            run_name=args.run_name,
            save_model=args.save_model,
            save_predictions=args.save_predictions,
        )
        save_score(
            test_loss=test_loss,
            test_accuracy=test_accuracy,
            machine_learning_framework=args.machine_learning_framework,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            training_time=training_time,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            deterministic=args.deterministic,
            seed_val=seed_val,
            run_name=args.run_name,
        )

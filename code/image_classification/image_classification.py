import argparse, csv, math, os, random, sys, yaml
import numpy as np
from datetime import datetime

script_version = "3.0.0"


def get_model_details(model_name):
    models = {
        "ResNet20": {
            "epochs": 200,
        },
        "ResNet32": {
            "epochs": 200,
        },
        "ResNet44": {
            "epochs": 200,
        },
        "ResNet56": {
            "epochs": 200,
        },
        "ResNet110": {
            "epochs": 200,
        },
        "ResNet1202": {
            "epochs": 200,
        },
    }

    return models[model_name]


def get_dataset_details(dataset_name):
    """
    ## Datasets definition dictionary
    """
    datasets = {
        "cifar100": {
            "train_path": "./cifar100/train/",
            "val_path": "./cifar100/val/",
            "test_path": "./cifar100/test/",
            "num_classes": 100,
            "dataset_shape": (32, 32, 3),
            "batch_size": 128,
            "normalization": {
                "mean": (0.5071, 0.4865, 0.4409),
                "std": (0.2673, 0.2564, 0.2762),
            },
        },
        "cifar10": {
            "train_path": "./cifar10/train/",
            "val_path": "./cifar10/val/",
            "test_path": "./cifar10/test/",
            "num_classes": 10,
            "dataset_shape": (32, 32, 3),
            "batch_size": 128,
            "normalization": {
                "mean": (0.4914, 0.4822, 0.4465),
                "std": (0.247, 0.2435, 0.2616),
            },
        },
        "cats_vs_dogs": {
            "train_path": "./cats_vs_dogs/train/",
            "val_path": "./cats_vs_dogs/test/",
            "num_classes": 2,
            "dataset_shape": (128, 128, 3),
            "batch_size": 128,
            # Normalization done on images resized to 128x128
            "normalization": {
                "mean": (0.4872, 0.4544, 0.4165),
                "std": (0.2622, 0.256, 0.2584),
            },
        },
    }

    return datasets[dataset_name]


def image_classification(
    run_number,
    machine_learning_framework="TensorFlow",
    model_name="ResNet20",
    dataset_name="cifar10",
    deterministic=False,
    epochs=200,
    seed_val=1,
    run_name="",
    save_model=False,
    save_predictions=False,
    save_epoch_logs=False,
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
    ## Get the dataset details
    """
    dataset_details = get_dataset_details(dataset_name)

    """
    ## Load the dataset
    """
    # Always use the same random seed for the dataset
    train_dataset, val_dataset, test_dataset = framework.load_dataset(
        dataset_details, 42
    )

    """
    ## Create the model
    """
    model = framework.load_model(model_name, dataset_details)

    """
    ## Create the base name for the log and model files
    """
    base_name = os.path.basename(sys.argv[0]).split(".")[0]

    if len(run_name) >= 1:
        base_name = run_name

    """
    ## Train the model
    """
    # Time the training
    start_time = datetime.now()

    if deterministic or seed_val != 1:
        csv_train_log_file = "%s_log_%s_%s.csv" % (
            base_name,
            machine_learning_framework,
            seed_val,
        )
    else:
        csv_train_log_file = "%s_log_%s_%s.csv" % (
            base_name,
            machine_learning_framework,
            run_number,
        )

    # Train the model
    trained_model = framework.train(
        model,
        train_dataset,
        val_dataset,
        epochs,
        save_epoch_logs,
        csv_train_log_file,
    )

    # Calculate the training time
    end_time = datetime.now()
    training_time = end_time - start_time

    """
    ## Reset the deterministic configuration before evaluating the model
    """
    if deterministic or seed_val != 1:
        """
        ## Configure framework for fixed seed runs
        """
        framework.deterministic(seed_val)

    """
    ## Evaluate the trained model and save the predictions
    """
    prediction_path = ""

    if save_predictions:
        if os.path.exists("predictions/") == False:
            os.mkdir("predictions/")

        prediction_path = "predictions/"

        if run_name != "":
            if os.path.exists("predictions/" + run_name + "/") == False:
                os.mkdir("predictions/" + run_name + "/")
            prediction_path = "predictions/" + run_name + "/"

    predictions_csv_file = (
        prediction_path + base_name + "_seed_" + str(seed_val) + ".csv"
    )

    score = framework.evaluate(
        trained_model, test_dataset, save_predictions, predictions_csv_file
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

        framework.save(trained_model, model_path)

    return score[0], score[1], training_time


def get_system_info(filename):
    if os.path.exists("system_info.py"):
        import system_info

        sysinfo = system_info.get_system_info()

        with open("%s.yaml" % filename, "w") as system_info_file:
            yaml.dump(sysinfo, system_info_file, default_flow_style=False)

        return sysinfo
    else:
        return None


def save_score(
    test_loss,
    test_accuracy,
    machine_learning_framework,
    epochs,
    training_time,
    model_name,
    dataset_name,
    deterministic,
    seed_val,
    filename,
    run_name="",
):
    if machine_learning_framework == "TensorFlow":
        from tensorflow_framework import Tensorflow

        framework = Tensorflow()
    elif machine_learning_framework == "PyTorch":
        from pytorch_framework import Pytorch

        framework = Pytorch()

    csv_file = filename + ".csv"
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
        default=0,
    )

    parser.add_argument(
        "--run-name",
        dest="run_name",
        help="Name of training run",
        default="",
    )

    parser.add_argument(
        "--save-filename",
        dest="save_filename",
        help="filename used to save the results",
        type=str,
        default=str(os.path.basename(sys.argv[0]).split(".")[0]),
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
        "--save-epoch-logs",
        dest="save_epoch_logs",
        help="Save the accuracy and loss logs for each epoch",
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
        default="ResNet20",
        choices=[
            "ResNet20",
            "ResNet32",
            "ResNet44",
            "ResNet56",
            "ResNet110",
            "ResNet1202",
        ],
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

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])

    save_filename = args.save_filename

    system_info = get_system_info(save_filename)
    seed_val = args.seed_val

    if args.epochs > 0:
        epochs = args.epochs
    else:
        epochs = get_model_details(args.model_name)["epochs"]

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
            epochs=epochs,
            seed_val=seed_val,
            run_name=args.run_name,
            save_model=args.save_model,
            save_predictions=args.save_predictions,
            save_epoch_logs=args.save_epoch_logs,
        )
        save_score(
            test_loss=test_loss,
            test_accuracy=test_accuracy,
            machine_learning_framework=args.machine_learning_framework,
            epochs=epochs,
            training_time=training_time,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            deterministic=args.deterministic,
            seed_val=seed_val,
            filename=save_filename,
            run_name=args.run_name,
        )

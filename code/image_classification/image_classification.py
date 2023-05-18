import argparse, csv, math, os, random, sys, yaml
import tensorflow as tf
import numpy as np
from datetime import datetime
import dataset_preprocess

script_version = "1.0.4"


def image_classification(
    run_number,
    model_name="Densenet",
    dataset_name="cifar10",
    deterministic=False,
    epochs=50,
    steps_per_epoch=392,
    learning_rate=4e-4,
    lr_decay=45,
    seed_val=1,
    run_name="",
    save_model=False,
):
    if deterministic or seed_val != 1:
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

    """
    ## Datasets definition dictionary
    """
    datasets = {
        "cifar100": {
            "num_class": 100,
            "input_shape": (128, 128, 3),
            "batch_size": 128,
        },
        "cifar10": {
            "num_class": 10,
            "input_shape": (128, 128, 3),
            "batch_size": 128,
        },
        "fashion_mnist": {
            "num_class": 10,
            "input_shape": (128, 128, 1),
            "batch_size": 128,
        },
        "cats_vs_dogs": {
            "num_class": 2,
            "input_shape": (128, 128, 3),
            "batch_size": 128,
        },
    }

    input_shape = datasets[dataset_name]["input_shape"]
    num_class = datasets[dataset_name]["num_class"]
    batch_size = datasets[dataset_name]["batch_size"]

    """
    ## Models definition dictionary
    """
    models = {
        "EfficientNet": {
            "application": tf.keras.applications.EfficientNetB4,
            "args": {
                "include_top": True,
                "weights": None,
                "input_shape": input_shape,
                "classes": num_class,
                "classifier_activation": "softmax",
            },
        },
        "Xception": {
            "application": tf.keras.applications.Xception,
            "args": {
                "include_top": True,
                "weights": None,
                "input_shape": input_shape,
                "classes": num_class,
                "classifier_activation": "softmax",
            },
        },
        "InceptionV3": {
            "application": tf.keras.applications.InceptionV3,
            "args": {
                "include_top": True,
                "weights": None,
                "input_shape": input_shape,
                "classes": num_class,
                "classifier_activation": "softmax",
            },
        },
        "ResNet50V2": {
            "application": tf.keras.applications.resnet_v2.ResNet50V2,
            "args": {
                "include_top": True,
                "weights": None,
                "input_shape": input_shape,
                "classes": num_class,
                "classifier_activation": "softmax",
            },
        },
        "ResNet101V2": {
            "application": tf.keras.applications.resnet_v2.ResNet101V2,
            "args": {
                "include_top": True,
                "weights": None,
                "input_shape": input_shape,
                "classes": num_class,
                "classifier_activation": "softmax",
            },
        },
        "ResNet152V2": {
            "application": tf.keras.applications.resnet_v2.ResNet152V2,
            "args": {
                "include_top": True,
                "weights": None,
                "input_shape": input_shape,
                "classes": num_class,
                "classifier_activation": "softmax",
            },
        },
        "DenseNet121": {
            "application": tf.keras.applications.DenseNet121,
            "args": {
                "include_top": True,
                "weights": None,
                "input_shape": input_shape,
                "classes": num_class,
                "classifier_activation": "softmax",
            },
        },
        "DenseNet169": {
            "application": tf.keras.applications.DenseNet169,
            "args": {
                "include_top": True,
                "weights": None,
                "input_shape": input_shape,
                "classes": num_class,
                "classifier_activation": "softmax",
            },
        },
        "DenseNet201": {
            "application": tf.keras.applications.DenseNet201,
            "args": {
                "include_top": True,
                "weights": None,
                "input_shape": input_shape,
                "classes": num_class,
                "classifier_activation": "softmax",
            },
        },
    }

    """
    ## Load the dataset
    """
    train_dataset, val_dataset = dataset_preprocess.get_dataset(
        dataset_name, batch_size, shuffle_seed=42, shape=input_shape
    )

    """
    ## Create the model
    """
    model = tf.keras.Sequential()
    model.add(models[model_name]["application"](**models[model_name]["args"]))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Print the model summary
    model.summary()

    """
    ## Define the learning rate scheduler to decrease the learning rate by 10x every 45 epochs
    """

    def lr_scheduler(epoch):
        new_lr = learning_rate * (0.1 ** (epoch // (lr_decay - 1)))
        return new_lr

    reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    """
    ## Create the base name for the log and model files
    """
    base_name = os.path.basename(sys.argv[0]).split(".")[0]

    if len(run_name) >= 1:
        base_name = base_name + "_" + run_name

    """
    ## Define csv logger callback
    """
    csv_train_log_file = base_name + "_log_" + str(run_number) + ".csv"

    csv_logger = tf.keras.callbacks.CSVLogger(csv_train_log_file)

    # Define callbacks
    callbacks = [reduce_lr, csv_logger]

    # Time the training
    start_time = datetime.now()

    # Train the model
    model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        callbacks=callbacks,
    )

    # Calculate the training time
    end_time = datetime.now()
    training_time = end_time - start_time

    """
    ## Evaluate the trained model
    """
    score = model.evaluate(val_dataset, steps=steps_per_epoch, verbose=0)

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
            if os.path.exists("models/") == False:
                os.mkdir("models/" + run_name + "/")
            model_path = "models/" + run_name + "/"

        if deterministic or seed_val != 1:
            model_path = model_path + base_name + "_seed_" + str(seed_val) + ".h5"

        model.save(model_path)

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
    epochs,
    steps_per_epoch,
    learning_rate,
    lr_decay,
    training_time,
    model_name,
    dataset_name,
    deterministic,
    seed_val,
    run_name="",
):
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
            "tensorflow_version",
            "tensorflow_compiler_version",
            "epochs",
            "steps_per_epoch",
            "learning_rate",
            "lr_decay",
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
                "tensorflow_version": tf.version.VERSION,
                "tensorflow_compiler_version": tf.version.COMPILER_VERSION,
                "epochs": epochs,
                "steps_per_epoch": steps_per_epoch,
                "learning_rate": learning_rate,
                "lr_decay": lr_decay,
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
        "--steps-per-epoch",
        dest="steps_per_epoch",
        help="Total number of batches of samples per epoch",
        type=int,
        default=392,
    )

    parser.add_argument(
        "--learning-rate",
        dest="learning_rate",
        help="Set the learning rate",
        type=float,
        default=4e-4,
    )

    parser.add_argument(
        "--lr-decay",
        dest="lr_decay",
        help="Number learning rate decay epochs",
        type=int,
        default=45,
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
        "--model-name",
        dest="model_name",
        help="Name of model to train",
        default="DenseNet",
    )

    parser.add_argument(
        "--dataset-name",
        dest="dataset_name",
        help="cifar10, cifar100, fashion_mnist",
        default="cifar10",
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
            "\nImage Classification (%s - %s): %s of %s [%s]\n======================\n"
            % (args.model_name, args.dataset_name, str(x + 1), args.num_runs, seed_val)
        )
        test_loss, test_accuracy, training_time = image_classification(
            x + 1,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            deterministic=args.deterministic,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            learning_rate=args.learning_rate,
            lr_decay=args.lr_decay,
            seed_val=seed_val,
            run_name=args.run_name,
            save_model=args.save_model,
        )
        save_score(
            test_loss=test_loss,
            test_accuracy=test_accuracy,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            learning_rate=args.learning_rate,
            lr_decay=args.lr_decay,
            training_time=training_time,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            deterministic=args.deterministic,
            seed_val=seed_val,
            run_name=args.run_name,
        )

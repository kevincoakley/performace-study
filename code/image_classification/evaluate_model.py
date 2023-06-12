import argparse
import tensorflow as tf
import numpy as np
from sklearn.metrics import top_k_accuracy_score
import dataset_preprocess


script_version = "1.0.0"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    dest="model",
    help="Path to the model",
    default="",
    required=True,
)

parser.add_argument(
    "--dataset",
    dest="dataset",
    help="Dataset to use",
    default="cifar10",
    choices=["cifar10", "cifar100", "fashion_mnist", "cats_vs_dogs"],
    required=True,
)

parser.add_argument(
    "--seed",
    dest="seed",
    help="Set the seed for the random number generator",
    type=int,
    default=42,
)

parser.add_argument(
    "--batch-size",
    dest="batch_size",
    help="Set the batch size",
    type=int,
    default=128,
)

args = parser.parse_args()
model_path = args.model
dataset = args.dataset
seed_val = args.seed
batch_size = args.batch_size

# Load the model
new_model = tf.keras.models.load_model(model_path)

# Show the model architecture
new_model.summary()

# Read the dataset
train_dataset, val_dataset, train_size, val_size = dataset_preprocess.get_dataset(
    dataset, batch_size, shuffle_seed=42, shape=(128, 128, 3)
)

# Get the predictions
predictions = new_model.predict(val_dataset)

# Get the labels of the validation dataset
val_dataset = val_dataset.unbatch()
labels = np.asarray(list(val_dataset.map(lambda x, y: y)))

# Get the index to the highest probability
y_true = np.argmax(labels, axis=1)
y_pred = np.argmax(predictions, axis=1)

# Calculate the top 1 and top 5 accuracy
top_1_score = top_k_accuracy_score(y_true, predictions, k=1)
top_5_score = top_k_accuracy_score(y_true, predictions, k=5)

print("Top 1 Accuracy: ", top_1_score)
print("Top 5 Accuracy: ", top_5_score)

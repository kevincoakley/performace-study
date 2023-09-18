import argparse, os
import torch
import torchvision

dataset_names = ("cifar10", "cifar100", "mnist")

parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset",
    dest="dataset",
    help="cifar10, cifar100, fashion_mnist",
    default="cifar10",
    choices=["cifar10", "cifar100", "cats_vs_dogs"],
    required=True,
)

args = parser.parse_args()
print(args.dataset)

dataset_path = os.path.join(".", args.dataset, "train")

train_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
    ]
)

train = torchvision.datasets.ImageFolder(root=dataset_path, transform=train_transform)

train_loader = torch.utils.data.DataLoader(
    train, batch_size=len(train), shuffle=False, num_workers=1
)

imgs = None
for batch in train_loader:
    image_batch = batch[0]
    if imgs is None:
        imgs = image_batch.cpu()
    else:
        imgs = torch.cat([imgs, image_batch.cpu()], dim=0)
imgs = imgs.numpy()

# calculate mean over each channel (r,g,b)
mean_r = round(imgs[:, 0, :, :].mean(), 4)
mean_g = round(imgs[:, 1, :, :].mean(), 4)
mean_b = round(imgs[:, 2, :, :].mean(), 4)
print("mean: %s, %s, %s" % (mean_r, mean_g, mean_b))

# calculate std over each channel (r,g,b)
std_r = round(imgs[:, 0, :, :].std(), 4)
std_g = round(imgs[:, 1, :, :].std(), 4)
std_b = round(imgs[:, 2, :, :].std(), 4)
print("std: %s, %s, %s" % (std_r, std_g, std_b))

from __future__ import division, print_function

import os
import traceback
from collections import Counter

import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import seaborn as sns
import torch
import torch.backends.cudnn as cudnn
import torchnet.meter.confusionmeter as cm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

cudnn.benchmark = True
torch.manual_seed(0)
plt.ion()  # interactive mode
plt.style.use("ggplot")
sns.set_theme(style="white")


def preprocessing(data_dir, batch_size):
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"), data_transforms["train"]
    )
    train_data, val_data = train_test_split(
        train_dataset, test_size=0.2, random_state=42
    )

    image_datasets = {
        "train": train_data,
        "val": val_data,
        "test": datasets.ImageFolder(
            os.path.join(data_dir, "test"), data_transforms["test"]
        ),
    }

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4
        )
        for x in ["train", "val", "test"]
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val", "test"]}

    class_names = ["Carton", "Métal", "Plastique", "Verre"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print dataset sizes
    print("Train dataset size:", dataset_sizes["train"])
    print("Validation dataset size:", dataset_sizes["val"])
    print("Test dataset size:", dataset_sizes["test"])
    print("Class: ", class_names)

    return dataloaders, device, dataset_sizes, class_names


def compute_labels(dataloaders, device):
    ls_labels = []
    for inputs, labels in dataloaders:
        labels = labels.to(device)
        ls_labels = [*ls_labels, *labels.tolist()]
    return ls_labels


def compute_weights(labels):
    n_samples = len(labels)
    label_counts = Counter(labels)
    class_weights = [
        n_samples / (label_counts[i] * len(label_counts))
        for i in range(len(label_counts))
    ]
    return class_weights

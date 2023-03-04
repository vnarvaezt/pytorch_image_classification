import os

import pandas as pd
import torch

from src.evaluate_model import evaluate_model
from src.model import feature_extractor, freezed_feature_extractor
from src.preprocessing import preprocessing
from src.tools import plot_model, visualize_wrong_labels

ROOT_DIR = os.path.abspath(os.curdir)
output_path_fe = os.path.join(ROOT_DIR, "output/feature_extraction")
output_path_w = os.path.join(ROOT_DIR, "output/freezed_weights")
isExist = os.path.exists(output_path_fe)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(output_path_fe)

data_dir = "data/garbage_classification"
batch_size = 100
num_epochs = 1
do_train = False

dataloaders, device, dataset_sizes, class_names = preprocessing(data_dir, batch_size)

if do_train:
    # train and save model
    model_ft = feature_extractor(device, dataloaders, dataset_sizes, num_epochs)
    torch.save(
        model_ft,
        "output/feature_extraction/tf_feature_extraction_weights_wd001_es_bis.pt",
    )

    model_freeze = freezed_feature_extractor(
        device, dataloaders, dataset_sizes, num_epochs
    )
    torch.save(model_freeze, "output/freezed_weights/freezed_network_gamma_01.pt")
else:
    model_ft = torch.load(
        "output/feature_extraction/tf_feature_extraction_weights_wd001_es_bis.pt"
    )
    model_freeze = torch.load("output/freezed_weights/freezed_network_gamma_01.pt")

# evaluate
df = pd.read_csv("output/summary_with_regularization_early_stopping_bis.csv", sep=";")
plot_model(
    df["f1_train_score_epoch"],
    df["f1_val_score_epoch"],
    len(df["f1_train_score_epoch"]),
    "F1-score",
    " ",
)
plot_model(df["train_losses"], df["val_losses"], len(df["train_losses"]), "Loss", " ")

# statistics for trained models
evaluate_model(dataloaders["train"], model_ft, device, class_names)
evaluate_model(dataloaders["test"], model_ft, device, class_names)
evaluate_model(dataloaders["train"], model_freeze, device, class_names)
evaluate_model(dataloaders["test"], model_freeze, device, class_names)

# visualize wrong labels
visualize_wrong_labels(model_ft, device, dataloaders["test"], class_names)
visualize_wrong_labels(model_freeze, device, dataloaders["test"], class_names)

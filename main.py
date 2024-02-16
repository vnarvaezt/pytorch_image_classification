import argparse

import torch

from config import batch_size, data_dir, num_epochs
from src.evaluate_model import evaluate_model
from src.model import feature_extractor, freezed_feature_extractor
from src.preprocessing import preprocessing
from src.tools import plot_model, visualize_wrong_labels


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        type=bool,
        help=f"Possible values --train or --no-train",
        action=argparse.BooleanOptionalAction,
        required=True,
        default=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = vars(parse_args())
    do_train = arguments["train"]
    print(do_train)
    dataloaders, device, dataset_sizes, class_names = preprocessing(
        data_dir, batch_size
    )

    if do_train:
        print(">>>> Training models")
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
        print(">>>> Load models")
        model_ft = torch.load(
            "output/feature_extraction/tf_feature_extraction_weights_wd001_es_bis.pt"
        )
        model_freeze = torch.load("output/freezed_weights/freezed_network_gamma_01.pt")

    # evaluate
    plot_model(
        "output/summary_with_regularization_early_stopping_bis.csv",
        "output/feature_extraction/loss_model1",
    )
    plot_model(
        "output/summary_freezed_network_gamma_01.csv",
        "output/freezed_weights/loss_model2",
    )

    # statistics for trained models
    evaluate_model(
        "Model 1",
        dataloaders,
        model_ft,
        device,
        class_names,
        "output/feature_extraction/mx_model1",
    )
    evaluate_model(
        "Model 2",
        dataloaders,
        model_freeze,
        device,
        class_names,
        "output/freezed_weights/mx_model2",
    )

    # visualize wrong labels
    visualize_wrong_labels(
        "Model 1: Wrong predictions",
        "output/feature_extraction/wronglabels_model1",
        model_ft,
        device,
        dataloaders["test"],
        class_names,
    )
    visualize_wrong_labels(
        "Model 2: Wrong predictions",
        "output/freezed_weights/wronglabels_model2",
        model_freeze,
        device,
        dataloaders["test"],
        class_names,
    )

import copy
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.metrics import f1_score
from torch.optim import lr_scheduler
from torchvision import models

from src.preprocessing import compute_labels, compute_weights


def train_model(
    dataloaders,
    device,
    dataset_sizes,
    model,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    alias,
):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    # accuracy
    # best_acc = 0.0
    # F1 score
    best_F1_score = 0.0
    f1_train_score_epoch = []
    f1_val_score_epoch = []
    # keeping-track-of-losses
    train_losses = []
    valid_losses = []
    # stop training if loss doesnt improve for 5 consecutive epochs
    early_stopping_epochs = 5
    # Keep track of the number of epochs without improvement
    no_improvement_epochs = 0
    flag = False
    # start training
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            # set training and validation loss equal to 0
            running_loss = 0.0
            running_f1 = 0.0
            total_f1 = 0.0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # input to the f1 score must be array, not tensor
                f1 = f1_score(
                    labels.cpu().numpy(), preds.cpu().numpy(), average="macro"
                )
                # weight score by sample size in the batch
                running_f1 += f1 * inputs.size(0)
                total_f1 += f1
            # the learning rate is only updated during the training phase
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_f1 = running_f1 / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.2f} F1 score: {epoch_f1:.2f}")
            if phase == "train":
                f1_train_score_epoch.append(epoch_f1)
                train_losses.append(epoch_loss)
            else:
                f1_val_score_epoch.append(epoch_f1)
                valid_losses.append(epoch_loss)

            # deep copy the model
            if phase == "val" and epoch_f1 > best_F1_score:
                best_F1_score = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
                no_improvement_epochs = 0
            elif phase == "val" and epoch_f1 <= best_F1_score:
                no_improvement_epochs += 1
            else:
                print()

            # If the performance has not improved for early_stopping_epochs, stop training
            if no_improvement_epochs >= early_stopping_epochs:
                print(no_improvement_epochs)
                print(early_stopping_epochs)
                print("Early stopping after epoch", epoch)
                flag = True
                break
        if flag:
            break
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best F1 score: {best_F1_score:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    # store loss and F1 score
    data = {
        "train_losses": train_losses,
        "val_losses": valid_losses,
        "f1_train_score_epoch": f1_train_score_epoch,
        "f1_val_score_epoch": f1_val_score_epoch,
    }
    df_summary = pd.DataFrame(data)
    df_summary.to_csv(f"output/summary_{alias}.csv", sep=";")

    return model


def feature_extractor(device, dataloaders, dataset_sizes, num_epochs):
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, 4)

    # dropout layer
    model_ft.fc = nn.Sequential(nn.Dropout(p=0.5), model_ft.fc)
    model_ft = model_ft.to(device)

    # calculate class weights based on frequency of classes in training dataset
    ls_labels = compute_labels(dataloaders["train"], device)
    class_weights = torch.tensor(compute_weights(ls_labels), dtype=torch.float32)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # set optimizer
    optimizer_ft = optim.SGD(
        model_ft.parameters(), lr=0.001, momentum=0.9, weight_decay=0.01
    )
    # decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.01)

    # train the model
    model_ft = train_model(
        dataloaders,
        device,
        dataset_sizes,
        model_ft,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        num_epochs,
        alias="with_regularization_early_stopping_bis",
    )

    return model_ft


def freezed_feature_extractor(device, dataloaders, dataset_sizes, num_epochs):

    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Sequential(
        nn.Linear(num_ftrs, 300),
        nn.ReLU(),
        nn.BatchNorm1d(300),
        nn.Linear(300, 200),
        nn.ReLU(),
        nn.BatchNorm1d(200),
        nn.Linear(200, 100),
        nn.ReLU(),
        nn.BatchNorm1d(100),
        nn.Linear(100, 4),
    )

    # dropout layer
    model_conv.fc = nn.Sequential(nn.Dropout(p=0.2), model_conv.fc)
    model_conv = model_conv.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_conv = optim.SGD(
        model_conv.fc.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001
    )

    # Decay LR by a factor of 0.1 every 7 epochs, adjusts learning rate
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.01)

    # train model
    model_conv = train_model(
        dataloaders,
        device,
        dataset_sizes,
        model_conv,
        criterion,
        optimizer_conv,
        exp_lr_scheduler,
        num_epochs,
        alias="freezed_network_gamma_01",
    )

    return model_conv

import pandas as pd
import seaborn as sns
import torch
import torchnet.meter.confusionmeter as cm
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

plt.style.use("ggplot")

def make_prediction(dataloader, device, model):
    with torch.no_grad():
        y_true = []
        y_pred = []
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
    return y_true, y_pred

def compute_overall_metrics(y_true, y_pred):
    # macro option is used here bcs we care about each class equally
    model_f1 = f1_score(y_true, y_pred, average="macro")
    model_recall = recall_score(y_true, y_pred, average="macro")
    model_precision = precision_score(y_true, y_pred, average="macro")
    print("Overall F1 score of the network is: %d %%" % (100 * model_f1))
    print("Overall weighted recall score is: %d %%" % (100 * model_recall))
    print("Overall weighted precison score is: %d %%" % (100 * model_precision))

def compute_class_wise_stats(dataloader, device, model, class_names):
    # Class wise statistics
    FP = list(0.0 for i in range(4))
    TP_FN = list(0.0 for i in range(4))
    TP = list(0.0 for i in range(4))
    try:
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                # true labels
                labels = labels.to(device)
                # predicted labels
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                # keep only if predicted classe equals true class
                point = (predicted == labels).squeeze()
                for j in range(len(labels)):
                    label = labels[j]
                    # keep track of the true positifs
                    if len(point) > 0:
                        TP[label] += point[j].item()
                    else:
                        TP[label] += point
                    # keep track of all the true labels for a class
                    TP_FN[label] += 1
                    if predicted[j] != label:
                        FP[predicted[j]] += 1
    except Exception as e:
        print(e)
    # compute recall and precision per class
    recall_all = []
    precision_all = []
    for i in range(4):
        recall = TP[i] / TP_FN[i]
        precision = TP[i] / (TP[i] + FP[i])
        print(
            "%5s ==> Recall = %2d %%, Precision = %2d %%"
            % (class_names[i], 100 * recall, 100 * precision)
        )
        recall_all.append(recall)
        # print("Precision of %5s : %2d %%" % (class_names[i], 100 * precision))
        precision_all.append(precision)
    return recall_all, precision_all

def compute_confusion_mtx(dataloader, device, model, class_names):
    # Get the confusion matrix for testing data
    confusion_matrix = cm.ConfusionMeter(4)
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            confusion_matrix.add(predicted, labels)

    # Confusion matrix as a heatmap
    con_m = confusion_matrix.conf
    df_conf_matrix = pd.DataFrame(
        con_m, index=[i for i in class_names], columns=[i for i in class_names]
    )
    return df_conf_matrix

def evaluate_model(title, dataloaders, model, device, class_names, save_path):
    print("===> train")
    y_true_train, y_pred_train = make_prediction(dataloaders["train"], device, model)
    compute_overall_metrics(y_true_train, y_pred_train)
    compute_class_wise_stats(dataloaders["train"], device, model, class_names)
    print("\n\n===> test")
    y_true_test, y_pred_test = make_prediction(dataloaders["test"], device, model)
    compute_overall_metrics(y_true_test, y_pred_test)
    compute_class_wise_stats(dataloaders["test"], device, model, class_names)

    df_conf_matrix_train = compute_confusion_mtx(
        dataloaders["train"], device, model, class_names
    )
    df_conf_matrix_test = compute_confusion_mtx(
        dataloaders["test"], device, model, class_names
    )

    f, axes = plt.subplots(1, 2, figsize=(20, 5), sharey="row")

    sns.heatmap(
        df_conf_matrix_train,
        annot=True,
        fmt="g",
        annot_kws={"size": 10},
        cbar=False,
        cmap="Greens",
        ax=axes[0],
    )
    sns.heatmap(
        df_conf_matrix_test,
        annot=True,
        fmt="g",
        annot_kws={"size": 10},
        cbar=False,
        cmap="Greens",
        ax=axes[1],
    )
    axes[0].set(ylabel="True label", title="Train")
    axes[1].set(title="Test")
    f.text(0.5, 0.04, "epoch", va="center", ha="center")
    plt.suptitle(title)
    plt.show()
    plt.savefig(f"{save_path}.png")

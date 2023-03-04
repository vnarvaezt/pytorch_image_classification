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

        print("Recall of %5s : %2d %%" % (class_names[i], 100 * recall))
        recall_all.append(recall)
        print("Precision of %5s : %2d %%" % (class_names[i], 100 * precision))
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
    df_con_m = pd.DataFrame(
        con_m, index=[i for i in class_names], columns=[i for i in class_names]
    )
    sns.set(font_scale=1.5)
    sns.heatmap(
        df_con_m, annot=True, fmt="g", annot_kws={"size": 10}, cbar=False, cmap="Greens"
    )
    plt.xlabel("Predicted values")
    plt.ylabel("True labels")
    plt.show()


def evaluate_model(dataloaders, model, device, class_names):
    train_true, train_pred = make_prediction(dataloaders, device, model)
    compute_overall_metrics(train_true, train_pred)
    compute_class_wise_stats(dataloaders, device, model, class_names)
    compute_confusion_mtx(dataloaders, device, model, class_names)

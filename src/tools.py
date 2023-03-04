from __future__ import division, print_function

import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow

plt.style.use("ggplot")
torch.manual_seed(0)
cudnn.benchmark = True
plt.ion()  # interactive mode
sns.set_theme(style="white")


def imshow(inp, title=None, normalize=True):
    inp = inp.numpy().transpose((1, 2, 0))
    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def visualize_model(model, dataloaders, class_names, device, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders):
            inputs = inputs.to(device)
            labels = labels.to(device)
            ls_labels = labels.tolist()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis("off")
                ax.set_title(
                    f"predicted: {class_names[preds[j]]}, true label: {class_names[ls_labels[j]]}"
                )
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def visualize_wrong_labels(model, device, dataloaders, class_names, num_images=8):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 15))

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                if labels[j].item() != preds[j].item():
                    images_so_far += 1
                    if images_so_far > num_images:
                        model.train(mode=was_training)
                        return
                    ax = plt.subplot(4, 4, images_so_far)
                    ax.axis("off")
                    ax.set_title(
                        f"True: {class_names[labels[j].item()]} / Predicted: {class_names[preds[j].item()]}",
                        fontsize=9,
                    )
                    # imshow(inputs.cpu().data[j], normalize=True)
                    inp = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    inp = std * inp + mean
                    inp = np.clip(inp, 0, 1)
                    plt.imshow(inp)
        model.train(mode=was_training)


def plot_model(train_loss, val_loss, title, y_label):
    plt.style.use("ggplot")
    sns.lineplot(x=np.arange(len(train_loss)), y=train_loss, label="train")
    sns.lineplot(x=np.arange(len(val_loss)), y=val_loss, label="validation")
    plt.xlabel("epoch")
    plt.ylabel(y_label)
    plt.title(title)
    plt.xlim(0, len(train_loss))
    plt.grid(False)
    plt.show()

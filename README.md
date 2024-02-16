# Tranfer Learning for Garbage Classification

## General info
This project classifies images using transfer learning. The pre-trained model used is ResNet-18.
2 scenarios were tested:
1. Pre-trained model: In this scenario, a model is trained on a large dataset or task,
 and then the model's weights are fine-tuned on a smaller, related dataset. 
 A pre-trained image classification model like ResNet-18 is fine-tuned on a smaller
  dataset of images related to a specific domain of garbage classification.
  
2. Domain adaptation: In this scenario, we assume that the garbage classification domain
is different than the pre-trained model. The goal is to adapt the model's knowledge from the source domain to improve performance on the target domain. This involved retraining the last layers of the model, and adding additional layers.

Without surprise, the 1st method is faster and shows better results than the 2nd.


## Technologies
Project is created with:
* torch 1.13
* torchvision 0.14.1

## Run the program
From the terminal run :   
```
python main.py --train 
```
- --train / --no-train: allows to choose whether to train the model or just load it.

## Project structure:
- Config : sets parameter values such as number of epochs and batch number
- Preprocessing : builds data loaders
- Model : training using the aforementioned scenarios
- Evaluate model : computes graphs and scores
- Data : stores training and test data downloaded from [kaggle](https://www.kaggle.com/datasets/ionutandreivaduva/garbage-classification)
- Output : stores images, models and summary datasets
- notebooks : data exploration and a draft for the modeling process


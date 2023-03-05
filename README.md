# Tranfer Learning for Garbage Classification

## General info
This project classifies images using transfer learning. The pre-trained model use is ResNet-18.
The main scenarios for trarnsfer learning are:
1. Pre-trained models: In this scenario, a model is trained on a large dataset or task,
 and then the model's weights are fine-tuned on a smaller, related dataset or task. 
 A pre-trained image classification model like ResNet-18 is fine-tuned on a smaller
  dataset of images related to a specific domain of garbage classification.
  
2. Domain adaptation: In this scenario, we assume that the garbage classification domain
is different than the domain of the pre-trained model, and the goal is to adapt the model's knowledge from the source domain to improve performance
 on the target domain. This involved retraining the last layers of the model, and adding additional layers
 to better capture the features of the target domain.


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


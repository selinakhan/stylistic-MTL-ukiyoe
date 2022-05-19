import torch
import wandb
import json
import numpy as np
from tqdm import tqdm
from data_utils import DataProcessor
from model import MultiTaskHS, SingleTaskRegression, SingleTaskClassification


def accuracy_MTL(model, val_data, device, mode, model_config):
    ''' Calculate accuracy for multi-task learning model'''

    total = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0

    regression_threshold = 20

    for i, data in enumerate(val_data):

        # Retrieve labels
        image, label1, label2, label3 = data

        # Send tensors to device
        image = image.to(device)

        # Retrieve model outputs
        with torch.no_grad():
            out1, out2, out3, _ = model(image, model_config)

        # Make prediction
        pred1 = torch.argmax(out1.cpu().data, 1)
        pred2 = out2.cpu().squeeze(1)
        pred3 = torch.argmax(out3.cpu().data, 1)

        correct1 += (pred1 == label1).sum().item()
        correct2 += (abs(pred2 - label2) < regression_threshold).sum().item()
        correct3 += (pred3 == label3).sum().item()

        # Every image has label 1
        total += label1.size(0) 

        if i == 6:
                if mode == 'train':
                    print('calculating training accracy...')
                    a1, a2, a3 = (100 * (correct1 / total)), (100 * (correct2 / total)), (100 * (correct3 / total))
                    acc = (a1 + a2 + a3) / 3
                    return a1, a2, a3, acc
        
    a1, a2, a3 = (100 * (correct1 / total)), (100 * (correct2 / total)), (100 * (correct3 / total))
    acc = (a1 + a2 + a3) / 3
    return a1, a2, a3, acc

def accuracy_MTL_supp(model, val_data, device, mode):
    ''' Calculate accuracy for multi-task learning model'''

    total = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    correct4 = 0

    regression_threshold = 20

    for i, data in enumerate(val_data):

        # Retrieve labels
        image, label1, label2, label3 = data

        # Send tensors to device
        image = image.to(device)

        # Retrieve model outputs
        with torch.no_grad():
            out1, out2, out3 = model(image)

        # Make prediction
        pred1 = torch.argmax(out1.cpu().data, 1)
        pred2 = out2.cpu().squeeze(1)
        pred3 = torch.argmax(out3.cpu().data, 1)

        correct1 += (pred1 == label1).sum().item()
        correct2 += (abs(pred2 - label2) < regression_threshold).sum().item()
        correct3 += (pred3 == label3).sum().item()

        # Every image has label 1
        total += label1.size(0) 

        if i == 6:
                if mode == 'train':
                    print('calculating training accracy...')
                    a1, a2, a3 = (100 * (correct1 / total)), (100 * (correct2 / total)), (100 * (correct3 / total))
                    acc = (a1 + a2 + a3) / 3
                    return a1, a2, a3, acc
        
    a1, a2, a3 = (100 * (correct1 / total)), (100 * (correct2 / total)), (100 * (correct3 / total))
    acc = (a1 + a2 + a3) / 3
    return a1, a2, a3, acc


def accuracy_STL_regression(model, val_data, device, mode, model_config):
    ''' Calculate accuracy for single-task regression model'''

    total = 0
    correct = 0

    regression_threshold = 20
    
    for i, data in enumerate(val_data):

        # Retrieve label
        image, _, label2, _ = data

        # Send  tensors to device
        image = image.to(device)

        with torch.no_grad():
            out, _ = model(image, model_config)

        pred = out.squeeze(1)

        correct += (abs(pred - label2) < regression_threshold).sum().item()
        total += label2.size(0) 

    
        if i == 6:
                if mode == 'train':
                    return 100 * (correct / total)

    return 100 * (correct / total)


def accuracy_STL_classification(model, val_data, task, device, mode, classes, model_config):
    ''' Calculate accuracy for single-task classification model'''

    total = 0
    correct = 0

    predictions, labels = [], []
    
    for i, data in enumerate(val_data):
        
        # Retrieve all inputs
        image, label1, _, label3 = data

        # Send tensors to device
        image = image.to(device)

        # Retrieve correct label for specified task
        if task == 1:
            label = label1.long()
        if task == 3:
            label = label3.long()

        with torch.no_grad():
            out, _ = model(image, model_config)

        # Make prediction
        pred = torch.argmax(out, dim=1).cpu()

        correct += (pred == label).sum().item()
        
        total += label.size(0)

        if task == 3:
            predictions.extend(pred.tolist())
            labels.extend(label.tolist())

        if i == 6:
            if mode == 'train':
                print('train acc')
                return 100 * (correct / total)

    if task == 3:
        wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,
                y_true=labels, preds=predictions,
                class_names=classes)})


    return 100 * (correct / total)

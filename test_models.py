import torch
import os
import argparse
import numpy as np
import json
import cv2
from tqdm import tqdm
from data_utils import DataProcessor
from model import MultiTaskHS, SingleTaskRegression, SingleTaskClassification
# from train import get_class_weights

from mtl_repo.utils.config import create_config
from mtl_repo.utils.common_config import get_train_dataset, get_transformations,\
                                get_val_dataset, get_train_dataloader, get_val_dataloader,\
                                get_optimizer, get_model, adjust_learning_rate,\
                                get_criterion
from mtl_repo.utils.logger import Logger
from mtl_repo.train.train_utils import train_vanilla
from mtl_repo.evaluation.evaluate_utils import eval_model, validate_results, save_model_predictions,\
                                    eval_all_results
from termcolor import colored


def conf_date(save_path, label2, pred2):
    f = open(f"{save_path}/2.txt", "a+")
    for a, b in zip(list(label2), list(pred2)):
        pair = (a.item(), int(b.item()))
        f.write(f'{pair}\n')


def conf_class(save_path, task, label, pred):
    f = open(f"{save_path}/{task}.txt", "a+")
    for a, b in zip(list(label), list(pred)):
        pair = (a.item(), b.item())
        f.write(f'{pair}\n')


def test_STL(path, task, test_loader, im_dir, divisor, save_path, test_classes):

    model = SingleTaskClassification(256, 512, test_classes[task], divisor, model_config='fr_vit')
    model = model.to(DEVICE)
    print('Model architecture dimensions correct')

    d = torch.load(path, map_location=torch.device('cpu'))
    d = {f'{k.replace("module.", "")}': v for k, v in d.items()}

    model.load_state_dict(d)
    print('Model successfully loaded')

    total = 0
    correct = 0

    predictions, labels = [], []
    
    for i, data in enumerate(tqdm(test_loader)):
        
        # Retrieve all inputs
        image, label1, _, label3 = data

        # Send tensors to device
        image = image.to(DEVICE)

        # Retrieve correct label for specified task
        if task == 0:
            label = label1.long()
        if task == 1:
            label = label3.long()

        with torch.no_grad():
            out, _ = model(image)
        out = out.cpu()

        # Make prediction
        pred = torch.argmax(out, dim=1)

        conf_class(save_path, task, label, pred)

        correct += (pred == label).sum().item()
        
        total += label.size(0)

    result = 100 * (correct / total)
    print(result)
    f = open(f"{save_path}/results.txt", "a+")
    f.write(f'Accuracy: {result}')
    return result

# test_STL('models/trained_models/STL_3/Skewed/model_3_classification_single_epoch_99_skewed_flowingsound.pt', 1, 'data/date_uniform/data_date_clean_train.csv', 'data/date_uniform/data_date_clean_test.csv', 'images/1', 256, 512, 4)


def test_STL_regression(path, test_loader, im_dir, save_path):

    model = SingleTaskRegression(256, 512)
    model = model.to(DEVICE)
    print('Model architecture dimensions correct')

    d = torch.load(path, map_location=torch.device(DEVICE))
    d = {f'{k.replace("module.", "")}': v for k, v in d.items()}

    model.load_state_dict(d)
    print('Model successfully loaded')

    total = 0
    correct = 0

    regression_threshold = 20
    
    for i, data in enumerate(tqdm(test_loader)):

        # Retrieve label
        image, _, label2, _, _ = data

        # Send  tensors to device
        image = image.to(DEVICE)

        with torch.no_grad():
            out, _ = model(image)
        out = out.cpu()

        pred = out.squeeze(1)

        conf_date(save_path, label2, pred)

        correct += (abs(pred - label2) < regression_threshold).sum().item()
        total += label2.size(0) 

    result = 100 * (correct / total)
    f = open(f"{save_path}/results.txt", "a+")
    f.write(f'Accuracy: {result}')
    return result


def test_MTL(path, test_loader, im_dir, save_path, train_classes):

    model = MultiTaskHS(256, 512, train_classes[0], train_classes[1], train_classes[2])
    model = model.to(DEVICE)
    print('Model architecture dimensions correct')

    d = torch.load(path, map_location=torch.device(DEVICE))
    d = {f'{k.replace("module.", "")}': v for k, v in d.items()}

    model.load_state_dict(d)
    print('Model successfully loaded')

    total = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0

    regression_threshold = 20

    for i, data in enumerate(tqdm(test_loader)):

        # Retrieve labels
        image, label1, label2, label3 = data

        # Send tensors to device
        image = image.to(DEVICE)

        # Retrieve model outputs
        with torch.no_grad():
            out1, out2, out3, _ = model(image)
        
        out1 = out1.cpu()
        out2 = out2.cpu()
        out3 = out3.cpu()
        # Make prediction
        pred1 = torch.argmax(out1.cpu().data, 1)
        pred2 = out2.cpu().squeeze(1)
        pred3 = torch.argmax(out3.cpu().data, 1)

        conf_date(save_path, label2, pred2)
        conf_class(save_path, 1, label1, pred1)
        conf_class(save_path, 3, label3, pred3)

        correct1 += (pred1 == label1).sum().item()
        correct2 += (abs(pred2 - label2) < regression_threshold).sum().item()
        correct3 += (pred3 == label3).sum().item()

        # Every image has label 1
        total += label1.size(0) 

    a1, a2, a3 = (100 * (correct1 / total)), (100 * (correct2 / total)), (100 * (correct3 / total))
    acc = (a1 + a2 + a3) / 4

    f = open(f"{save_path}/results.txt", "a+")
    f.write(f'Accuracy on artist: {a1}\n Accuracy on date: {a2}\n Accuracy on era {a3}')
    return a1, a2, a3, acc


def test_MTL_config(path, test_loader, im_dir, save_path, train_classes):

    cv2.setNumThreads(0)
    p = create_config('mtl_repo/configs/env.yml', 'mtl_repo/configs/nyud/resnet50/cross_stitch.yml')
    print(colored(p, 'red'))

    # Get model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    model = model.to(DEVICE)

    print('Model succesfully retrieved')

    d = torch.load(path, map_location=torch.device(DEVICE))
    d = {f'{k.replace("module.", "")}': v for k, v in d.items()}

    model.load_state_dict(d)
    print('Model successfully loaded')

    total = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0

    regression_threshold = 20

    for i, data in enumerate(tqdm(test_loader)):

        # Retrieve labels
        image, label1, label2, label3 = data
        label1 = label1.to(DEVICE)
        label2 = label2.to(DEVICE)
        label3 = label3.to(DEVICE)

        # Send tensors to device
        image = image.to(DEVICE)

        # Retrieve model outputs
        with torch.no_grad():
            out1, out2, out3 = model(image)

        # Make prediction
        pred1 = torch.argmax(out1.data, 1)
        pred2 = out2.squeeze(1)
        pred3 = torch.argmax(out3.data, 1)

        conf_date(save_path, label2, pred2)
        conf_class(save_path, 1, label1, pred1)
        conf_class(save_path, 3, label3, pred3)

        correct1 += (pred1 == label1).sum().item()
        correct2 += (abs(pred2 - label2) < regression_threshold).sum().item()
        correct3 += (pred3 == label3).sum().item()

        # Every image has label 1
        total += label1.size(0) 

    a1, a2, a3 = (100 * (correct1 / total)), (100 * (correct2 / total)), (100 * (correct3 / total))
    acc = (a1 + a2 + a3) / 4

    f = open(f"{save_path}/results.txt", "a+")
    f.write(f'Accuracy on artist: {a1}\n Accuracy on date: {a2}\n Accuracy on era {a3}')
    return a1, a2, a3, acc


# test_MTL('model_multi_epoch_299_date_era.pt', 'data/era/era_nonzero_train.csv', 'data/era/era_nonzero_validate.csv', 'images/1', 256, 512, 'models/trained_models/MTL/era_date/')


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Test a single/multitask model on a specified dataset.')

    parser.add_argument('--model', dest='model', required=True, help='Path to trained model')
    parser.add_argument('--train_data', dest='train_file', required=True, help='Path to training dataset')
    parser.add_argument('--val_data', dest='val_file', required=True, help='Path to validation dataset')
    parser.add_argument('--image_dir', dest='imdir', required=True, help='Path to images directory')
    parser.add_argument('--save_path', dest='save_path', required=False, help='Directory to store results')
    parser.add_argument('--task', dest='task', required=True, help='Task to train')
    parser.add_argument('--single_task', dest='single_task', type=int, required=False, help='Singletask model to train')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=2, required=False, help='Size of training batch')
    
    args = parser.parse_args()

    if args.task not in ['single', 'multi', 'mtl_conf']:
        raise ValueError(f'Invalid task specification {args.task}. Select from: single, multi')
    if not os.path.isfile(args.train_file):
        raise FileExistsError('Training dataset does not exist.')
    if not os.path.isfile(args.val_file):
        raise FileExistsError('Validation dataset does not exist.')
    if not os.path.isfile(args.model):
        raise FileExistsError('Model does not exist.')


    # Process data accordingly
    train_data = DataProcessor(args.train_file, 256, 'train', args.imdir)
    train_classes = train_data.get_labels()

    mapping = json.load(open('data/mapping.json', 'r'))

    json.dump(mapping, open(f'{args.save_path}/map.json', 'w+'))

    val_data = DataProcessor(args.val_file, 256, 'test', args.imdir, mappings=mapping)
    val_classes = val_data.get_labels()

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Select device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running on: {DEVICE}')


    #Train single task models
    if args.task == 'single':
        if args.single_task == 1:
            test_STL(args.model, 0, val_loader, args.imdir, 2, args.save_path, train_classes)

        if args.single_task == 2:
            test_STL_regression(args.model, val_loader, args.imdir, args.save_path)

        if args.single_task == 3:
            test_STL(args.model, 1, val_loader, args.imdir, 4, args.save_path, train_classes)

        if args.single_task == 4:
            test_STL(args.model, 2, val_loader, args.imdir, 2, args.save_path, train_classes)
    
    # Train multi task models
    if args.task == 'multi':
        test_MTL(args.model, val_loader, args.imdir, args.save_path, train_classes)

    if args.task == 'mtl_conf':
        test_MTL_config(args.model, val_loader, args.imdir, args.save_path, train_classes)
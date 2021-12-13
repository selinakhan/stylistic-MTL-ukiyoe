import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
import torch.nn as nn
import numpy as np
import wandb
import torch
import cv2
import math
import sys
import os
import argparse
from tqdm import tqdm, trange
from data_utils import DataProcessor
from evaluate import *
from model import *
import json


# MTL experiments
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

def train_MTL(train_data, val_data, model, epochs, weights, config, model_config):
    ''' Train multi-task learning model '''

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Send model to device
    model = model.to(DEVICE)

    loss_weights1 = torch.FloatTensor(weights[0]).to(DEVICE)
    loss_weights2 = torch.FloatTensor(weights[1]).to(DEVICE)

    criterion1 = nn.CrossEntropyLoss(weight=loss_weights1)
    criterion2 = nn.CrossEntropyLoss(weight=loss_weights2)
    criterion3 = nn.L1Loss()

    print(f'Dataloader length: {len(train_data)}')

    n_datapoints = 0

    for epoch in trange(epochs):
        for data in train_data:

            # Retrieve labels
            image, label1, label2, label3 = data

            # Send tensors to device
            image = image.to(DEVICE)
            label1 = label1.long().to(DEVICE)
            label2 = label2.float().unsqueeze(1).to(DEVICE)
            label3 = label3.long().to(DEVICE)

            out1, out2, out3, _ = model(image, model_config)
            
            loss1 = criterion1(out1, label1)
            loss2 = criterion3(out2, label2)
            loss3 = criterion2(out3, label3)

            loss = mtl_loss(config, loss1, loss2, loss3)

            wandb.log({'artist loss': loss1, 'date loss': loss2, 
                       'era loss': loss3, 'loss': loss, 
                       'n_datapoints': n_datapoints})

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Backward and optimize
            loss.backward()
            optimizer.step()

            n_datapoints += len(image)

        if epoch % 1 == 0:
            a1t, a2t, a3t, acct = accuracy_MTL(model, train_data, DEVICE, 'train', model_config)
            a1, a2, a3, acc = accuracy_MTL(model, val_data, DEVICE, 'test', model_config)

            wandb.log({'v_accuracy on artist': a1, 'v_accuracy on date': a2, 
                       'v_accuracy on era': a3, 'total v_accurac': acc, 
                       't_accuracy on artist': a1t, 't_accuracy on date': a2t, 
                       't_accuracy on era': a3t, 'total t_accurac': acct,
                       'loss': loss.item(), 'n_datapoints': n_datapoints, 'epoch': epoch})
            
        if epoch % 10 == 0:
            save_path = args.out_dir + f'/model_multi_epoch_{epoch}_{config}.pt'
            torch.save(model.state_dict(), save_path)
            
    torch.save(model.state_dict(), args.out_dir + f'/model_multi_epoch_{epoch}_{config}.pt')
    print('Finished Training')

def train_MTL_configs(train_data, val_data, epochs, weights, model, config, optimizer):
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # Send model to device
    model = model.to(DEVICE)

    # loss_weights1 = torch.FloatTensor(weights[0]).to(DEVICE)
    # loss_weights2 = torch.FloatTensor(weights[1]).to(DEVICE)

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    criterion3 = nn.L1Loss()

    print(f'Dataloader length: {len(train_data)}')

    n_datapoints = 0

    for epoch in trange(epochs):
        for data in train_data:

            # Retrieve labels
            image, label1, label2, label3 = data

            # Send tensors to device
            image = image.to(DEVICE)
            label1 = label1.long().to(DEVICE)
            label2 = label2.float().unsqueeze(1).to(DEVICE)
            label3 = label3.long().to(DEVICE)

            out1, out2, out3 = model(image)
            
            loss1 = criterion1(out1, label1)
            loss2 = criterion3(out2, label2)
            loss3 = criterion2(out3, label3)

            loss = mtl_loss(config, loss1, loss2, loss3)

            wandb.log({'artist loss': loss1, 'date loss': loss2, 
                       'era loss': loss3, 'loss': loss, 
                       'n_datapoints': n_datapoints})

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Backward and optimize
            loss.backward()
            optimizer.step()

            n_datapoints += len(image)

        if epoch % 1 == 0:
            a1t, a2t, a3t, acct = accuracy_MTL_supp(model, train_data, DEVICE, 'train')
            a1, a2, a3, acc = accuracy_MTL_supp(model, val_data, DEVICE, 'test')

            wandb.log({'v_accuracy on artist': a1, 'v_accuracy on date': a2, 
                       'v_accuracy on era': a3, 'total v_accurac': acc, 
                       't_accuracy on artist': a1t, 't_accuracy on date': a2t, 
                       't_accuracy on era': a3t, 'total t_accurac': acct,
                       'loss': loss.item(), 'n_datapoints': n_datapoints, 'epoch': epoch})
            
        if epoch % 10 == 0:
            save_path = args.out_dir + f'/model_multi_epoch_{epoch}_{config}.pt'
            torch.save(model.state_dict(), save_path)
            
    torch.save(model.state_dict(), args.out_dir + f'/model_multi_epoch_{epoch}_{config}.pt')
    print('Finished Training')


def train_STL_classification(train_data, val_data, model, task, epochs, weights, classes, model_config):
    ''' Train single-task classification model '''

    loss_weights = torch.FloatTensor(weights).to(DEVICE)

    # Classification loss
    # criterion = nn.CrossEntropyLoss(weight=loss_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Send model to device
    model = model.to(DEVICE)

    print(f'Dataloader length: {len(train_data)}')

    n_datapoints = 0 

    for epoch in trange(epochs):
        for data in train_data:

            # Retrieve labels
            image, label1, _, label3 = data

            # Send tensors to device
            image = image.to(DEVICE)
            label1 = label1.to(DEVICE)
            label3 = label3.to(DEVICE)

            # Determine label for specified task
            if task == 1:
                label = label1.long()
            if task == 3:
                label = label3.long()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward, backward, optimize
            out, _ = model(image, model_config)

            loss = criterion(out, label)
            loss.backward()

            wandb.log({f'{task_map[task]} loss': loss.item(), 'n_datapoints': n_datapoints})

            n_datapoints += len(image)

            optimizer.step()

        if epoch % 1 == 0:
            a1 = accuracy_STL_classification(model, val_data, task, DEVICE, 'test', classes, model_config)
            a1t = accuracy_STL_classification(model, train_data, task, DEVICE, 'train', classes, model_config)

            wandb.log({f'v_accuracy on {task_map[task]}': a1, f't_accuracy on {task_map[task]}': a1t,
                        "loss": loss.item(), 'n_datapoints': n_datapoints, 'epoch': epoch})

        if epoch % 10 == 0:
            save_path = args.out_dir + f'/model_{task}_classification_single_epoch_{epoch}.pt'
            torch.save(model.state_dict(), save_path)

    torch.save(model.state_dict(), args.out_dir + f'/model_{task}_classification_single_epoch_{epoch}.pt')
    print('Finished Training')

def train_STL_regression(train_data, val_data, model, epochs, model_config):

    ''' Train single-task regression model '''

    # Regression loss
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model = model.to(DEVICE)

    print(f'Dataloader length: {len(train_data)}')

    n_datapoints = 0

    for epoch in trange(epochs):
        for data in train_data:

            # Rertieve label
            image, _, label2, _ = data

            # Send tensors to device
            image = image.to(DEVICE)
            label = label2.float().unsqueeze(1).to(DEVICE)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward, backward, optimize
            out, _ = model(image, model_config)

            loss = criterion(out, label)

            wandb.log({'date loss': loss.item(), 'n_datapoints': n_datapoints})

            loss.backward()
            optimizer.step()

            n_datapoints += len(image)

        if epoch % 1 == 0:
            del image, label
            torch.cuda.empty_cache()
            a1 = accuracy_STL_regression(model, val_data, DEVICE, 'test', model_config)
            a1t = accuracy_STL_regression(model, train_data, DEVICE, 'train', model_config)

            wandb.log({'v_accuracy on date': a1, 't_accuracy on date': a1t,
                        "loss": loss.item(), 'n_datapoints': n_datapoints, 
                        'epoch': epoch})

        if epoch % 10 == 0:
            save_path = args.out_dir + f'/model_2_regression_single_epoch_{epoch}.pt'
            torch.save(model.state_dict(), save_path)

    torch.save(model.state_dict(), args.out_dir + f'/model_2_regression_single_epoch_{epoch}.pt')
    print('Finished Training')


def get_class_weights(data):
    artists = dict(data.artists_weights)
    eras = dict(data.eras_weights)

    artists_weights = {k: artists[list(artists.keys())[0]] / v for k, v in artists.items()}
    eras_weights = {k: eras[list(eras.keys())[0]] / v for k, v in eras.items()}

    class_artist_weights = []
    class_eras_weights = []

    for key in data.artist_map:
        class_artist_weights.append(artists_weights[key])

    for key in data.era_map:
        class_eras_weights.append(eras_weights[key])

    return class_artist_weights, class_eras_weights


def mtl_loss(tasks, loss1, loss2, loss3):
    if tasks == 'ed':
        return (loss2 / 1000) + loss3 
    if tasks == 'ae':
        return loss1 + loss2
    if tasks == 'ad':
        return loss1 + (loss2 / 1000) 
    if tasks == 'aed':
        return loss1 + (loss2 / 1000) + loss3


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train a single/multitask model on a specified dataset.')

    # Flag mtl config 
    parser.add_argument('--train_data', dest='train_file', required=True, help='Path to training dataset')
    parser.add_argument('--val_data', dest='val_file', required=True, help='Path to validation dataset')
    parser.add_argument('--image_dir', dest='imdir', required=True, help='Path to images directory')
    parser.add_argument('--output', dest='out_dir', required=False, help='Directory to store model')
    parser.add_argument('--task_config', dest='config', required=False, help='Task configuration')
    parser.add_argument('--model_config', dest='model_config', required=False, help='Model configuration')
    parser.add_argument('--image_size', dest='im_size', type=int, default=224, required=True, help='Image size')
    parser.add_argument('--task', dest='task', required=True, help='Task to train')
    parser.add_argument('--single_task', dest='single_task', type=int, required=False, help='Singletask model to train')

    parser.add_argument('--learning_rate', dest='lr', type=float, default=0.0001, required=False, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=2, required=False, help='Size of training batch')
    parser.add_argument('--epochs', dest='n_epochs', type=int, default=2, required=False, help='Number of epochs to train')
    parser.add_argument('--hidden_s', dest='hidden_size', type=int, default=512, required=False, help='Size of hidden layers')
    parser.add_argument('--offline', dest='offline', action='store_true', help='Run without syncing WandB')
    
    args = parser.parse_args()

    if args.task not in ['single', 'multi', 'cross-stitch', 'nddr-cnn', 'mtan']:
        raise ValueError(f'Invalid task specification {args.model}. Select from: single, multi')
    if args.task == 'multi':
        if args.config not in ['ae', 'ed', 'ad', 'aed']:
            raise ValueError(f'Invalid mtl task config specification {args.config}. Select from: ae, ed, ad, aed')
    if args.model_config not in ['fr_vit', 'vit', 'fr_res', 'res']:
        raise ValueError(f'Invalid mtl model config specification {args.model_config}. Select from: fr_vit, vit, fr_res, res')
    if not os.path.isfile(args.train_file):
        raise FileExistsError('Training dataset does not exist.')
    if not os.path.isfile(args.val_file):
        raise FileExistsError('Validation dataset does not exist.')

    task_map = {1: 'artist', 2: 'date', 3: 'era'}

    # Process data accordingly
    train_data = DataProcessor(args.train_file, args.im_size, 'train', args.imdir)
    train_classes = train_data.get_labels()

    mapping = json.load(open('data/mapping.json', 'r'))
    
    val_data = DataProcessor(args.val_file, args.im_size, 'test', args.imdir, mappings=mapping)
    val_classes = val_data.get_labels()

    tr_artist_weights, tr_era_weights = get_class_weights(train_data)

    artist_classes, era_classes = val_data.artists, val_data.eras

    # Load train and validation sets
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if args.offline:
        os.environ['WANDB_MODE'] = 'dryrun'

    # Initialize wandb
    wandb.init(project='thesis-art-analysis')
    wandb.config.update({'learning method': args.task, 'task': args.single_task, 
                         'batch_size': args.batch_size, 'n_epochs': args.n_epochs})

    # Select device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running on: {DEVICE}')

    # Train single task models
    if args.task == 'single':
        if args.single_task == 1:
            model1 = SingleTaskClassification(args.im_size, args.hidden_size, train_classes[0], 2, args.model_config)
            wandb.watch(model1)
            if torch.cuda.device_count() > 1:
                model1 = nn.DataParallel(model1)

            train_STL_classification(train_loader, val_loader, model1, args.single_task, args.n_epochs, tr_artist_weights, artist_classes, args.model_config)

        if args.single_task == 2:
            model2 = SingleTaskRegression(args.im_size, args.hidden_size, args.model_config)
            wandb.watch(model2)
            if torch.cuda.device_count() > 1:
                model2 = nn.DataParallel(model2)

            train_STL_regression(train_loader, val_loader, model2, args.n_epochs, args.model_config)

        if args.single_task == 3:
            model3 = SingleTaskClassification(args.im_size, args.hidden_size, train_classes[1], 4, args.model_config)
            wandb.watch(model3)
            if torch.cuda.device_count() > 1:
                model3 = nn.DataParallel(model3)

            train_STL_classification(train_loader, val_loader, model3, args.single_task, args.n_epochs, tr_era_weights, era_classes, args.model_config)

    
    # Train multi task models
    if args.task == 'multi':
        model = MultiTaskHS(args.im_size, args.hidden_size, train_classes[0], train_classes[1], args.model_config)
        # print(f'mtl model params: {count_parameters(model)}')
        # exit()
        wandb.watch(model)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        train_MTL(train_loader, val_loader, model, args.n_epochs, [tr_artist_weights, tr_era_weights], args.config, args.model_config)


if args.task == 'cross-stitch':
    # Retrieve config file
    cv2.setNumThreads(0)
    p = create_config('mtl_repo/configs/env.yml', 'mtl_repo/configs/nyud/resnet50/cross_stitch.yml')
    print(colored(p, 'red'))

    # Get model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    model = torch.nn.DataParallel(model)

    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    train_MTL_configs(train_loader, val_loader, args.n_epochs, [tr_artist_weights, tr_era_weights], model, args.config, optimizer)


if args.task == 'nddr-cnn':
    
    # Retrieve config file
    cv2.setNumThreads(0)
    p = create_config('mtl_repo/configs/env.yml', 'mtl_repo/configs/nyud/resnet50/nddr_cnn.yml')
    print(colored(p, 'red'))

    # Get model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    model = torch.nn.DataParallel(model)

    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    train_MTL_configs(train_loader, val_loader, args.n_epochs, [tr_artist_weights, tr_era_weights], model, args.config, optimizer)

if args.task == 'mtan':

    # Retrieve config file
    cv2.setNumThreads(0)
    p = create_config('mtl_repo/configs/env.yml', 'mtl_repo/configs/nyud/resnet50/mtan.yml')
    print(colored(p, 'red'))

    # Get model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    model = torch.nn.DataParallel(model)

    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    train_MTL_configs(train_loader, val_loader, args.n_epochs, [tr_artist_weights, tr_era_weights], model, args.config, optimizer)

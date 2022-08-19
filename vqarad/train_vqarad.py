from torch import cuda

# print("I love you programmer ")
from PIL import Image
import torch
import timm
import requests
import torchvision.transforms as transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


import argparse     ### passing argument to program
# from turtle import color

from cv2 import dft  ##Changing the contrast and brightness of an image! ??
"""Fourier Transform is used to analyze the frequency characteristics of various 
filters. For images, 2D Discrete Fourier Transform (DFT) is used to find the 
frequency domain."""
from utils_vqarad import seed_everything, Model, VQAMed, train_one_epoch, validate,\
test, load_data, LabelSmoothing
# import wandb  aya niaz hast estefadeh shavad???
import pandas as pd
import numpy as np
import torch ###  orch is an open-source machine learning library, a scientific computing framework, and a script language based on the Lua programming language. It provides a wide range of algorithms for deep learning, \
###and uses the scripting language LuaJIT, and an underlying C implementation.
import torch.nn as nn  ##  how it works to make the code either more concise, or more flexible.
###modules and classes torch.nn , torch.optim , Dataset , and DataLoader to help you create and train neural networks. In order to fully utilize their power and customize them for your problem
from torch.utils.data import DataLoader
###Sampler classes are used to specify the sequence of indices/keys used in data loading
###They represent iterable objects over the indices to datasets
import torch.optim as optim
### different optimization object !!
"""
torch.optim is a package implementing various optimization algorithms. 
Most commonly used methods are already supported, and the interface is general enough,
so that more sophisticated ones can be also easily integrated in the future.
To use torch.optim you have to construct an optimizer object, that will 
hold the current state and will update the parameters based on the computed gradients.
"""

import torch.optim.lr_scheduler as lr_scheduler
### custom scheduler !!
from torchvision import transforms
""" 
This library is part of the PyTorch project. PyTorch is an open source machine learning framework.
The torchvision package consists of popular datasets, model architectures, 
and common image transformations for computer vision."""

from torch.cuda.amp import GradScaler

"""
To prevent underflow, “gradient scaling” multiplies the network’s loss(es) by a scale factor and 
invokes a backward pass on the scaled loss(es). Gradients flowing backward through the network
 are then scaled by the same factor. In other words, gradient values have a larger magnitude,
so they don’t flush to zero.
idea!!
 with autocast():
        output = model(input)
        loss = loss_fn(output, target)"""
from transformers import BertTokenizer

import os

import warnings

import matplotlib.pyplot as plt

import datetime

import utils_vqarad


# np.random.seed(10)
warnings.simplefilter("ignore", UserWarning)


### we should alt this 

   
    

if __name__ == '__main__':
#### maximize accuracy, minimize loss
    if cuda.empty_cache:
        cuda.empty_cache
        print("Succseeful Empty cache\n")
    parser = argparse.ArgumentParser(description = "Finetune on VQARAD")

    # parser.add_argument('--category', type = str, required =False,default="all", help = "SPECIFIC run name")
    # parser.add_argument('--allcategory', type = str, required = False,default="False", help = "SPECIFIC run name")
    # parser.add_argument('--num_vis', type = int, required = False, default=5, help = "num of visual embeddings")

    parser.add_argument('--run_name', type = str, required = True, help = "SPECIFIC run name")
    parser.add_argument('--data_dir', type = str, required = False, default = "../data/vqarad/", help = "path for data")
    parser.add_argument('--model_dir', type = str, required = False, default = "../roco_mlm/recorder_520.pt", help = "path to load weights")
    
    # parser.add_argument('--model_dir', type = str, required = False, default = "/home/viraj.bagal/viraj/medvqa/Weights/roco_mlm/val_loss_3.pt", help = "path to load weights")
    parser.add_argument('--save_dir', type = str, required = False, default = "../VQAradSave/", help = "path to save weights")
    parser.add_argument('--question_type', type = str, required = False, default = None,  help = "choose specific category if you want")
    parser.add_argument('--use_pretrained', action = 'store_true', default = False, help = "use pretrained weights or not")
    parser.add_argument('--mixed_precision', action = 'store_true', default = False, help = "use mixed precision or not")
    parser.add_argument('--clip', action = 'store_true', default = False, help = "clip the gradients or not")

    parser.add_argument('--seed', type = int, required = False, default = 42, help = "set seed for reproducibility")
    parser.add_argument('--num_workers', type = int, required = False, default = 0, help = "number of workers")
    parser.add_argument('--epochs', type = int, required = False, default =200, help = "num epochs to train")
    parser.add_argument('--train_pct', type = float, required = False, default = 1.0, help = "fraction of train samples to select")
    parser.add_argument('--valid_pct', type = float, required = False, default = 1.0, help = "fraction of validation samples to select")
    parser.add_argument('--test_pct', type = float, required = False, default = 1.0, help = "fraction of test samples to select")

    parser.add_argument('--max_position_embeddings', type = int, required = False, default = 28, help = "max length of sequence")
    parser.add_argument('--batch_size', type = int, required = False, default =4, help = "batch size")
    parser.add_argument('--lr', type = float, required = False, default = 1e-4, help = "learning rate'")
    # parser.add_argument('--weight_decay', type = float, required = False, default = 1e-2, help = " weight decay for gradients")
    parser.add_argument('--factor', type = float, required = False, default = 0.1, help = "factor for rlp")
    parser.add_argument('--patience', type = int, required = False, default = 10, help = "patience for rlp")
    # parser.add_argument('--lr_min', type = float, required = False, default = 1e-6, help = "minimum lr for Cosine Annealing")
    parser.add_argument('--hidden_dropout_prob', type = float, required = False , default = 0.3, help = "hidden dropout probability")
    """
    This conceptualization suggests that perhaps dropout breaks-up situations where network layers 
    -adapt to correct mistakes from prior layers, in turn making the model more robust.
    sparse representations in autoencoder models
    The term “dropout” refers to dropping out units (hidden and visible) in a neural network.
    p – probability of an element to be zeroed. Default: 0.5
    """
    parser.add_argument('--smoothing', type = float, required = False, default = None, help = "label smoothing")

    parser.add_argument('--image_size', type = int, required = False, default = 224, help = "image size")
    parser.add_argument('--hidden_size', type = int, required = False, default = 768, help = "hidden size")
    parser.add_argument('--vocab_size', type = int, required = False, default = 30522, help = "vocab size")
    parser.add_argument('--type_vocab_size', type = int, required = False, default = 2, help = "type vocab size")
    parser.add_argument('--heads', type = int, required = False, default = 12, help = "heads")
    parser.add_argument('--n_layers', type = int, required = False, default = 4, help = "num of layers")
    parser.add_argument('--bert_model', type = str, required = False, default = "bert-base-uncased", help = "Name of Bert Model")
    parser.add_argument('--image_embedding', type = str, required = False, default = "hybrid", help = "Name of image extractor")


    args = parser.parse_args()

    seed_everything(args.seed)


    train_df,_, test_df = load_data(args)
    # train_df, test_df = load_data(args)
    print("successful loading data ")

    if args.question_type:
            
        train_df = train_df[train_df['question_type']==args.question_type].reset_index(drop=True)
        # val_df = val_df[val_df['question_type']==args.question_type].reset_index(drop=True)
        test_df = test_df[test_df['question_type']==args.question_type].reset_index(drop=True)


    df = pd.concat([train_df, test_df]).reset_index(drop=True)
    df['answer'] = df['answer'].str.lower()
    ans2idx = {ans:idx for idx,ans in enumerate(df['answer'].unique())}
    # print(ans2idx)
    idx2ans = {idx:ans for ans,idx in ans2idx.items()}
    df['answer'] = df['answer'].map(ans2idx).astype(int)
    train_df = df[df['mode']=='train'].reset_index(drop=True)
    # val_df = df[df['mode']=='eval'].reset_index(drop=True)
    test_df = df[df['mode']=='test'].reset_index(drop=True)
    
    # print("labeling the classes")
    num_classes = len(ans2idx) ### i think that the model do not generate answer just classify them

    args.num_classes = num_classes


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Model(args)
    # print("3")
    if args.use_pretrained:
        model.load_state_dict(torch.load(args.model_dir))
    
    
    model.classifier[2] = nn.Linear(args.hidden_size, num_classes)
        
    model.to(device) ### If you need to move a model to GPU via .cuda(),
    ##please do so before constructing optimizers for it. 
    ##Parameters of a model after .cuda() will be different objects with those before the call.

    # wandb.watch(model, log='all')

    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience = args.patience, factor = args.factor, verbose = True)

    if args.smoothing:
        criterion = LabelSmoothing(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    scaler = GradScaler()


    if args.image_embedding == "resnet":

    
        train_tfm = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.RandomResizedCrop(224,scale=(0.5,1.0),ratio=(0.75,1.333)),
                                        transforms.RandomRotation(10),
                                        # Cutout(),
                                        transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.4),
                                        transforms.ToTensor(), 
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


        test_tfm = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        val_tfm = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    elif args.image_embedding == "hybrid" :
        train_tfm = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
        test_tfm = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
        val_tfm = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])




    # traindataset = VQAMed(train_df, imgsize = args.image_size, tfm = train_tfm, args = args)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    # print('tokenizer!')

    traindataset = VQAMed(train_df, tfm = train_tfm, args = args,mode="train", tokenizer=tokenizer)
    # valdataset = VQAMed(val_df, tfm = val_tfm, args = args)
    testdataset = VQAMed(test_df, tfm = test_tfm, args = args,mode="test", tokenizer=tokenizer)

    trainloader = DataLoader(traindataset, batch_size = args.batch_size, shuffle=True, num_workers = args.num_workers)
    # valloader = DataLoader(valdataset, batch_size = args.batch_size, shuffle=False, num_workers = args.num_workers)
    testloader = DataLoader(testdataset, batch_size = args.batch_size, shuffle=False, num_workers = args.num_workers)

    val_best_acc = 0
    test_best_acc = 0
    best_loss = np.inf
    counter = 0

     # Early stopping
    last_loss = 100
    patience = 20
    triggertimes = 0


    all_train_loss= []
    all_train_acc=[]
    all_test_acc = []
    all_test_loss= []
    log_dict= []
    last_epoch= args.epochs
    for epoch in range(args.epochs):

        print(f'Epoch {epoch+1}/{args.epochs}')
        
        
        # if epoch == 80:
        #     args.lr=3e-6



        train_loss, train_acc = train_one_epoch(trainloader, model, optimizer, criterion, device, scaler, args, train_df,idx2ans)
        # val_loss, val_predictions, val_acc, val_bleu = validate(valloader, model, criterion, device, scaler, args, val_df,idx2ans)
        test_loss, test_predictions, test_acc = test(testloader, model, criterion, device, scaler, args, test_df,idx2ans)
        if test_loss != "Noisy":
            current_loss = test_loss


            print('The Current Loss:', current_loss)

            if current_loss > last_loss:
                trigger_times += 1
                print('Trigger Times:', trigger_times)

                if trigger_times >= patience:
                    print('Early stopping!\nStart to test process.')
                    last_epoch = epoch
                    break

            else:
                print('trigger times: 0')
                trigger_times = 0

            last_loss = current_loss
            # log_dict = test_acc
            log_dict.append(test_acc)

            # for k,v in test_acc.items():
            #     log_dict[k] = v
            # log_dict['test_loss'] = test_loss

            all_test_loss.append(test_loss)
            all_test_acc.append(test_acc['total_acc'])

            if test_acc['total_acc'] > test_best_acc:
                torch.save(model.state_dict(),os.path.join(args.save_dir, f'{args.run_name}_test_acc.pt'))
                test_best_acc=test_acc['total_acc']
        
        
        
        scheduler.step(train_loss)

       

        all_train_loss.append(train_loss)
        all_train_acc.append(train_acc["total_acc"])
        

        # log_dict['train_loss'] = train_loss
        # log_dict['learning_rate'] = optimizer.param_groups[0]["lr"]

        # wandb.log(log_dict)
        if test_loss != "Noisy":
            content = f'Train loss: {(train_loss):.4f}, Train acc: {train_acc},Test loss: {(test_loss):.4f},  Test acc: {test_acc}'
            print(content)
        else :
            print(f'Train loss: {(train_loss):.4f}, Train acc: {train_acc}')   
      
      
    torch.save(model.state_dict(),os.path.join(args.save_dir, f'{args.run_name}FINAL_acc.pt'))

   

    df = pd.read_excel('OUTPUTrad.xlsx')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]



    df = df.append({'model_name' : args.run_name, 'bert_model' : args.bert_model, \
        'image_embedding':args.image_embedding,\
        'epoch' : last_epoch, "lr" : args.lr, "loss_train" : \
        all_train_loss, "overall_accuracy_train" : all_train_acc,"loss_test":all_test_loss, "overall_accuracy_test": all_test_acc},\
        ignore_index = True)
    # ["model2",args.bert_model ,args.epochs, args.lr, train_loss,train_acc["total_acc"]]

    df.to_excel("OUTPUTrad.xlsx") 
    
    with open(f'recorder{args.run_name}.txt', 'w') as fp:
        for item in log_dict:
            # write each item on a new line
            fp.write("%s\n" % item)
    print('Done')
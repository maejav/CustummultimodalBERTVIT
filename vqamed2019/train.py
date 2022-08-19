import argparse
import pwd
from xmlrpc.client import boolean
from utils import seed_everything, Model, VQAMed, train_one_epoch, validate, test, load_data, LabelSmoothing, train_img_only, val_img_only, test_img_only
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms, models
from torch.cuda.amp import GradScaler
import os
import warnings
import albumentations as A
import pretrainedmodels
from albumentations.core.composition import OneOf
from albumentations.pytorch.transforms import ToTensorV2
import timm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

warnings.simplefilter("ignore", UserWarning)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description = "Finetune on ImageClef 2019")

    parser.add_argument('--run_name', type = str, required = True, help = "run name for wandb")


    parser.add_argument('--data_dir', type = str, required = False, default = "../data/vqamed", help = "path for data")
    
    parser.add_argument('--model_dir', type = str, required = False, default = "../VQAmedSave/recorderpremedabnormalityresume30FINAL_acc.pt", help = "path to load weights")
    parser.add_argument('--save_dir', type = str, required = False, default = "../VQAmedSave/", help = "path to save weights")
    parser.add_argument('--category', type =str, required = False, default ="Abnormality",  help = "choose specific category if you want")
    parser.add_argument('--use_pretrained', action = 'store_true', default = True, help = "use pretrained weights or not")
    parser.add_argument('--mixed_precision', action = 'store_true', default = True, help = "use mixed precision or not")
    parser.add_argument('--clip', action = 'store_true', default = False, help = "clip the gradients or not")

    parser.add_argument('--seed', type = int, required = False, default = 42, help = "set seed for reproducibility")
    parser.add_argument('--num_workers', type = int, required = False, default = 4, help = "number of workers")
    parser.add_argument('--epochs', type = int, required = False, default = 100, help = "num epochs to train")
    parser.add_argument('--train_pct', type = float, required = False, default = 1, help = "fraction of train samples to select")
    parser.add_argument('--valid_pct', type = float, required = False, default = 1, help = "fraction of validation samples to select")
    parser.add_argument('--test_pct', type = float, required = False, default = 1, help = "fraction of test samples to select")

    parser.add_argument('--max_position_embeddings', type = int, required = False, default = 28, help = "max length of sequence")
    parser.add_argument('--batch_size', type = int, required = False, default = 8 , help = "batch size")
    parser.add_argument('--lr', type = float, required = False, default = 1e-3, help = "learning rate'")
    # parser.add_argument('--weight_decay', type = float, required = False, default = 1e-2, help = " weight decay for gradients")
    parser.add_argument('--factor', type = float, required = False, default = 0.1, help = "factor for rlp")
    parser.add_argument('--patience', type = int, required = False, default = 20, help = "patience for rlp")
    # parser.add_argument('--lr_min', type = float, required = False, default = 1e-6, help = "minimum lr for Cosine Annealing")
    parser.add_argument('--hidden_dropout_prob', type = float, required = False, default = 0, help = "hidden dropout probability")
    parser.add_argument('--smoothing', type = float, required = False, default =0.001, help = "label smoothing")

    parser.add_argument('--image_size', type = int, required = False, default = 224, help = "image size")
    parser.add_argument('--hidden_size', type = int, required = False, default = 768, help = "hidden size")
    parser.add_argument('--vocab_size', type = int, required = False, default = 1671, help = "vocab size")
    parser.add_argument('--type_vocab_size', type = int, required = False, default = 2, help = "type vocab size")
    parser.add_argument('--heads', type = int, required = False, default = 12, help = "heads")
    parser.add_argument('--n_layers', type = int, required = False, default= 4, help = "num of layers")
    parser.add_argument('--num_vis', type = int, required = False, default=5, help = "num of visual embeddings")
    # parser.add_argument('--all_category', type =boolean, required= False, help = "yes or no category")
    parser.add_argument('--image_embedding', type = str, required = False, default = "vision", help = "Name of image extractor")
    parser.add_argument('--bert_model', type = str, required = False, default = "bert-base-uncased", help = "Name of Bert Model")
    # parser.add_argument('--allcategory', type =boolean, required =False , default =False ,  help = "choose specific category if you want")
    parser.add_argument('--allcategory', type = str, required =False , default ="False" ,  help = "choose specific category if you want")


    args = parser.parse_args()

    seed_everything(args.seed)

    # if  args.category==None:
    train_df, val_df, test_df = load_data(args)

    # else:

    #     train_df, val_df, test_df = load_data(args)
        # print("oooooooooooooooooooooo")

    # if  args.category: 
    #     print("arg.categorykkkkkkkkkkkkkkkkkkkkkk ", args.category)

    #     ##  DataFrame has a MultiIndex, this method can remove one or more levels
    #     ## added as a column, and a new sequential index is used
    #     ## We can use the drop parameter to avoid the old index being added as a column

    #     # train_df = train_df[train_df['category']==args.category].reset_index(drop=True)
    #     # print("RESET ",train_df)
    #     # val_df = val_df[val_df['category']==args.category].reset_index(drop=True)
    #     # test_df = test_df[test_df['category']==args.category].reset_index(drop=True)

    #     ## Whether each element in the DataFrame is contained in values

    # temp = train_df['answer'].isin(['yes', 'no'])
    # print("~train_df['answer'].isin(['yes', 'no'])",temp)
    # train_df = train_df[~train_df['answer'].isin(['yes', 'no'])]
    # print("gggggggggggggggggggggggggggggggg",train_df.columns)
    # val_df = val_df[val_df['answer'].isin(['yes', 'no'])]
    # test_df = test_df[test_df['answer'].isin(['yes', 'no'])]


    df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)

    # print("TTTTTTTTTTTTTTTRAinnnnnnn\n", train_df)
    # print("vvvvvvvvvvvvvvvvvvvvvvvv\n", val_df)
    # print("sssssssssssssssssss\n", test_df)

    ans2idx = {ans:idx for idx,ans in enumerate(df['answer'].unique())}
    idx2ans = {idx:ans for ans,idx in ans2idx.items()}
    df['answer'] = df['answer'].map(ans2idx).astype(int)
    train_df = df[df['mode']=='train'].reset_index(drop=True)
    val_df = df[df['mode']=='eval'].reset_index(drop=True)
    test_df = df[df['mode']=='test'].reset_index(drop=True) ## with lower 

    num_classes = len(ans2idx)

    args.num_classes = num_classes
    print("number of class\n ",num_classes)



    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Model(args)

    if args.use_pretrained:
        print("Resume Model")
        model.load_state_dict(torch.load(args.model_dir))


    # model.classifier[2] = nn.Linear(args.hidden_size, num_classes)


        
    model.to(device)



    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience = args.patience, factor = args.factor, verbose = True)


    if args.smoothing:
        criterion = LabelSmoothing(smoothing=args.smoothing)
        # print("smoooooottthhth")
    else:
        criterion = nn.CrossEntropyLoss()

    scaler = GradScaler()
### transformation on images 
    if args.image_embedding == "vision":
        train_tfm = transforms.Compose([transforms.ToPILImage(),
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
        test_tfm = transforms.Compose([transforms.ToPILImage(),
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
        val_tfm = transforms.Compose([transforms.ToPILImage(),
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])



    else:

        train_tfm = transforms.Compose([
                                        
                                        transforms.ToPILImage(),
                                        transforms.RandomResizedCrop(224,scale=(0.75,1.25),ratio=(0.75,1.25)),
                                        transforms.RandomRotation(10),
                                        # Cutout(),
                                        transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4,hue=0.4),
                                        transforms.ToTensor(), 
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


        val_tfm = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        test_tfm = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    print("TTTTTTTTTTTTTTTRAinnnnnnn\n", len(train_df))
    print("vvvvvvvvvvvvvvvvvvvvvvvv\n", len(val_df))
    print("sssssssssssssssssss\n", len(test_df))


    traindataset = VQAMed(train_df, imgsize = args.image_size, tfm = train_tfm, args = args)
    valdataset = VQAMed(val_df, imgsize = args.image_size, tfm = val_tfm, args = args)
    testdataset = VQAMed(test_df, imgsize = args.image_size, tfm = test_tfm, args = args)

    trainloader = DataLoader(traindataset, batch_size = args.batch_size, shuffle=True, num_workers = args.num_workers)
    valloader = DataLoader(valdataset, batch_size = args.batch_size, shuffle=False, num_workers = args.num_workers)
    testloader = DataLoader(testdataset, batch_size = args.batch_size, shuffle=False, num_workers = args.num_workers)

    best_acc1 = 0
    best_acc2 = 0
    best_loss = np.inf
    counter = 0

    last_loss = 100
    patience = 20
    triggertimes = 0

    all_train_loss= []
    all_train_acc=[]
    all_test_acc = []
    all_test_loss= []
    all_val_loss = []
    # all_val_bleu=[]
    all_test_bleu=[]
    logdict=[]
    last_epoch= args.epochs
    for epoch in range(args.epochs):

        print(f'Epoch {epoch+1}/{args.epochs}')

        if epoch == 200:
            args.lr=1e-4
        train_loss, preddddict , train_acc, _, _ = train_one_epoch(trainloader, model, optimizer, criterion, device, scaler, args, idx2ans)
        test_loss, predictions, acc, bleu = test(testloader, model, criterion, device, scaler, args, test_df,idx2ans)
        logdict.append({"trainloss":train_loss})
        logdict.append({"trainacc":train_acc})

        logdict.append({"testloss":test_loss})
        logdict.append({"testacc",acc})
        logdict.append({"testbleu",bleu})
        
        all_train_loss.append(train_loss)
        all_train_acc.append(train_acc)
        all_test_loss.append(test_loss)
        
        
        # print("train_loss" , train_loss)
        

        # scheduler.step(train_loss)


        if  args.category == "all": ### false
            all_test_acc.append(acc["total_acc"])
            all_test_bleu.append(bleu["total_bleu"])
            # for k,v in bleu.items():
            #     log_dict[k] = v

            # log_dict['train_loss'] = train_loss
            # log_dict['test_loss'] = test_loss
            # log_dict['learning_rate'] = optimizer.param_groups[0]["lr"]

            # log_dict = acc
        else :
            print("we are in specific category")
            all_test_acc.append(acc)
            all_test_bleu.append(bleu)


          


            # print(log_dict)
        if (epoch % 5) == 0 :
            # print(type(epoch))
            # print(epoch)
            # print ("baghimandeh:",(epoch % 5))
            val_loss, predictions, val_acc, val_bleu = validate(valloader, model, criterion, device, scaler, args, val_df,idx2ans)
            # if args.category == "all":
            #     current_loss=val_loss
            # else :
            #     current_loss=val_loss
            logdict.append({"valloss":val_loss})
            logdict.append({"valacc":val_acc})
            logdict.append({"valbleu":val_bleu})
            current_loss=val_loss
            all_val_loss.append(val_loss)
            # all_val_bleu.append(val_bleu)
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

            scheduler.step(val_loss)

            if  args.category== "all":

                if val_acc['val_total_acc'] > best_acc1:
                    torch.save(model.state_dict(),os.path.join(args.save_dir, f'{args.run_name}_acc.pt'))
                    best_acc1=val_acc['val_total_acc']

            else:

                if val_acc > best_acc1:
                    print('Saving model')
                    torch.save(model.state_dict(),os.path.join(args.save_dir, f'recorder{args.run_name}_acc.pt'))
                    best_acc1 = val_acc 
    # epoo = [epoch+1 for epoch in range(args.epochs)]
   
    torch.save(model.state_dict(),os.path.join(args.save_dir, f'recorder{args.run_name}FINAL_acc.pt'))

    df = pd.read_excel('output_train.xlsx')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]



    df = df.append({'model_name' :args.run_name, 'bert_model' : args.bert_model,\
        'image_embedding':args.image_embedding,\
        'epoch' : last_epoch, "lr" : args.lr, "loss_train":all_train_loss, "overall_accuracy_train" :all_train_acc, "loss_val":all_val_loss,"loss_test":all_test_loss, "overall_accuracy_test": all_test_acc,"all_test_bleu":all_test_bleu, "category":args.category}, ignore_index = True)
    # ["model2",args.bert_model ,args.epochs, args.lr, train_loss,train_acc["total_acc"]]
    df.to_excel("output_train.xlsx") 
    with open(f'recorder{args.run_name}.txt', 'w') as fp:
        for item in logdict:
            # write each item on a new line
            fp.write("%s\n" % item)
    print('Done')
        # if  args.allcategory:


        #     if val_acc['val_total_acc']  > best_acc2:
        #         counter = 0
        #         best_acc2 = val_acc['val_total_acc'] 
        #     else:
        #         counter+=1
        #         if counter > 20:
        #             break      
import os
# import wandb
import argparse
import numpy as np


import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import optim
from torch.optim import lr_scheduler

from roco_utils_rad import load_mlm_data, train_one_epoch, validate, get_keywords, Model, ROCO
# from roco_utils import load_mlm_data, train_one_epoch, validate, get_keywords, Model, ROCO

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Pretrain on ROCO with MLM")

    parser.add_argument('--run_name', type=str, help="name for wandb run", required=True)
    # parser.add_argument('--category', type =str, required = False, default ="Abnormality",  help = "choose specific category if you want")

    parser.add_argument('--data_dir', type=str, default = '../data/roco/all_data', help='path to dataset', required = False)
    ### we do not use absolute address
    parser.add_argument('--save_dir', type=str, default = '../roco_mlm', help='save model weights in this dir', required = False)
    parser.add_argument('--mlm_prob', type=float, default =1.0 ,required = False, help='probability of token being masked')
    parser.add_argument('--mixed_precision', action='store_true', required = False, default = False,  help='mixed precision training or not')
    parser.add_argument('--resume', action='store_true', required = False, default = False,  help='resume training or train from scratch')

    parser.add_argument('--batch_size', type=int, default=1, help='batch_size.')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--patience', type=int, default=5, help='rlp patience')
    parser.add_argument('--factor', type=float, default=0.1, help='rlp factor')
    parser.add_argument('--num_workers', type=int, default= 4, help='num works to generate data.')
    parser.add_argument('--epochs', type=int, default=60, help='epochs to train')

    parser.add_argument('--train_pct', type=float, default=1, help='fraction of train set')
    parser.add_argument('--valid_pct', type=float, default=1, help='fraction of validation set')
    parser.add_argument('--test_pct', type=float, default=1, help='fraction of test set ')

    parser.add_argument('--max_position_embeddings', type=int, default=75, help='embedding size')
    parser.add_argument('--n_layers', type=int, default=4, help='num of heads in multihead attenion')
    parser.add_argument('--heads', type=int, default=12, help='num of bertlayers')
    parser.add_argument('--type_vocab_size', type=int, default=2, help='types of tokens ')
    parser.add_argument('--vocab_size', type=int, default=110000, help='vocabulary size')
    parser.add_argument('--hidden_size', type=int, default=768, help='embedding size')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.3, help='dropout')
    parser.add_argument('--image_embedding', type = str, required = False, default = "hybrid", help = "Name of image extractor")
    parser.add_argument('--bert_model', type = str, required = False, default = "bert-base-multilingual-uncased", help = "Name of Bert Model")
    # parser.add_argument('--num_vis', type = int, required = False,default=5, help = "num of visual embeddings")
    # parser.add_argument('--allcategory', type = str, required =False , default ="False" ,  help = "choose specific category if you want")


    args = parser.parse_args()

    # wandb.init(project='medvqa', name = args.run_name, config = args)

    train_data, val_data  = load_mlm_data(args)
    # No Image: PMC4240561_MA-68-291-g002.jpg
    train_data = train_data[train_data['name']!='PMC4240561_MA-68-291-g002.jpg'].reset_index(drop=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Model(args)

    model.to(device)

    # wandb.watch(model, log='all')

    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience = args.patience, factor = args.factor, verbose = True)
    criterion = nn.NLLLoss()


    # Be careful with the transforms. These medical images must remain meaningful after transform
    if args.image_embedding == "resnet":

        train_tfm = transforms.Compose([transforms.Resize((224,224)), 
                                    transforms.RandomResizedCrop(224,scale=(0.95,1.05),ratio=(0.95,1.05)),
                                    transforms.RandomRotation(5),
                                    transforms.ColorJitter(brightness=0.05,contrast=0.05,saturation=0.05,hue=0.05),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        val_tfm = transforms.Compose([transforms.Resize((224,224)), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif args.image_embedding == "hybrid" :
        train_tfm = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
        # test_tfm = transforms.Compose([
        #     transforms.Resize(256, interpolation=3),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        # ])
        val_tfm = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])


    train_path = os.path.join(args.data_dir,'train')
    val_path = os.path.join(args.data_dir,'validation')
    # test_path = os.path.join(args.data_dir,'test')

    keywords = get_keywords(args)   
    # print("helooooo:",train_data)

    traindataset = ROCO(args, train_data, train_tfm, keywords, mode='train')
    # valdataset = ROCO(args, val_data, val_tfm, keywords, mode = 'validation')
    # print(traindataset.df)
    # print(valdataset.df)

    trainloader = DataLoader(traindataset, batch_size = args.batch_size, shuffle=True, num_workers = args.num_workers)
    # valloader = DataLoader(valdataset, batch_size = args.batch_size, shuffle=False, num_workers = args.num_workers)

    scaler = GradScaler()

    if args.resume:
        ckpt = torch.load(os.path.join(args.save_dir, 'recorder_2.pt'))
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])

    if args.resume:
        best_loss = scheduler.best
    else:
        best_loss = np.inf

    save_recorder = 5
    # for l in trainloader :
        # print(l)
    for epoch in range(args.epochs):
        
        print(f'Epoch {epoch+1}/{args.epochs}')
        

        train_loss, train_acc = train_one_epoch(trainloader, model, criterion, optimizer, scaler, device, args, epoch)
        # val_loss, predictions, acc = validate(valloader, model, criterion, scaler, device, args, epoch)

        scheduler.step(train_loss)

        # if (epoch + 1) % save_recorder == 0:
        #     recorder = {'epoch': epoch,
        #             'optimizer': optimizer.state_dict(),
        #             'scheduler': scheduler.state_dict(),
        #             'scaler': scaler.state_dict(),
        #             'model': model.state_dict()}
        #     name = str('recorder_2'+f'{(epoch+1)}'+'.pt')
        #     torch.save(recorder, os.path.join(args.save_dir,name ))
        if (epoch + 1) % save_recorder == 0:
            name = f'recorder_{args.run_name}{(epoch+1)}.pt'
            torch.save(model.state_dict(), os.path.join(args.save_dir,name ))    

        # wandb.log({'epoch_train_loss': train_loss,
        #         'epoch_val_loss': val_loss,
        #         'epoch_train_acc': train_acc,
        #         'epoch_val_acc': acc,
        #         'learning_rate': optimizer.param_groups[0]["lr"],
        #         'epoch': epoch})

        content = f'Train loss: {(train_loss):.4f}'
        # content = f'Learning rate: {(optimizer.param_groups[0]["lr"]):.7f}, Train loss: {(train_loss):.4f}, Train acc: {(train_acc):.4f}'

        print(content)
        
        # if val_loss<best_loss:
        #     torch.save(model.state_dict(), os.path.join(args.save_dir, 'val_loss_3.pt'))
        #     best_loss=val_loss

    name = f'recorder_{args.run_name}FINALRAD.pt'
    torch.save(model.state_dict(), os.path.join(args.save_dir,name ))    

import os
import numpy as np
import pandas as pd
import math
import torch
import random
# import wandb
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import pickle

import pytorch_lightning as pl

import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import os
from PIL import Image


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_permutation(n):
    index_list = list(np.arange(n))
    perms = []
    for i in range(n):
        lst = index_list[:i] + index_list[i+1:]##np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
        perms.append(np.random.choice(lst))  ##Generate a non-uniform random sample from np.arange(5) of size 3 without replacement:
    return perms ## reorder

    # with open(os.path.join(args.data_dir, 'train/radiology', 'med_vocab.pkl'), 'rb') as f:

def get_keywords(args):
    data = {}
    keywords = []
    with open(os.path.join(args.data_dir, 'train/radiology', 'keywords.txt'), 'rb') as f:
        path =os.path.join(args.data_dir, 'train/radiology', 'keywords.txt')
        # print("!!!!!!!!! it is importan !!!!!!!!!:",path )
        for line in f:
            listt= line.split()
            # print("listtttttt:",listt)
            for ele in range(len(listt)):
                listt[ele]= listt[ele].decode("utf-8") 

            # print("listtttttt:",listt)
            # (key, val) = (listt[0],listt[1:])
            # data[key]= val
            keywords.extend(listt[1:])
    # print(data)
    # Save as pickle
    # with open('data.pkl', 'wb') as f:
    #     pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # # Load as pickle
    # with open('data.pkl', 'rb') as f:
    #     # data_pickle = pickle.load(f)
    #     key = pickle.load(f)

    # keywords = []

    # for k,v in key.items():
    #     keywords.extend(v)
    
    keywords_ = list(set(keywords))

    for word in keywords_:
        # word=word.decode("utf-8")
        keywords.extend(word + '.')
    
    keywords = list(set(keywords))

    return keywords


def load_mlm_data(args):
    val_data=[]
    train_path = os.path.join(args.data_dir,'train','radiology')
    val_path = os.path.join(args.data_dir,'validation','radiology')
    test_path = os.path.join(args.data_dir,'test','radiology')

    train_image_names = os.listdir(os.path.join(train_path,'images'))
    # print("imageeeee ",train_image_names)
    # val_image_names = os.listdir(os.path.join(val_path,'images'))
    # test_image_names = os.listdir(os.path.join(test_path,'images'))

    train_data = pd.read_csv(os.path.join(train_path,'radiologytraindata.csv'))

    # train_data = train_data[['id', 'name', 'caption']]
    # train_data = train_data.reindex(['id', 'name', 'caption'], axis=1)

    train_data = train_data[train_data['name'].isin(train_image_names)]

    # val_data = pd.read_csv(os.path.join(val_path, 'radiologyvaldata.csv'))
    # val_data = val_data[val_data['name'].isin(val_image_names)]

    # test_data = pd.read_csv(os.path.join(test_path, 'testdata.csv'))
    # test_data = test_data[test_data['name'].isin(test_image_names)]
    
    print("load_mlm_data   train data :!!!!!!!",train_data)
    
    print(train_data.columns)
    # train_data = train_data.sample(frac = args.train_pct)
    # val_data = val_data.sample(frac = args.valid_pct)
    train_data=train_data.rename({"caption":"id","id":"name","name":"caption"},axis='columns')
    # val_data=val_data.rename({"caption":"id","id":"name","name":"caption"},axis='columns')


    # test_data = test_data.sample(frac = args.test_pct)
    print("load_mlm_data   train data :!!!!!!!",train_data)
    # print("daaaaaaaataaaa   validation :!!!!!!!",val_data)

    return train_data, val_data

def shuffle_list(some_list):
    length = len(some_list)
    for i in range(length):
        j = i + np.floor(np.random.uniform()*(length - i - 1))
        j = int(j)
        some_list[i], some_list[j] = some_list[j], some_list[i]

    return some_list

#Utils
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def mask_word(sentence, tokenizer, keywords, args):
    tokens = sentence.split()
    output_label = []
    new_tokens = []

    for i, char in enumerate(tokens):
        if char in keywords:
            t = tokenizer.tokenize(char)
            for j in range(len(t)):
                prob = random.random()
                # print("prob", prob)
                # print("args.mlm_prob", args.mlm_prob)

                if prob < args.mlm_prob:
                    
                    output_label.extend([tokenizer.encode(t[j])[1]])
                    t[j] = '[MASK]'

                else:
                    output_label.extend([0])
            new_tokens.extend(t)
        else:
            t = tokenizer.tokenize(char)
            new_tokens.extend(t)
            output_label.extend([0]*len(t))
            
    assert (len(new_tokens)==len(output_label)), "Token len must be equal to label len"
    
    return  new_tokens, output_label

def encode_text(caption,tokenizer, keywords, args):
    part1 = [0 for _ in range(5)]
    #get token ids and remove [CLS] and [SEP] token id
    caption, labels = mask_word(caption, tokenizer, keywords, args)

    
    part2 = tokenizer.convert_tokens_to_ids(caption)
    part2 = part2[:args.max_position_embeddings-8]
    labels = labels[:args.max_position_embeddings-8]
    
    tokens = [tokenizer.cls_token_id] + part1 + [tokenizer.sep_token_id] + part2 + [tokenizer.sep_token_id]
    labels = [0]*(2+len(part1)) + labels + [0]
    
    segment_ids = [0]*(len(part1)+2) + [1]*(len(part2[:args.max_position_embeddings-8])+1)
    input_mask = [1]*len(tokens)
    n_pad = args.max_position_embeddings - len(tokens)
    tokens.extend([0]*n_pad)
    segment_ids.extend([0]*n_pad)
    input_mask.extend([0]*n_pad)
    labels.extend([0]*(n_pad))

    
    return torch.tensor(tokens,dtype=torch.long), torch.tensor(segment_ids,dtype=torch.long), torch.tensor(input_mask,dtype=torch.long), torch.tensor(labels)


def calculate_bleu_score(preds,targets):
  bleu_per_answer = np.asarray([sentence_bleu([idx2ans[target].split()],idx2ans[pred].split()) for pred,target in zip(preds,targets)])
  return np.mean(bleu_per_answer)


def train_one_epoch(loader, model, criterion, optimizer, scaler, device, args, epoch):
    model.train()
    train_loss = []
    PREDS = []
    TARGETS = []
    # bar = tqdm(loader, leave = False)
    for i, (img, caption_token,segment_ids,attention_mask,target) in enumerate(loader):
        # print("(img, caption_token,segment_ids,attention_mask,target):",(img, caption_token,segment_ids,attention_mask,target))

        img, caption_token,segment_ids,attention_mask,target = img.to(device), caption_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)

        caption_token = caption_token.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        # print("imggggggggggggg:", img)

        # print("captionnnnnnnnnnnnnnnnnn:", caption_token)
        # print("segment_idsssssss:", segment_ids)
        # print("attention_mask:", attention_mask)
        # print("targettttttttt:", target)  ## shape target is 2 dim 


        loss_func = criterion
        print('loss_func',loss_func)
        optimizer.zero_grad()

        if args.mixed_precision:
            with torch.cuda.amp.autocast():
                logits = model(img, caption_token, segment_ids, attention_mask)
                logits = logits.log_softmax(-1)  # (bs x seq_len x vocab_size)
                loss = loss_func(logits.permute(0,2,1), target)
                
        else:
            (logits,_,_) = model(img, caption_token, segment_ids, attention_mask)
            # print("lllllllllllllllllllllllogits:",logits)
            logits = logits.log_softmax(-1)
            print("lllllllllllllllllllllllogits:",logits.size())
            print('targetttttttttttttttttttttttttttttt',target.size())
            


            loss = loss_func(logits.permute(0,2,1), target) 
            # print(loss) 
            # print(logits.permute(0,2,1))     

        loss.backward()
        optimizer.step()    

        # if args.mixed_precision:
        #     scaler.scale(loss).backward()
        #     scaler.step(optimizer)
        #     scaler.update()
        # else:
        #     loss.backward()
        #     optimizer.step()    

        # logits = model(img, caption_token, segment_ids, attention_mask)
        # logits = logits.log_softmax(-1)  # (bs x seq_len x vocab_size)
        # loss = loss_func(logits.permute(0,2,1), target)

        # loss.backward()
        # optimizer.step()  


        # logits = logits.permute(0,2,1)   
        # for i in range(logits.size()[])  
        # print('logitssssssssssssssssssssssssssss',logits)
        # print('targetttttttttttttttttttttttttttttt',target)
        
        # bool_label = target > 0
        # # print("bool_label",bool_label)
        # pred=logits[bool_label, :].argmax(1)
####???????
        # pred = logits[aya==0]
        # pred = aya
        # print("pred",pred)

        # valid_labels = target[bool_label]  ### i increment this 
        # print("valid_labels",valid_labels)
        # print("valid_labels.size()",valid_labels.size())

        # for i in range(list(valid_labels.size())[0]) :
        #     valid_labels[i] = 0
    
        # print("valid_labels",valid_labels)

        
        # PREDS.append(pred)
        # TARGETS.append(valid_labels)
        
        loss_np = loss.detach().cpu().numpy()
        # acc = (pred == valid_labels).type(torch.float).mean() * 100.
        train_loss.append(loss_np)
        # bar.set_description('train_loss: %.5f, train_acc: %.2f' % (loss_np,"it is not important"))
        content = f' Train loss: {(loss_np):.4f}'

        print(content)
        

        # wandb.log({'step_train_loss': loss_np,
        #     'step_train_acc': acc,
        #     'train_batch': epoch*len(loader) + i})
        

    # PREDS = torch.cat(PREDS).cpu().numpy()
    # TARGETS = torch.cat(TARGETS).cpu().numpy()

#     # Calculate total accuracy
    # total_acc = (PREDS == TARGETS).mean() * 100.

    return np.mean(train_loss), None

def validate(loader, model, criterion, scaler, device, args, epoch):

    model.eval()
    val_loss = []

    PREDS = []
    TARGETS = []
    bar = tqdm(loader, leave=False)

    with torch.no_grad():
        for i, (img, caption_token,segment_ids,attention_mask,target) in enumerate(bar):

            img, caption_token,segment_ids,attention_mask,target = img.to(device), caption_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
            caption_token = caption_token.squeeze(1)
            attention_mask = attention_mask.squeeze(1)
            
            loss_func = criterion

            if args.mixed_precision:
                with torch.cuda.amp.autocast():
                    logits = model(img, caption_token, segment_ids, attention_mask)
                    logits = logits.log_softmax(-1)  # (bs x seq_len x vocab_size)
                    loss = loss_func(logits.permute(0,2,1), target)
            else:
                logits = model(img, caption_token, segment_ids, attention_mask)
                logits = logits.log_softmax(-1)  # (bs x seq_len x vocab_size)
                loss = loss_func(logits.permute(0,2,1), target)       


            # logits = model(img, caption_token, segment_ids, attention_mask)
            # logits = logits.log_softmax(-1)  # (bs x seq_len x vocab_size)
            # loss = loss_func(logits.permute(0,2,1), target)

            bool_label = target > 0
            pred = logits[bool_label, :].argmax(1)
            valid_labels = target[bool_label]   
        
            PREDS.append(pred)
            TARGETS.append(valid_labels)
            
            loss_np = loss.detach().cpu().numpy()

            val_loss.append(loss_np)

            acc = (pred == valid_labels).type(torch.float).mean() * 100.

            bar.set_description('val_loss: %.5f, val_acc: %.5f' % (loss_np, acc))

            # wandb.log({'step_val_loss': loss_np,
            #     'step_val_acc': acc,
            #     'val_batch': epoch*len(loader) + i})

        val_loss = np.mean(val_loss)

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    # Calculate total accuracy
    total_acc = (PREDS == TARGETS).mean() * 100.


    return val_loss, PREDS, total_acc
    
def test(loader):

    model.eval()

    PREDS = []
    TARGETS = []

    with torch.no_grad():
        for (img,caption_token,attention_mask,target) in tqdm(loader, leave=False):

            img, caption_token,segment_ids,attention_mask,target = img.to(device), caption_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
            caption_token = caption_token.squeeze(1)
            attention_mask = attention_mask.squeeze(1)
            
            logits = model(img, caption_token, segment_ids, attention_mask)
        
            bool_label = target > 0
            pred = logits[bool_label, :].argmax(1)
            valid_labels = target[bool_label]   
        
            PREDS.append(pred)
            TARGETS.append(valid_labels)

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    total_acc = (PREDS == TARGETS).mean() * 100.

    return PREDS, total_acc



class ROCO(Dataset):
    def __init__(self, args, df, tfm, keys, mode):
        self.df = df.values
        self.args = args
        self.path = args.data_dir
        self.tfm = tfm
        self.keys = keys
        self.mode = mode
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        name = self.df[idx,1]              
        path = os.path.join(self.path, self.mode, 'radiology', 'images',name)


        img = Image.open(path).convert('RGB')
        
        if self.tfm:
            img = self.tfm(img)
    
        caption = self.df[idx, 2].strip()
    
        
        tokens, segment_ids, input_mask, targets = encode_text(caption, self.tokenizer, self.keys, self.args)

        
        return img, tokens, segment_ids, input_mask, targets


# img, caption_token,segment_ids,attention_mask,target

class ROCOModule(pl.LightningDataModule):
    def __init__(self, args):
        super(ROCOModule, self).__init__()

        self.args = args

    def setup(self, stage=None):

        train, val, _ = load_mlm_data(self.args)

        train = train[train['name']!='PMC4240561_MA-68-291-g002.jpg'].reset_index(drop=True)

        train_tfm = transforms.Compose([transforms.Resize((224,224)), 
                                    transforms.RandomResizedCrop(224,scale=(0.95,1.05),ratio=(0.95,1.05)),
                                    transforms.RandomRotation(5),
                                    transforms.ColorJitter(brightness=0.05,contrast=0.05,saturation=0.05,hue=0.05),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        val_tfm = transforms.Compose([transforms.Resize((224,224)), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.train = ROCO(self.args, train, train_tfm, 'train')
        self.val = ROCO(self.args, val, val_tfm, 'validation')
        self.test = ROCO(self.args, test, val_tfm, 'test')

    def train_dataloader(self):
        return DataLoader(self.train, batch_size = self.args.batch_size, shuffle = True, num_workers = self.args.num_workers, pin_memory = True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size = self.args.batch_size, shuffle = False, num_workers = self.args.num_workers, pin_memory = True)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size = self.args.batch_size, shuffle = False, num_workers = self.args.num_workers, pin_memory = True)



class ROCOModel(pl.LightningModule):
    def __init__(self, args, model):
        super(ROCOModel,self).__init__()

        self.args = args
        self.model = model

    def training_step(self, batch, batch_idx):

        loss, acc = self.shared_step(batch, batch_idx)
        result = pl.TrainResult(loss)

        container = {'train_loss': loss, 'train_acc': acc}

        result.log_dict(container, on_step = True, on_epoch = True, prog_bar = True, logger = True)

        return result

    def validation_step(self, batch, batch_idx):

        loss, acc = self.shared_step(batch, batch_idx)
        result = pl.EvalResult(checkpoint_on = loss)

        container = {'val_loss': loss, 'val_acc': acc}        
        result.log_dict(container, on_step = True, on_epoch = True, prog_bar = True, logger = True)

        return result

    def shared_step(self, batch, batch_idx):

        img, caption_token, segment_ids, attention_mask, target = batch
        caption_token = caption_token.squeeze(1)
        attention_mask = attention_mask.squeeze(1)

        logits = self.model(img, caption_token, segment_ids, attention_mask)
        
        bool_label = target > 0
        pred = logits[bool_label, :].argmax(1)
        valid_labels = target[bool_label]  

        logits = logits.log_softmax(-1)  # (bs x seq_len x vocab_size)
        
        loss = self.loss_func(logits.permute(0,2,1), target)
        acc = (pred == valid_labels).type(torch.float).mean() * 100.

        return loss, acc

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr = self.args.lr)

        return [optimizer]

    def loss_func(self, pred, target):
        return nn.NLLLoss()(pred, target)


def calculate_bleu_score(preds,targets):
  bleu_per_answer = np.asarray([sentence_bleu([idx2ans[target].split()],idx2ans[pred].split()) for pred,target in zip(preds,targets)])
  return np.mean(bleu_per_answer)



class Embeddings(nn.Module):
    def __init__(self, args):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, 128, padding_idx=0)
        self.word_embeddings_2 = nn.Linear(128, args.hidden_size, bias=False)
        self.position_embeddings = nn.Embedding(args.max_position_embeddings, args.hidden_size)
        self.type_embeddings = nn.Embedding(3, args.hidden_size)
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.len = args.max_position_embeddings
    def forward(self, input_ids, segment_ids, position_ids=None):
        if position_ids is None:
            if torch.cuda.is_available():
                position_ids = torch.arange(self.len, dtype=torch.long).cuda()
            else:
                position_ids = torch.arange(self.len, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        words_embeddings = self.word_embeddings_2(words_embeddings)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.type_embeddings(segment_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


# class Transfer(nn.Module):
    # def __init__(self,args):
    #     super(Transfer, self).__init__()

    #     self.args = args
    #     self.model = models.resnet152(pretrained=True)
    #     # for p in self.parameters():
    #     #     p.requires_grad=False
    #     self.relu = nn.ReLU()
    #     self.conv2 = nn.Conv2d(2048, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     self.gap2 = nn.AdaptiveAvgPool2d((1,1))
    #     self.conv3 = nn.Conv2d(1024, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     self.gap3 = nn.AdaptiveAvgPool2d((1,1))
    #     self.conv4 = nn.Conv2d(512, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     self.gap4 = nn.AdaptiveAvgPool2d((1,1))
    #     self.conv5 = nn.Conv2d(256, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     self.gap5 = nn.AdaptiveAvgPool2d((1,1))
    #     self.conv7 = nn.Conv2d(64, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
    #     self.gap7 = nn.AdaptiveAvgPool2d((1,1))
    # def forward(self, img):
    #     modules2 = list(self.model.children())[:-2]
    #     fix2 = nn.Sequential(*modules2)
    #     v_2 = self.gap2(self.relu(self.conv2(fix2(img)))).view(-1,self.args.hidden_size)
    #     modules3 = list(self.model.children())[:-3]
    #     fix3 = nn.Sequential(*modules3)
    #     v_3 = self.gap3(self.relu(self.conv3(fix3(img)))).view(-1,self.args.hidden_size)
    #     modules4 = list(self.model.children())[:-4]
    #     fix4 = nn.Sequential(*modules4)
    #     v_4 = self.gap4(self.relu(self.conv4(fix4(img)))).view(-1,self.args.hidden_size)
    #     modules5 = list(self.model.children())[:-5]
    #     fix5 = nn.Sequential(*modules5)
    #     v_5 = self.gap5(self.relu(self.conv5(fix5(img)))).view(-1,self.args.hidden_size)
    #     modules7 = list(self.model.children())[:-7]
    #     fix7 = nn.Sequential(*modules7)
    #     v_7 = self.gap7(self.relu(self.conv7(fix7(img)))).view(-1,self.args.hidden_size)
    #     return v_2, v_3, v_4, v_5, v_7


class Transfer(nn.Module):
    def __init__(self,args):
        super(Transfer, self).__init__()

        self.args = args
        self.num_vis = args.num_vis
        # for p in self.parameters():
        #     p.requires_grad=False

        if self.num_vis == 5:


            if self.args.image_embedding == "vision":

                self.model1 = models.resnet152(pretrained=True)
                self.relu = nn.ReLU()

                self.conv21 = nn.Conv2d(196, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
                self.gap21 = nn.AdaptiveAvgPool2d((1,1))

                self.conv2 = nn.Conv2d(2048, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
                self.gap2 = nn.AdaptiveAvgPool2d((1,1))

                self.conv3 = nn.Conv2d(1024, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
                self.gap3 = nn.AdaptiveAvgPool2d((1,1))

                self.conv31 = nn.Conv2d(196, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
                self.gap31 = nn.AdaptiveAvgPool2d((1,1))

                self.conv4 = nn.Conv2d(512, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
                self.gap4 = nn.AdaptiveAvgPool2d((1,1))

                
                self.conv41 = nn.Conv2d(196, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
                self.gap41 = nn.AdaptiveAvgPool2d((1,1))

                self.conv5 = nn.Conv2d(256, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
                self.gap5 = nn.AdaptiveAvgPool2d((1,1))


                self.conv51 = nn.Conv2d(196, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
                self.gap51 = nn.AdaptiveAvgPool2d((1,1))
                
                self.conv7 = nn.Conv2d(64, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
                self.gap7 = nn.AdaptiveAvgPool2d((1,1))

                self.conv71 = nn.Conv2d(196, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
                self.gap71 = nn.AdaptiveAvgPool2d((1,1))

                self.model2 = \
                torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
            else :

                self.model = models.resnet152(pretrained=True)

                self.relu = nn.ReLU()
                self.conv2 = nn.Conv2d(2048, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
                self.gap2 = nn.AdaptiveAvgPool2d((1,1))
                self.conv3 = nn.Conv2d(1024, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
                self.gap3 = nn.AdaptiveAvgPool2d((1,1))
                self.conv4 = nn.Conv2d(512, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
                self.gap4 = nn.AdaptiveAvgPool2d((1,1))
                self.conv5 = nn.Conv2d(256, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
                self.gap5 = nn.AdaptiveAvgPool2d((1,1))
                self.conv7 = nn.Conv2d(64, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
                self.gap7 = nn.AdaptiveAvgPool2d((1,1))

        elif self.num_vis == 3:
            self.model = models.resnet152(pretrained=True)

            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(2048, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.gap2 = nn.AdaptiveAvgPool2d((1,1))
            self.conv3 = nn.Conv2d(1024, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.gap3 = nn.AdaptiveAvgPool2d((1,1))
            self.conv4 = nn.Conv2d(512, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.gap4 = nn.AdaptiveAvgPool2d((1,1))

        else:
            self.model = models.resnet152(pretrained=True)

            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(2048, args.hidden_size, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.gap2 = nn.AdaptiveAvgPool2d((1,1))            
            
    def forward(self, img):

        if self.num_vis == 5: 

            if self.args.image_embedding == "vision":
            
                modules2 = list(self.model1.children())[:-2]
                fix2 = nn.Sequential(*modules2)
                inter_2 = self.conv2(fix2(img))
                v_2 = self.gap2(self.relu(inter_2)).view(-1,self.args.hidden_size)

                modules21 = list(self.model2.children())[:]
                fix21 = nn.Sequential(*modules21)
                z=fix21(img)
                z=z.view(img.size()[0],196,10,100)
                z=self.conv21(z)
                inter_21=z
                z_relu=self.relu(z)
                z_gap = self.gap21(z_relu)
                v_21 = z_gap.view(img.size()[0],-1)
                v_2 = torch.add(v_2, v_21)
                
                modules3 = list(self.model1.children())[:-3] ### 3 ta laye be akhari 
                fix3 = nn.Sequential(*modules3)
                inter_3 = self.conv3(fix3(img))
                v_3 = self.gap3(self.relu(inter_3)).view(-1,self.args.hidden_size)

                modules31 = list(self.model2.children())[:-1]
                fix31 = nn.Sequential(*modules31)
                z=fix31(img)
                z=z.view(img.size()[0],196,24,32)
                z=self.conv31(z)
                inter_31=z
                z_relu=self.relu(z)
                z_gap = self.gap31(z_relu)
                v_31 = z_gap.view(img.size()[0],-1)
                v_2 = torch.add(v_3, v_31)

                modules4 = list(self.model1.children())[:-4]
                fix4 = nn.Sequential(*modules4)
                inter_4 = self.conv4(fix4(img))
                v_4 = self.gap4(self.relu(inter_4)).view(-1,self.args.hidden_size)

                modules41 = list(self.model2.children())[:-2]
                fix41 = nn.Sequential(*modules41)
                inter_41 = self.conv41(fix41(img).view(img.size()[0],196,24,32))
                v_41 = self.gap41(self.relu(inter_41)).view(-1,self.args.hidden_size)
                v_4 = torch.add(v_4, v_41)

                modules5 = list(self.model1.children())[:-5]
                fix5 = nn.Sequential(*modules5)
                inter_5 = self.conv5(fix5(img))
                v_5 = self.gap5(self.relu(inter_5)).view(-1,self.args.hidden_size)

                modules51 = list(self.model2.children())[:-3]
                fix51 = nn.Sequential(*modules5)
                inter_51 = self.conv51(fix51(img).view(img.size()[0],196,-1,32))
                v_51 = self.gap51(self.relu(inter_51)).view(-1,self.args.hidden_size)
                v_5 = torch.add(v_5, v_51)

                modules7 = list(self.model1.children())[:-7]
                fix7 = nn.Sequential(*modules7)
                inter_7 = self.conv7(fix7(img))
                v_7 = self.gap7(self.relu(inter_7)).view(-1,self.args.hidden_size)

                modules71 = list(self.model2.children())[:-4]
                fix71 = nn.Sequential(*modules71)
                inter_71 = self.conv71(fix71(img).view(img.size()[0],196,24,32))

                v_71 = self.gap7(self.relu(inter_71)).view(-1,self.args.hidden_size)
                v_7 = torch.add(v_7, v_71)
                
                return v_2, v_3, v_4, v_5, v_7, [inter_2.mean(1), inter_3.mean(1), inter_4.mean(1), inter_5.mean(1), inter_7.mean(1)]


            else:

                modules2 = list(self.model.children())[:-2]
                fix2 = nn.Sequential(*modules2)
                inter_2 = self.conv2(fix2(img))
                v_2 = self.gap2(self.relu(inter_2)).view(-1,self.args.hidden_size)
                modules3 = list(self.model.children())[:-3]
                fix3 = nn.Sequential(*modules3)
                inter_3 = self.conv3(fix3(img))
                v_3 = self.gap3(self.relu(inter_3)).view(-1,self.args.hidden_size)
                modules4 = list(self.model.children())[:-4]
                fix4 = nn.Sequential(*modules4)
                inter_4 = self.conv4(fix4(img))
                v_4 = self.gap4(self.relu(inter_4)).view(-1,self.args.hidden_size)
                modules5 = list(self.model.children())[:-5]
                fix5 = nn.Sequential(*modules5)
                inter_5 = self.conv5(fix5(img))
                v_5 = self.gap5(self.relu(inter_5)).view(-1,self.args.hidden_size)
                modules7 = list(self.model.children())[:-7]
                fix7 = nn.Sequential(*modules7)
                inter_7 = self.conv7(fix7(img))
                v_7 = self.gap7(self.relu(inter_7)).view(-1,self.args.hidden_size)

                return v_2, v_3, v_4, v_5, v_7, [inter_2.mean(1), inter_3.mean(1), inter_4.mean(1), inter_5.mean(1), inter_7.mean(1)]

        if self.num_vis == 3: 
            modules2 = list(self.model.children())[:-2]
            fix2 = nn.Sequential(*modules2)
            inter_2 = self.conv2(fix2(img))
            v_2 = self.gap2(self.relu(inter_2)).view(-1,self.args.hidden_size)
            modules3 = list(self.model.children())[:-3]
            fix3 = nn.Sequential(*modules3)
            inter_3 = self.conv3(fix3(img))
            v_3 = self.gap3(self.relu(inter_3)).view(-1,self.args.hidden_size)
            modules4 = list(self.model.children())[:-4]
            fix4 = nn.Sequential(*modules4)
            inter_4 = self.conv4(fix4(img))
            v_4 = self.gap4(self.relu(inter_4)).view(-1,self.args.hidden_size)

            return v_2, v_3, v_4, [inter_2.mean(1), inter_3.mean(1), inter_4.mean(1)]

        else:
            modules2 = list(self.model.children())[:-2]
            fix2 = nn.Sequential(*modules2)
            inter_2 = self.conv2(fix2(img))
            v_2 = self.gap2(self.relu(inter_2)).view(-1,self.args.hidden_size)    
            
            return v_2, [inter_2.mean(1)] 


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self,args):
        super(MultiHeadedSelfAttention,self).__init__()
        self.proj_q = nn.Linear(args.hidden_size, args.hidden_size)
        self.proj_k = nn.Linear(args.hidden_size, args.hidden_size)
        self.proj_v = nn.Linear(args.hidden_size, args.hidden_size)
        self.drop = nn.Dropout(args.hidden_dropout_prob)
        self.scores = None
        self.n_heads = args.heads
    def forward(self, x, mask):
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (self.split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        h = (scores @ v).transpose(1, 2).contiguous()
        h = self.merge_last(h, 2)
        self.scores = scores
        return h
    def split_last(self, x, shape):
        shape = list(shape)
        assert shape.count(-1) <= 1
        if -1 in shape:
            shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
        return x.view(*x.size()[:-1], *shape)
    def merge_last(self, x, n_dims):
        s = x.size()
        assert n_dims > 1 and n_dims < len(s)
        return x.view(*s[:-n_dims], -1)

class PositionWiseFeedForward(nn.Module):
    def __init__(self,args):
        super(PositionWiseFeedForward,self).__init__()
        self.fc1 = nn.Linear(args.hidden_size, args.hidden_size*4)
        self.fc2 = nn.Linear(args.hidden_size*4, args.hidden_size)
    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))

class BertLayer(nn.Module):
    def __init__(self,args, share='all', norm='pre'):
        super(BertLayer, self).__init__()
        self.share = share
        self.norm_pos = norm
        self.norm1 = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.norm2 = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.drop1 = nn.Dropout(args.hidden_dropout_prob)
        self.drop2 = nn.Dropout(args.hidden_dropout_prob)
        if self.share == 'ffn':
            self.attention = nn.ModuleList([MultiHeadedSelfAttention(args) for _ in range(args.n_layers)])
            self.proj = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size) for _ in range(args.n_layers)])
            self.feedforward = PositionWiseFeedForward(args)
        elif self.share == 'att':
            self.attention = MultiHeadedSelfAttention(args)
            self.proj = nn.Linear(args.hidden_size, args.hidden_size)
            self.feedforward = nn.ModuleList([PositionWiseFeedForward(args) for _ in range(args.n_layers)])
        elif self.share == 'all':
            self.attention = MultiHeadedSelfAttention(args)
            self.proj = nn.Linear(args.hidden_size, args.hidden_size)
            self.feedforward = PositionWiseFeedForward(args)
        elif self.share == 'none':
            self.attention = nn.ModuleList([MultiHeadedSelfAttention(args) for _ in range(args.n_layers)])
            self.proj = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size) for _ in range(args.n_layers)])
            self.feedforward = nn.ModuleList([PositionWiseFeedForward(args) for _ in range(args.n_layers)])
    def forward(self, hidden_states, attention_mask, layer_num):
        if self.norm_pos == 'pre':
            if isinstance(self.attention, nn.ModuleList):
                h = self.proj[layer_num](self.attention[layer_num](self.norm1(hidden_states), attention_mask))
            else:
                h = self.proj(self.attention(self.norm1(hidden_states), attention_mask))
            out = hidden_states + self.drop1(h)
            if isinstance(self.feedforward, nn.ModuleList):
                h = self.feedforward[layer_num](self.norm1(out))
            else:
                h = self.feedforward(self.norm1(out))
            out = out + self.drop2(h)
        if self.norm_pos == 'post':
            if isinstance(self.attention, nn.ModuleList):
                h = self.proj[layer_num](self.attention[layer_num](hidden_states, attention_mask))
            else:
                h = self.proj(self.attention(hidden_states, attention_mask))
            out = self.norm1(hidden_states + self.drop1(h))
            if isinstance(self.feedforward, nn.ModuleList):
                h = self.feedforward[layer_num](out)
            else:
                h = self.feedforward(out)
            out = self.norm2(out + self.drop2(h))
        return out





class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer,self).__init__()
#         base_model = BertModel.from_pretrained('bert-base-uncased')
#         bert_model = nn.Sequential(*list(base_model.children())[0:])
#         self.bert_embedding = bert_model[0]
# #         self.embed = Embeddings(args)
#         self.trans = Transfer(args)
#         self.blocks = BertLayer(args,share='none', norm='pre')
#         self.n_layers = args.n_layers

        base_model = BertModel.from_pretrained('bert-base-uncased')
        bert_model = nn.Sequential(*list(base_model.children())[0:])
        self.bert_embedding = bert_model[0]
        # self.embed = Embeddings(args)
        self.num_vis = args.num_vis
        self.trans = Transfer(args)
        self.blocks = BertLayer(args,share='none', norm='pre')
        self.n_layers = args.n_layers
        
    def forward(self, img, input_ids, token_type_ids, mask):
        h = self.bert_embedding(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=None)

        if self.num_vis==5:
            v_2, v_3, v_4, v_5, v_7, intermediate = self.trans(img)
        elif self.num_vis==3:
            v_2, v_3, v_4, intermediate = self.trans(img)
        else:
            v_2, intermediate = self.trans(img)
        # h = self.embed(input_ids, token_type_ids)

        if self.num_vis == 5:
            for i in range(len(h)):
                h[i][1] = v_2[i]
            for i in range(len(h)):
                h[i][2] = v_3[i]
            for i in range(len(h)):
                h[i][3] = v_4[i]
            for i in range(len(h)):
                h[i][4] = v_5[i]
            for i in range(len(h)):
                h[i][5] = v_7[i]

        elif self.num_vis == 3:
            for i in range(len(h)):
                h[i][1] = v_2[i]
            for i in range(len(h)):
                h[i][2] = v_3[i]
            for i in range(len(h)):
                h[i][3] = v_4[i]

        else:
            for i in range(len(h)):
                h[i][1] = v_2[i]


        for i in range(self.n_layers):
            h = self.blocks(h, mask, i)
        return h ,intermediate
        


class Model(nn.Module):
    def __init__(self,args):
        super(Model,self).__init__()
        self.transformer = Transformer(args)
        self.fc1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Sequential(nn.Linear(args.hidden_size, args.hidden_size),
                                        nn.LayerNorm(args.hidden_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(args.hidden_size, args.vocab_size))
    def forward(self, img, input_ids, segment_ids, input_mask):
        h,intermediate = self.transformer(img, input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc1(h))
        logits = self.classifier(pooled_h)
        return logits,None,intermediate


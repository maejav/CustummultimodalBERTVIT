import os
import numpy as np
import pandas as pd
import random
import math
import cv2
import timm 
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


import torch
from torchvision import transforms, models
from torch.cuda.amp import GradScaler
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from PIL import Image
from random import choice
import matplotlib.pyplot as plt

import pretrainedmodels
 
##  seed every thing 

### Function that sets seed for pseudo-random number generators in

###Second, you can configure PyTorch to avoid using nondeterministic algorithms for some operations, 
###so that multiple calls to those operations, given the same inputs, will produce the same result.
"""

reproducible results are not guaranteed across PyTorch releases, individual commits, or 
different platforms. Furthermore, 
results may not be reproducible between CPU and GPU executions, 
even when using identical seeds.

"""
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



# def make_df(file_path):
#     paths = os.listdir(file_path)
    
#     df_list = []
    
#     for p in paths:
#         df = pd.read_csv(os.path.join(file_path, p), sep='|', names = ['img_id', 'question', 'answer'])
#         df['category'] = p.split('_')[1]  ## column
#         df['mode'] = p.split('_')[2][:-4]
#         df_list.append(df)
    
#     return pd.concat(df_list)

def load_data(args, remove = None):
    testdf = pd.read_csv(os.path.join(args.data_dir, 'testdf.csv'))
    testdf = testdf.loc[:, ~testdf.columns.str.contains('^Unnamed')]
    testdf['img_id'] = testdf['img_id'].apply(lambda x: os.path.join(args.data_dir, 'ImageClef-2019-VQA-Med-Test/VQAMed2019_Test_Images', x + '.jpg'))


    if args.category == "all" :
        # print(os.getcwd())
        traindf = pd.read_csv(os.path.join(args.data_dir, 'traindf.csv'))
        valdf = pd.read_csv(os.path.join(args.data_dir, 'valdf.csv'))

        traindf = traindf.loc[:, ~traindf.columns.str.contains('^Unnamed')]
        valdf = valdf.loc[:, ~valdf.columns.str.contains('^Unnamed')]


        if remove is not None:
            traindf = traindf[~traindf['img_id'].isin(remove)].reset_index(drop=True)

        traindf['img_id'] = traindf['img_id'].apply(lambda x: os.path.join(args.data_dir, 'ImageClef-2019-VQA-Med-Training/Train_images', x + '.jpg'))

        valdf['img_id'] = valdf['img_id'].apply(lambda x: os.path.join(args.data_dir, 'ImageClef-2019-VQA-Med-Validation/Val_images', x + '.jpg'))

    else:
        # print("goooooooooooooooooo")
        l= (os.path.join(args.data_dir, 'ImageClef-2019-VQA-Med-Training/QAPairsByCategory/'))
        paths= os.listdir(l)
        for path in paths:
            if path.split("_")[1]== args.category:
                traindf = pd.read_csv(f"{l}/{path}", sep="|", header=None, names=["img_id", "question","answer"])
                traindf["category"]=args.category.lower()
                # print("read train ppppppppppppppppppppppppppppsucsee")
                traindf["mode"]="train"
                break
        traindf['img_id'] = traindf['img_id'].apply(lambda x: os.path.join(args.data_dir, 'ImageClef-2019-VQA-Med-Training/Train_images', x + '.jpg'))
        # print("llllllllllllllllllllllllllllllllllllllllllll",traindf.columns)
        # print("goooooooooooooooooo")
        l= (os.path.join(args.data_dir, 'ImageClef-2019-VQA-Med-Validation/QAPairsByCategory/'))
        paths= os.listdir(l)
        for path in paths:
            if path.split("_")[1]== args.category:
                # print("sssssssssssssssssssssssssssssssssssssssssssssssssssusss")
                valdf= pd.read_csv(f"{l}/{path}", sep="|", header=None, names=["img_id", "question","answer"])
                valdf["category"]=args.category.lower()
                
                valdf["mode"]="eval"
                break
        valdf['img_id'] = valdf['img_id'].apply(lambda x: os.path.join(args.data_dir, 'ImageClef-2019-VQA-Med-Validation/Val_images', x + '.jpg'))
        when= testdf['category']== args.category.lower() 
        testdf= testdf[when]
        # print("length of test dfffffffffffff",len(testdf))
        # testdf=testdf.drop(['category'], axis=1)


    traindf['category'] = traindf['category'].str.lower()
    traindf['answer'] = traindf['answer'].str.lower()

    valdf['category'] = valdf['category'].str.lower()
    valdf['answer'] = valdf['answer'].str.lower()

    testdf['answer'] = testdf['answer'].str.lower()
    testdf['category'] = testdf['category'].str.lower()



    traindf = traindf.sample(frac = args.train_pct)
    valdf = valdf.sample(frac = args.valid_pct)
    testdf = testdf.sample(frac = args.test_pct)
    # print("tttttttttttttttt\n",traindf.head())
    # print("tttttttttttttttt\n",valdf.head())
    # print("tttttttttttttttt\n",testdf.head())
    
    return traindf, valdf, testdf

     



#Utils
def gelu(x):  ### Computes the error function of input
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))  ## square root of a number 


def encode_text(caption,tokenizer, args):
    part1 = [0 for _ in range(5)]
    #get token ids and remove [CLS] and [SEP] token id
    part2 = tokenizer.encode(caption)[1:-1]

    tokens = [tokenizer.cls_token_id] + part1 + [tokenizer.sep_token_id] + part2[:args.max_position_embeddings-8] + [tokenizer.sep_token_id]
    segment_ids = [0]*(len(part1)+2) + [1]*(len(part2[:args.max_position_embeddings-8])+1)
    input_mask = [1]*len(tokens)
    n_pad = args.max_position_embeddings - len(tokens)
    tokens.extend([0]*n_pad)
    segment_ids.extend([0]*n_pad)
    input_mask.extend([0]*n_pad)

    
    return tokens, segment_ids, input_mask

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


class LabelSmoothing(nn.Module):
    def __init__(self, smoothing = 0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim = -1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
    
            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)

def crop(img):
    c_y, c_x = img.shape[:2]
    c_y = c_y // 2
    c_x = c_x // 2
    shorter = min(img.shape[:2])
    if img.shape[0] <= img.shape[1]:
        img = img[c_y - shorter // 2: c_y + (shorter - shorter // 2) - 20, c_x - shorter // 2: c_x + (shorter - shorter // 2), :]
    else:
        img = img[c_y - shorter // 2: c_y + (shorter - shorter // 2), c_x - shorter // 2: c_x + (shorter - shorter // 2), :]
    
    return img


class VQAMed(Dataset):
    def __init__(self, df, imgsize, tfm, args, mode = 'train'):
        self.df = df
        self.tfm = tfm
        self.size = imgsize
        self.args = args
        self.tokenizer =BertTokenizer.from_pretrained(args.bert_model) ## it takes time??? 
        ##for two model
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # print(idx , "uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuu")

        path = self.df.loc[idx,'img_id']
        question = self.df.loc[idx, 'question']
 
        answer = self.df.loc[idx, 'answer']

        if self.mode == 'eval':
            tok_ques = self.tokenizer.tokenize(question)

        if self.args.smoothing:
            answer = onehot(self.args.num_classes, answer)

        img = cv2.imread(path)
  

        if self.tfm:
            img = self.tfm(img)
            
        tokens, segment_ids, input_mask= encode_text(question, self.tokenizer, self.args)


        return img, torch.tensor(tokens, dtype = torch.long), torch.tensor(segment_ids, dtype = torch.long), torch.tensor(input_mask, dtype = torch.long), torch.tensor(answer, dtype = torch.long), path


# class VQAMed_Binary(Dataset):
#     def __init__(self, df, imgsize, tfm, args, mode = 'train'):
#         self.df = df.values
#         self.tfm = tfm
#         self.size = imgsize
#         self.args = args
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         self.mode = mode

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         path = self.df[idx,0]

#         question = self.df[idx, 1]

#         if self.mode != 'test':
#             answer = self.df[idx, 2]

#         if self.mode == 'eval':
#             tok_ques = self.tokenizer.tokenize(question)

#         if self.args.smoothing:
#             answer = onehot(self.args.num_classes, answer)

#         img = cv2.imread(path)


#         if self.tfm:
#             img = self.tfm(image = img)['image']
            
#         tokens, segment_ids, input_mask= encode_text(question, self.tokenizer, self.args)

#         if self.args.smoothing:
#             return img, torch.tensor(tokens, dtype = torch.long), torch.tensor(segment_ids, dtype = torch.long) ,torch.tensor(input_mask, dtype = torch.long), answer
#         else:
#             if self.mode == 'train':
#                 return img, torch.tensor(tokens, dtype = torch.long), torch.tensor(segment_ids, dtype = torch.long) ,torch.tensor(input_mask, dtype = torch.long), torch.tensor(answer, dtype = torch.long), path
#             elif self.mode == 'test':
#                 return img, torch.tensor(tokens, dtype = torch.long), torch.tensor(segment_ids, dtype = torch.long) ,torch.tensor(input_mask, dtype = torch.long)
#             else:
#                 return img, torch.tensor(tokens, dtype = torch.long), torch.tensor(segment_ids, dtype = torch.long) ,torch.tensor(input_mask, dtype = torch.long), torch.tensor(answer, dtype = torch.long), tok_ques



# class Model_Keyword(nn.Module):
#     def __init__(self, num_classes):
#         super(Model_Keyword, self).__init__()
#         self.model = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=1000, pretrained='imagenet')
#         last_in = self.model.last_linear.in_features
#         self.model.last_linear = nn.Identity()
#         self.embed = nn.Embedding(3, last_in)
#         self.last_layer = nn.Linear(2 * last_in, num_classes)

#     def forward(self, img, keyword):

#         img_feat = self.model(img)
#         key_feat = self.embed(keyword)

#         feat = torch.cat([img_feat, key_feat], -1)

#         logits = self.last_layer(feat)

#         return logits



def calculate_bleu_score(preds,targets, idx2ans):
    # print ("PRED FOR ",preds)
    # print("TARGEt",targets)
    bleu_per_answer = np.asarray([sentence_bleu([idx2ans[target].split()],idx2ans[pred].split(), weights = [1]) for pred,target in zip(preds,targets)])
    # print("bbbbbbbbblue accuracy ", bleu_per_answer)
    
    return np.mean(bleu_per_answer) if len(bleu_per_answer)>0 else torch.tensor(0.)



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


class Transfer(nn.Module):
    def __init__(self,args):
        super(Transfer, self).__init__()

        self.args = args
        self.num_vis = args.num_vis
        
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

                v_71 = self.gap71(self.relu(inter_71)).view(-1,self.args.hidden_size)
                v_7 = torch.add(v_7, v_71)
                return v_2, v_3, v_4, v_5, v_7, [inter_2.mean(1), inter_3.mean(1), inter_4.mean(1), inter_5.mean(1), inter_7.mean(1)]

                # return v_2, v_3, v_4, v_5, v_7,[inter_2.mean(1)+inter_21.mean(1), inter_3.mean(1)+inter_31.mean(1), inter_4.mean(1)+inter_41.mean(1), inter_5.mean(1)+inter_51.mean(1), inter_7.mean(1)+inter_71.mean(1)]


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
        return h, scores
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
                attn_output, attn_scores = self.attention[layer_num](self.norm1(hidden_states), attention_mask)
                h = self.proj[layer_num](attn_output)
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
        return out, attn_scores

class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer,self).__init__()
        base_model = BertModel.from_pretrained('bert-base-uncased')
        bert_model = nn.Sequential(*list(base_model.children())[0:])
        self.bert_embedding = bert_model[0]
        # self.embed = Embeddings(args)
        self.num_vis = args.num_vis
        self.trans = Transfer(args)
        self.blocks = BertLayer(args,share='none', norm='pre')
        self.n_layers = args.n_layers
        
    def forward(self, img, input_ids, token_type_ids, mask):

        if self.num_vis==5:
            v_2, v_3, v_4, v_5, v_7, intermediate = self.trans(img)
        elif self.num_vis==3:
            v_2, v_3, v_4, intermediate = self.trans(img)
        else:
            v_2, intermediate = self.trans(img)
        # h = self.embed(input_ids, token_type_ids)
        h = self.bert_embedding(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=None)

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


        hidden_states = []
        all_attn_scores = []
        for i in range(self.n_layers):
            h, attn_scores = self.blocks(h, mask, i)
            hidden_states.append(h)
            all_attn_scores.append(attn_scores)

        return torch.stack(hidden_states, 0), torch.stack(all_attn_scores, 0), intermediate


class Model(nn.Module):
    def __init__(self,args):
        super(Model,self).__init__()
        self.args = args
        self.transformer = Transformer(args)
        self.fc1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Sequential(nn.Linear(args.hidden_size, args.hidden_size),
                                        nn.LayerNorm(args.hidden_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(args.hidden_size, args.vocab_size))
    def forward(self, img, input_ids, segment_ids, input_mask):
        h, attn_scores, intermediate = self.transformer(img, input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc1(h[-1].mean(1)))
        logits = self.classifier(pooled_h)
        return logits, attn_scores, intermediate






def train_one_epoch(loader, model, optimizer, criterion, device, scaler, args, idx2ans):

    model.train()
    train_loss = []
    IMGIDS = []
    PREDS = []
    TARGETS = []
    bar = tqdm(loader, leave = False)
    for (img, question_token,segment_ids,attention_mask,target, imgid) in bar:
        
        img, question_token,segment_ids,attention_mask,target = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
        question_token = question_token.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        loss_func = criterion
        optimizer.zero_grad()

        if args.mixed_precision:
            with torch.cuda.amp.autocast(): 
                logits, _, _ = model(img, question_token, segment_ids, attention_mask)
                loss = loss_func(logits, target)
        else:
            logits, _, _ = model(img, question_token, segment_ids, attention_mask)
            loss = loss_func(logits, target)

        if args.mixed_precision:
            scaler.scale(loss)
            loss.backward()

            if args.clip:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if args.clip:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
            optimizer.step()

        if args.smoothing:
            TARGETS.append(target.argmax(1))
        else:
            TARGETS.append(target)    

        pred = logits.softmax(1).argmax(1).detach()
        PREDS.append(pred)
        IMGIDS.append(imgid)

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        bar.set_description('train_loss: %.5f' % (loss_np))

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    IMGIDS = [i for sub in IMGIDS for i in sub]

    acc = (PREDS == TARGETS).mean() * 100.
    bleu = calculate_bleu_score(PREDS,TARGETS,idx2ans)
    print("innnnnnnnnnnnntrainlooooss ",np.mean(train_loss))
    print("iinnnnnnnnnnnnnnnnnnnnntrainaccc ", acc)

    return np.mean(train_loss), PREDS, acc, bleu, IMGIDS

def validate(loader, model, criterion, device, scaler, args, val_df, idx2ans):

    model.eval()
    val_loss = []

    PREDS = []
    TARGETS = []
    bar = tqdm(loader, leave=False)

    with torch.no_grad():
        for (img, question_token,segment_ids,attention_mask,target, _) in bar:

            img, question_token,segment_ids,attention_mask,target = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
            question_token = question_token.squeeze(1)
            attention_mask = attention_mask.squeeze(1)


            if args.mixed_precision:
                with torch.cuda.amp.autocast(): 
                    logits, _, _ = model(img, question_token, segment_ids, attention_mask)
                    loss = criterion(logits, target)
            else:
                logits, _ , _= model(img, question_token, segment_ids, attention_mask)
                loss = criterion(logits, target)


            loss_np = loss.detach().cpu().numpy()

            pred = logits.softmax(1).argmax(1).detach()

            PREDS.append(pred)

            if args.smoothing:
                TARGETS.append(target.argmax(1))
            else:
                TARGETS.append(target)

            val_loss.append(loss_np)

            bar.set_description('val_loss: %.5f' % (loss_np))

        val_loss = np.mean(val_loss)

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    # Calculate total and category wise accuracy
    if not args.category == "all":
        acc = (PREDS == TARGETS).mean() * 100.
        print("vallllacc",acc)
        bleu = calculate_bleu_score(PREDS,TARGETS,idx2ans)
    else:
        total_acc = (PREDS == TARGETS).mean() * 100.
        # binary_acc = (PREDS[val_df['category']=='binary'] == TARGETS[val_df['category']=='binary']).mean() * 100.
        plane_acc = (PREDS[val_df['category']=='plane'] == TARGETS[val_df['category']=='plane']).mean() * 100 if len(PREDS[val_df['category']=='plane'] == TARGETS[val_df['category']=='plane']) > 0 else torch.tensor(0.)
        # torch.mean(a_masked) if len(a_masked) > 0 else torch.tensor(0.)
        # if len(PREDS[val_df['category']=='plane'] == TARGETS[val_df['category']=='plane']) > 0:
        #     print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
        organ_acc = (PREDS[val_df['category']=='organ'] == TARGETS[val_df['category']=='organ']).mean() * 100 if len(PREDS[val_df['category']=='organ'] == TARGETS[val_df['category']=='organ'])>0   else torch.tensor(0.)  
        modality_acc = (PREDS[val_df['category']=='modality'] == TARGETS[val_df['category']=='modality']).mean() * 100. if len(PREDS[val_df['category']=='modality'] == TARGETS[val_df['category']=='modality'])>0 else torch.tensor(0.) 
        abnorm_acc = (PREDS[val_df['category']=='abnormality'] == TARGETS[val_df['category']=='abnormality']).mean() * 100. if len(PREDS[val_df['category']=='abnormality'] == TARGETS[val_df['category']=='abnormality'])>0 else torch.tensor(0.)

        acc = {'val_total_acc': np.round(total_acc, 4), 'val_plane_acc': np.round(plane_acc, 4), 'val_organ_acc': np.round(organ_acc, 4), 
               'val_modality_acc': np.round(modality_acc, 4), 'val_abnorm_acc': np.round(abnorm_acc, 4)}
        # bleu={}
        # add bleu score code
        total_bleu = calculate_bleu_score(PREDS,TARGETS,idx2ans)
        plane_bleu = calculate_bleu_score(PREDS[val_df['category']=='plane'],TARGETS[val_df['category']=='plane'],idx2ans)
        # binary_bleu = calculate_bleu_score(PREDS[val_df['category']=='binary'],TARGETS[val_df['category']=='binary'],idx2ans)
        organ_bleu = calculate_bleu_score(PREDS[val_df['category']=='organ'],TARGETS[val_df['category']=='organ'],idx2ans)
        modality_bleu = calculate_bleu_score(PREDS[val_df['category']=='modality'],TARGETS[val_df['category']=='modality'],idx2ans)
        abnorm_bleu = calculate_bleu_score(PREDS[val_df['category']=='abnormality'],TARGETS[val_df['category']=='abnormality'],idx2ans)


        bleu = {'val_total_bleu': np.round(total_bleu, 4), 'val_plane_bleu': np.round(plane_bleu, 4), 'val_organ_bleu': np.round(organ_bleu, 4), 
            'val_modality_bleu': np.round(modality_bleu, 4), 'val_abnorm_bleu': np.round(abnorm_bleu, 4)}
    print("valllloss", val_loss)
    return val_loss, PREDS, acc, bleu  
    
def test(loader, model, criterion, device, scaler, args, val_df,idx2ans):

    model.eval()

    PREDS = []
    TARGETS = []

    test_loss = []

    with torch.no_grad():
        for (img,question_token,segment_ids,attention_mask,target, _) in tqdm(loader, leave=False):

            img, question_token, segment_ids, attention_mask, target = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
            question_token = question_token.squeeze(1)
            attention_mask = attention_mask.squeeze(1)
            
            if args.mixed_precision:
                with torch.cuda.amp.autocast(): 
                    logits, _, _ = model(img, question_token, segment_ids, attention_mask)
                    loss = criterion(logits, target)
            else:
                logits, _, _ = model(img, question_token, segment_ids, attention_mask)
                loss = criterion(logits, target)


            loss_np = loss.detach().cpu().numpy()

            test_loss.append(loss_np)

            pred = logits.softmax(1).argmax(1).detach()
            
            PREDS.append(pred)

            if args.smoothing:
                TARGETS.append(target.argmax(1))
            else:
                TARGETS.append(target)

        test_loss = np.mean(test_loss)

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    if  not args.category == "all":
        acc = (PREDS == TARGETS).mean() * 100.
        print("testtttaccc",acc)
        bleu = calculate_bleu_score(PREDS,TARGETS,idx2ans)
    else:
        total_acc = (PREDS == TARGETS).mean() * 100.
        # binary_acc = (PREDS[val_df['category']=='binary'] == TARGETS[val_df['category']=='binary']).mean() * 100.
        plane_acc = (PREDS[val_df['category']=='plane'] == TARGETS[val_df['category']=='plane']).mean() * 100 if len(PREDS[val_df['category']=='plane'] == TARGETS[val_df['category']=='plane']) > 0 else torch.tensor(0.)
        # torch.mean(a_masked) if len(a_masked) > 0 else torch.tensor(0.)
        # if len(PREDS[val_df['category']=='plane'] == TARGETS[val_df['category']=='plane']) > 0:
            # print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
        organ_acc = (PREDS[val_df['category']=='organ'] == TARGETS[val_df['category']=='organ']).mean() * 100 if len(PREDS[val_df['category']=='organ'] == TARGETS[val_df['category']=='organ'])>0   else torch.tensor(0.)  
        modality_acc = (PREDS[val_df['category']=='modality'] == TARGETS[val_df['category']=='modality']).mean() * 100. if len(PREDS[val_df['category']=='modality'] == TARGETS[val_df['category']=='modality'])>0 else torch.tensor(0.) 
        abnorm_acc = (PREDS[val_df['category']=='abnormality'] == TARGETS[val_df['category']=='abnormality']).mean() * 100. if len(PREDS[val_df['category']=='abnormality'] == TARGETS[val_df['category']=='abnormality'])>0 else torch.tensor(0.)

        acc = {'total_acc': np.round(total_acc, 4),  'plane_acc': np.round(plane_acc, 4), 'organ_acc': np.round(organ_acc, 4), 
               'modality_acc': np.round(modality_acc, 4), 'abnorm_acc': np.round(abnorm_acc, 4)}

        # add bleu score code
        total_bleu = calculate_bleu_score(PREDS,TARGETS,idx2ans)
        # binary_bleu = calculate_bleu_score(PREDS[val_df['category']=='binary'],TARGETS[val_df['category']=='binary'],idx2ans)
        plane_bleu = calculate_bleu_score(PREDS[val_df['category']=='plane'],TARGETS[val_df['category']=='plane'],idx2ans)
        organ_bleu = calculate_bleu_score(PREDS[val_df['category']=='organ'],TARGETS[val_df['category']=='organ'],idx2ans)
        modality_bleu = calculate_bleu_score(PREDS[val_df['category']=='modality'],TARGETS[val_df['category']=='modality'],idx2ans)
        abnorm_bleu = calculate_bleu_score(PREDS[val_df['category']=='abnormality'],TARGETS[val_df['category']=='abnormality'],idx2ans)


        bleu = {'total_bleu': np.round(total_bleu, 4),   'plane_bleu': np.round(plane_bleu, 4), 'organ_bleu': np.round(organ_bleu, 4), 
            'modality_bleu': np.round(modality_bleu, 4), 'abnorm_bleu': np.round(abnorm_bleu, 4)}

    print("tessstllllllllooss",test_loss)
    return test_loss, PREDS, acc, bleu

def final_test(loader, all_models, device, args, val_df, idx2ans):

    PREDS = []

    with torch.no_grad():
        for (img,question_token,segment_ids,attention_mask,target) in tqdm(loader, leave=False):

            img, question_token, segment_ids, attention_mask, target = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
            question_token = question_token.squeeze(1)
            attention_mask = attention_mask.squeeze(1)
            
            for i, model in enumerate(all_models):
                if args.mixed_precision:
                    with torch.cuda.amp.autocast(): 
                        logits, _, _ = model(img, question_token, segment_ids, attention_mask)
                else:
                    logits, _, _ = model(img, question_token, segment_ids, attention_mask)
 
                if i == 0:
                    pred = logits.detach().cpu().numpy()/len(all_models)
                else:
                    pred += logits.detach().cpu().numpy()/len(all_models)
            
            PREDS.append(pred)

    PREDS = np.concatenate(PREDS)

    return PREDS

def val_img_only(loader, model, criterion, device, scaler, args, val_df, idx2ans):

    model.eval()
    val_loss = []

    PREDS = []
    TARGETS = []
    bar = tqdm(loader, leave=False)

    with torch.no_grad():
        for (img, question_token,segment_ids,attention_mask,target, _) in bar:

            img, question_token,segment_ids,attention_mask,target = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
            # question_token = question_token.squeeze(1)
            # attention_mask = attention_mask.squeeze(1)


            if args.mixed_precision:
                with torch.cuda.amp.autocast(): 
                    logits = model(img)
                    loss = criterion(logits, target)
            else:
                logits = model(img)
                loss = criterion(logits, target)


            loss_np = loss.detach().cpu().numpy()

            pred = logits.softmax(1).argmax(1).detach()

            PREDS.append(pred)

            if args.smoothing:
                TARGETS.append(target.argmax(1))
            else:
                TARGETS.append(target)

            val_loss.append(loss_np)

            bar.set_description('val_loss: %.5f' % (loss_np))

        val_loss = np.mean(val_loss)

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    acc = (PREDS == TARGETS).mean() * 100.
    bleu = calculate_bleu_score(PREDS,TARGETS,idx2ans)



    return val_loss, PREDS, acc, bleu

def test_img_only(loader, model, criterion, device, scaler, args, test_df, idx2ans):

    model.eval()
    TARGETS = []
    PREDS = []
    test_loss = []

    with torch.no_grad():
        for (img, question_token, segment_ids, attention_mask, target, _) in tqdm(loader, leave=False):

            img, question_token, segment_ids, attention_mask, target = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
            # question_token = question_token.squeeze(1)
            # attention_mask = attention_mask.squeeze(1)
            
            if args.mixed_precision:
                with torch.cuda.amp.autocast(): 
                    logits = model(img)
                    loss = criterion(logits, target)
            else:
                logits = model(img)
                loss = criterion(logits, target)


            pred = logits.softmax(1).argmax(1).detach()
            loss_np = loss.detach().cpu().numpy()
            
            PREDS.append(pred)
            TARGETS.append(target)
            test_loss.append(loss_np)

        test_loss = np.mean(test_loss)

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    acc = (PREDS == TARGETS).mean() * 100.
    bleu = calculate_bleu_score(PREDS,TARGETS,idx2ans)

    return test_loss, PREDS, acc, bleu

def train_img_only(loader, model, optimizer, criterion, device, scaler, args, idx2ans):

    model.train()
    train_loss = []
    PREDS = []
    TARGETS = []
    IMGIDS = []
    bar = tqdm(loader, leave = False)
    for (img, question_token,segment_ids,attention_mask,target, imgid) in bar:
        
        img, question_token,segment_ids,attention_mask,target = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
        # question_token = question_token.squeeze(1)
        # attention_mask = attention_mask.squeeze(1)
        loss_func = criterion
        optimizer.zero_grad()

        if args.mixed_precision:
            with torch.cuda.amp.autocast(): 
                logits = model(img)
                loss = loss_func(logits, target)
        else:
            logits = model(img)
            loss = loss_func(logits, target)

        if args.mixed_precision:
            scaler.scale(loss)
            loss.backward()

            if args.clip:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if args.clip:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
            optimizer.step()

        if args.smoothing:
            TARGETS.append(target.argmax(1))
        else:
            TARGETS.append(target)    

        pred = logits.softmax(1).argmax(1).detach()
        PREDS.append(pred)

        IMGIDS.append(imgid)
        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        bar.set_description('train_loss: %.5f' % (loss_np))

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    IMGIDS = [i for sub in IMGIDS for i in sub]

    acc = (PREDS == TARGETS).mean() * 100.
    bleu = calculate_bleu_score(PREDS,TARGETS,idx2ans)

    return np.mean(train_loss), PREDS, acc, bleu, IMGIDS

def train_binary(loader, model, optimizer, criterion, device, scaler, args, idx2ans):

    model.train()
    train_loss = []
    PREDS = []
    TARGETS = []
    IMGIDS = []
    bar = tqdm(loader, leave = False)
    for (img, question_token,segment_ids,attention_mask,target, imgid) in bar:
        
        img, question_token,segment_ids,attention_mask,target = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
        question_token = question_token.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        loss_func = criterion
        optimizer.zero_grad()

        if args.mixed_precision:
            with torch.cuda.amp.autocast(): 
                # logits = model(img)
                logits, _, _ = model(img, question_token, segment_ids, attention_mask)
                loss = loss_func(logits, target)
        else:
            # logits = model(img)
            logits, _, _ = model(img, question_token, segment_ids, attention_mask)
            loss = loss_func(logits, target)

        if args.mixed_precision:
            scaler.scale(loss)
            loss.backward()

            if args.clip:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if args.clip:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
            optimizer.step()

        if args.smoothing:
            TARGETS.append(target.argmax(1))
        else:
            TARGETS.append(target)    

        pred = logits.softmax(1).argmax(1).detach()
        PREDS.append(pred)

        IMGIDS.append(imgid)
        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        bar.set_description('train_loss: %.5f' % (loss_np))

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()
    IMGIDS = [i for sub in IMGIDS for i in sub]

    acc = (PREDS == TARGETS).mean() * 100.

    return np.mean(train_loss), PREDS, acc, IMGIDS

def val_binary(loader, model, criterion, device, scaler, args, val_df, idx2ans):

    model.eval()
    val_loss = []

    PREDS = []
    TARGETS = []
    bar = tqdm(loader, leave=False)

    with torch.no_grad():
        for (img, question_token,segment_ids,attention_mask,target, _) in bar:

            img, question_token,segment_ids,attention_mask,target = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
            question_token = question_token.squeeze(1)
            attention_mask = attention_mask.squeeze(1)


            if args.mixed_precision:
                with torch.cuda.amp.autocast(): 
                    # logits = model(img)
                    logits, _, _ = model(img, question_token, segment_ids, attention_mask)
                    loss = criterion(logits, target)
            else:
                # logits = model(img)
                logits, _, _ = model(img, question_token, segment_ids, attention_mask)
                loss = criterion(logits, target)


            loss_np = loss.detach().cpu().numpy()

            pred = logits.softmax(1).argmax(1).detach()

            PREDS.append(pred)

            if args.smoothing:
                TARGETS.append(target.argmax(1))
            else:
                TARGETS.append(target)

            val_loss.append(loss_np)

            bar.set_description('val_loss: %.5f' % (loss_np))

        val_loss = np.mean(val_loss)

    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    acc = (PREDS == TARGETS).mean() * 100.

    return val_loss, PREDS, acc
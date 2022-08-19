from ast import Not
import os
import numpy as np
import pandas as pd
import random
import math
import json
from PIL import Image
import torch
import timm
import requests
import torchvision.transforms as transforms
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
"""
BERT is a model with absolute position embeddings so it’s 
usually advised to pad the inputs on the right rather than the left.????

"""
from transformers import DistilBertModel, DistilBertConfig ,DistilBertTokenizer
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from PIL import Image
from random import choice
import matplotlib.pyplot as plt
import cv2



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

### it is unnecessary

# def make_df(file_path):
#     paths = os.listdir(file_path)
    
#     df_list = []
    
#     for p in paths:
#         df = pd.read_csv(os.path.join(file_path, p), sep='|', names = ['img_id', 'question', 'answer'])
#         df['category'] = p.split('_')[1]
#         df['mode'] = p.split('_')[2][:-4]
#         df_list.append(df)
    
#     return pd.concat(df_list)

def load_data(args):
    
    train_file = open(os.path.join(args.data_dir,'littestset.json'),)
    test_file = open(os.path.join(args.data_dir,'data.json'),)
    # validation_file = open(os.path.join(args.data_dir,'validationset.json'),)
        
    # train_file = open(os.path.join(args.data_dir,'littrainset.json'),)
    # test_file = open(os.path.join(args.data_dir,'littestset.json'),)
    # validation_file = open(os.path.join(args.data_dir,'litvalidationset.json'),)

    train_data = json.load(train_file)
    test_data = json.load(test_file)
    # validation_data = json.load(validation_file)

    traindf = pd.DataFrame(train_data) 
    traindf['mode'] = 'train'
    testdf = pd.DataFrame(test_data)
    testdf['mode'] = 'test' 
    # validdf = pd.DataFrame(validation_data) 
    # validdf['mode'] = 'eval'

    traindf['image_name'] = traindf['image_name'].apply(lambda x: os.path.join(args.data_dir, 'images', x))
    testdf['image_name'] = testdf['image_name'].apply(lambda x: os.path.join(args.data_dir, 'images', x))
    # validdf['image_name'] = validdf['image_name'].apply(lambda x: os.path.join(args.data_dir, 'images', x))

    traindf['question_type'] = traindf['question_type'].str.lower()
    testdf['question_type'] = testdf['question_type'].str.lower()
    # validdf['question_type'] = validdf['question_type'].str.lower()




    return traindf, None, testdf


#Utils
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def encode_text(caption,tokenizer, args):
    # print("cccccccccccccccccaptionnnnnsss", caption)
    if type(caption) is not  torch.NoneType :
        part1 = [0 for _ in range(5)]
        #get token ids and remove [CLS] and [SEP] token id
        part2 = tokenizer.encode(caption)[1:-1]

        tokens = [tokenizer.cls_token_id] + part1 + [tokenizer.sep_token_id] \
        + part2[:args.max_position_embeddings-8] + [tokenizer.sep_token_id]
        segment_ids = [0]*(len(part1)+2) + [1]*(len(part2[:args.max_position_embeddings-8])+1)
        input_mask = [1]*len(tokens)
        n_pad = args.max_position_embeddings - len(tokens)
        tokens.extend([0]*n_pad) ### expand inputs in a fix size 
        segment_ids.extend([0]*n_pad)
        input_mask.extend([0]*n_pad)
    else :
        print("NNNNNNNNone")
        return

    
    return tokens, segment_ids, input_mask


###  end of encode text 

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


class VQAMed(Dataset): ### inheritance does not accure????
    def __init__(self, df, tfm, args, mode = 'train', tokenizer=None):### construct object of VQA 
        self.df = df.values 
        self.tfm = tfm ### with special transformation
        self.args = args
        if args.bert_model=="distilbert-base-uncased":
            self.tokenizer=DistilBertTokenizer.from_pretrained(args.bert_model) ## get specific tokenizer 
        else:
            self.tokenizer =BertTokenizer.from_pretrained(args.bert_model)
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # print("df:::", self.df)
        # print("index:::", idx)
        # print("type of df::::", type(self.df))
        path = self.df[idx,4]
        # print("path of df::::", path)
        if self.mode == "train":
            print("We are in Train Mode\n")
            question = self.df[idx, 7]

        elif self.mode == "test":
            print("We are in Test Mode\n")
            question = self.df[idx, 8]
            # if question == "NA" or question == "NULL" or len(str(question))==0 or len(str(question))<=3:
                # print("We have exception\n")
                # question = self.df[idx, 7]


        answer = self.df[idx, 12]
        # print("answer::::", answer)
        # if self.mode == 'test':
            # tok_ques = self.tokenizer.tokenize(question)

        img = Image.open(path)


        if self.tfm:
            img = self.tfm(img)

            
        tokens, segment_ids, input_mask= encode_text(question, self.tokenizer, self.args) ## porsesh va token token mikonim 


        if self.mode != 'eval': ### trian 
            return img, torch.tensor(tokens, dtype = torch.long), \
            torch.tensor(segment_ids, dtype = torch.long)\
            ,torch.tensor(input_mask, dtype = torch.long), \
                torch.tensor(answer, dtype = torch.long)
        else: ### eval or test 
            return img, torch.tensor(tokens, dtype = torch.long), \
                torch.tensor(segment_ids, dtype = torch.long) ,\
                    torch.tensor(input_mask, dtype = torch.long), \
                        torch.tensor(answer, dtype = torch.long)  ### what is applicaton of this?

 
### the end of VQAMed class

def calculate_bleu_score(preds,targets, idx2ans):
       
    bleu_per_answer = np.asarray([sentence_bleu([str(idx2ans[target]).split()],str(idx2ans[pred]).split(),weights=[1]) for pred,target in zip(preds,targets)])

    return np.mean(bleu_per_answer)


## input ??
class Embeddings(nn.Module):
    def __init__(self, args):
        super(Embeddings, self).__init__()

        self.word_embeddings = nn.Embedding(args.vocab_size, 128, padding_idx=0)

        """num_embeddings (int) size of the dictionary of embeddings
            embedding_dim (int) the size of each embedding vector
        """

        """
          If specified, the entries at padding_idx do not contribute to the gradient; therefore, 
          the embedding vector at padding_idx is not updated during training, i.e. it remains as a 
          fixed “pad”. For a newly constructed Embedding, the embedding vector at padding_idx will 
          default to all zeros, but can be updated to another value to be used as the padding vector.
        
        """
        self.word_embeddings_2 = nn.Linear(128, args.hidden_size, bias=False)
         ### a linear layer we define
        self.position_embeddings = nn.Embedding(args.max_position_embeddings, args.hidden_size)
         ##for embede the position
        self.type_embeddings = nn.Embedding(3, args.hidden_size)
        ## we have three type 
        self.LayerNorm = nn.LayerNorm(args.hidden_size, eps=1e-12)  ###eps – a value added to the 
        ##denominator for numerical 
        ##stability. Default: 1e-5
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.len = args.max_position_embeddings

        ### define structure embedding class

    def forward(self, input_ids, segment_ids, position_ids=None):
        if position_ids is None:
            if torch.cuda.is_available():
                position_ids = torch.arange(self.len, dtype=torch.long).cuda()
            else:
                position_ids = torch.arange(self.len, dtype=torch.long)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids) ### unsqueeze means feshordan
            ##  i think expand to length of number of token or inputs
        words_embeddings = self.word_embeddings(input_ids)
        words_embeddings = self.word_embeddings_2(words_embeddings) ## a linear operation 
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.type_embeddings(segment_ids)

        ### construct final word embedding 
        embeddings = words_embeddings + position_embeddings + token_type_embeddings ### construct the embedding
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

###  end of embedding class 


class Transfer(nn.Module):
    def __init__(self,args):
        super(Transfer, self).__init__()
        self.args = args
        if args.image_embedding =="resnet":  ### it is a condition 
            self.args = args
            self.model = models.resnet152(pretrained=True)
            # for p in self.parameters():
            #     p.requires_grad=False
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
        elif args.image_embedding == "hybrid":

            
            self.model1 = models.resnet152(pretrained=True)
            self.relu = nn.ReLU()

            self.conv21 = nn.Conv2d(196, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
            self.gap21 = nn.AdaptiveAvgPool2d((1,1))

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

        
            self.model2 = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
            



            ### layer specification 


    def forward(self, img):  ## is a singleimg or a batch of images???!!!
        # print("imaaaaaaaaaaaaaaaaaaaaaaggggeeeeeee:",img.size())
        if self.args.image_embedding =="resnet":

            modules2 = list(self.model.children())[:-2] ## do ta mandeh be akhari !!
            fix2 = nn.Sequential(*modules2)  ## sequential
            # print("fix2 structure alone:",fix2[-1:])

            # print("fix2 structure:", fix2(img).size())
            v_2 = self.gap2(self.relu(self.conv2(fix2(img)))).view(-1,self.args.hidden_size)
            # print("self.conv2(fix2)",self.conv2(fix2(img)).size())
            # print("")
            # print("v2size:", v_2.size())  ### (12, 768)
            modules3 = list(self.model.children())[:-3] ### 3 ta laye be akhari 
            fix3 = nn.Sequential(*modules3)
            v_3 = self.gap3(self.relu(self.conv3(fix3(img)))).view(-1,self.args.hidden_size)
            # print("v3size:", v_3.size())  ###
            modules4 = list(self.model.children())[:-4]
            fix4 = nn.Sequential(*modules4)
            v_4 = self.gap4(self.relu(self.conv4(fix4(img)))).view(-1,self.args.hidden_size)
            modules5 = list(self.model.children())[:-5]
            fix5 = nn.Sequential(*modules5)
            v_5 = self.gap5(self.relu(self.conv5(fix5(img)))).view(-1,self.args.hidden_size)
            modules7 = list(self.model.children())[:-7]
            fix7 = nn.Sequential(*modules7)
            v_7 = self.gap7(self.relu(self.conv7(fix7(img)))).view(-1,self.args.hidden_size)
            return v_2, v_3, v_4, v_5, v_7  ## 5 feature vector extracted 

        elif self.args.image_embedding == "hybrid":

            
            modules2 = list(self.model1.children())[:-2]
            fix2 = nn.Sequential(*modules2)
            v_2 = self.gap2(self.relu(self.conv2(fix2(img)))).view(-1,self.args.hidden_size)

            modules21 = list(self.model2.children())[:]
            fix21 = nn.Sequential(*modules21)
           
            v_21 = self.gap21(self.relu(self.conv21(fix21(img).view(img.size()[0],196,10,100)))).view(img.size()[0],-1)
            # v_21 = z_gap
            v_2 = torch.add(v_2, v_21)
            
            modules3 = list(self.model1.children())[:-3] ### 3 ta laye be akhari 
            fix3 = nn.Sequential(*modules3)
            v_3 = self.gap3(self.relu(self.conv3(fix3(img)))).view(-1,self.args.hidden_size)

            modules31 = list(self.model2.children())[:-1]
            fix31 = nn.Sequential(*modules31)
           
            z_gap = self.gap21(self.relu(self.conv21(fix31(img).view(img.size()[0],196,24,32))))
            v_31 = z_gap.view(img.size()[0],-1)
            v_2 = torch.add(v_3, v_31)

            modules4 = list(self.model1.children())[:-4]
            fix4 = nn.Sequential(*modules4)
            v_4 = self.gap4(self.relu(self.conv4(fix4(img)))).view(-1,self.args.hidden_size)

            modules41 = list(self.model2.children())[:-2]
            fix41 = nn.Sequential(*modules41)
            inter_41 = self.conv21(fix41(img).view(img.size()[0],196,24,32))
            v_41 = self.gap21(self.relu(inter_41)).view(-1,self.args.hidden_size)
            v_4 = torch.add(v_4, v_41)

            modules5 = list(self.model1.children())[:-5]
            fix5 = nn.Sequential(*modules5)
            v_5 = self.gap5(self.relu(self.conv5(fix5(img)))).view(-1,self.args.hidden_size)

            modules51 = list(self.model2.children())[:-3]
            fix51 = nn.Sequential(*modules51)
            inter_51 = self.conv21(fix51(img).view(img.size()[0],196,-1,32))
            v_51 = self.gap21(self.relu(inter_51)).view(-1,self.args.hidden_size)
            v_5 = torch.add(v_5, v_51)

            modules7 = list(self.model1.children())[:-7]
            fix7 = nn.Sequential(*modules7)
            v_7 = self.gap7(self.relu(self.conv7(fix7(img)))).view(-1,self.args.hidden_size)

            modules71 = list(self.model2.children())[:-4]
            fix71 = nn.Sequential(*modules71)
            inter_71 = self.conv21(fix71(img).view(img.size()[0],196,24,32))
            v_71 = self.gap21(self.relu(inter_71)).view(-1,self.args.hidden_size)

            v_7 = torch.add(v_7, v_71)

            return v_2, v_3, v_4, v_5, v_7 


            ### ta in ghesmat kamel shavad 
## end of Transfer class



class MultiHeadedSelfAttention(nn.Module): ## we should know about thi function at first 
    def __init__(self, args):
        super(MultiHeadedSelfAttention,self).__init__()
        self.proj_q = nn.Linear(args.hidden_size, args.hidden_size)
        self.proj_k = nn.Linear(args.hidden_size, args.hidden_size)
        self.proj_v = nn.Linear(args.hidden_size, args.hidden_size)
        self.drop = nn.Dropout(args.hidden_dropout_prob)
        self.scores = None ## what does score mean hear?
        self.n_heads = args.heads


    def forward(self, x, mask):
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x) ### chon vorudi har se x ast pas be in 
        ### mana hast ke ba self atttentin kar mikonim tebgh estedlal "khodam" q va k bayad ham faza va entebaghpazir 
        ### bashand va v mitavanad motafavet bashad dar attention 

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

        ### self attention abstract 



    def split_last(self, x, shape):
        shape = list(shape)
        assert shape.count(-1) <= 1  ### this condition should be TRUE if it is False raise an error
        if -1 in shape:
            shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape)) 
            ### Return the product of array elements over a given axis
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

"""torch.nn.module 
Base class for all neural network modules.
Your models should also subclass this class.
Modules can also contain other Modules, allowing to nest them in a tree structure.
You can assign the submodules as regular attributes:

 Edpresso Team. Class attributes are variables of a class that are shared between all 
 of its instances. They differ from instance attributes in that instance attributes 
 are owned by one specific instance of the class only, and ​are not shared between instances.
"""
##training (bool) – Boolean represents whether this module is in training or evaluation mode.

class BertLayer(nn.Module):
    def __init__(self,args, share='all', norm='pre'): ### mitunim tarif konim custom shodeh dige??
        super(BertLayer, self).__init__()
        self.share = share  ## what does it mean??
        self.norm_pos = norm
        self.norm1 = nn.LayerNorm(args.hidden_size, eps=1e-12)
        """انتظار داریم که با تکرار نمونه‌گیری‌ها، متوسط مقدار برآوردگرهای حاصل، با پارامتر واقعی جامعه تقریبا برابر شود
         
         ممکن است عضوی از کلاس یا خانواده برآوردگرهای نااریب نسبت به یک برآوردگر خارج از این کلاس، دارای واریانس (Variance) بیشتری یا در حقیقت دقت کمتری باشد.
         برای مثال کلاس برآوردگرهای نااریب، برای پارامترهایشان دارای اریبی صفر هستند.
        """
        self.norm2 = nn.LayerNorm(args.hidden_size, eps=1e-12)
        ##Applies Layer Normalization over a mini-batch of inputs as described in the paper
        ##eps – a value added to the denominator for numerical stability. Default: 1e-5
        self.drop1 = nn.Dropout(args.hidden_dropout_prob)
        self.drop2 = nn.Dropout(args.hidden_dropout_prob)

        if self.share == 'ffn':
            self.attention = nn.ModuleList([MultiHeadedSelfAttention(args) for _ in range(args.n_layers)])
            self.proj = nn.ModuleList([nn.Linear(args.hidden_size, args.hidden_size) for _ in range(args.n_layers)])
            self.feedforward = PositionWiseFeedForward(args)  ## sakhtar sabeti darad dar vaghe baznevisi
            ###modular 
            
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
                h = self.proj[layer_num](attn_output) ### output kodum layeh ast?
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
        return out, attn_scores  ###  finalllllyyyy operation for atention

class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer,self).__init__()
        if args.bert_model=="distilbert-base-uncased": ### specify distile bert 
            base_model=DistilBertModel.from_pretrained(args.bert_model)
        else:
            base_model = BertModel.from_pretrained(args.bert_model) ## other model 

        # base_model = BertModel.from_pretrained('bert-base-multilingual-cased')
        # print("list::::", list(base_model.children())[0:])
        # print("list::::", list(base_model.children()))

        bert_model = nn.Sequential(*list(base_model.children())[0:]) ### Structure Of Model
        self.bert_embedding = bert_model[0]
        # self.embed = Embeddings(args)
        self.trans = Transfer(args)
        self.blocks = BertLayer(args,share='none', norm='pre')
        self.n_layers = args.n_layers

### if you whatch the code as an black        

    def forward(self, img, input_ids, token_type_ids, mask):
        v_2, v_3, v_4, v_5, v_7 = self.trans(img)
        # h = self.embed(input_ids, token_type_ids)
        h = self.bert_embedding(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=None)
        # print("hhhhhhh size", h.size())
        # print("v_2.size()",v_2.size())
        for i in range(len(h)):
            h[i][1] = v_2[i]
        # print("v_3.size()",v_3.size())
        for i in range(len(h)):
            h[i][2] = v_3[i]
        # print("v_4.size()",v_4.size())

        for i in range(len(h)):
            h[i][3] = v_4[i]
        # print("v_5.size()",v_5.size())
        for i in range(len(h)):
            h[i][4] = v_5[i]
        # print("v_7.size()",v_7.size())
        for i in range(len(h)):
            h[i][5] = v_7[i]

        hidden_states = []
        all_attn_scores = []
        for i in range(self.n_layers):
            h, attn_scores = self.blocks(h, mask, i)
            hidden_states.append(h)
            all_attn_scores.append(attn_scores)

        return torch.stack(hidden_states, 0), torch.stack(all_attn_scores, 0)

### end of Transformer 

class Model(nn.Module):
    def __init__(self,args):
        super(Model,self).__init__()
        self.args = args
        self.transformer = Transformer(args)


        """Some weights of the model checkpoint at bert-base-uncased were not used when 
        initializing BertModel: ['cls.predictions.transform.dense.weight', 
        'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.decoder.weight', 
        'cls.seq_relationship.weight', 'cls.seq_relationship.bias',
        'cls.predictions.transform.dense.bias', 'cls.predictions.bias', 
        'cls.predictions.transform.LayerNorm.weight']
        """
        """- This IS expected if you are initializing BertModel from the checkpoint of a model
        trained on another task or with another architecture (e.g. initializing a
        BertForSequenceClassification model from a BertForPreTraining model). - 
        This IS NOT expected if you are initializing BertModel from the checkpoint of a model 
        that you expect to be exactly identical (initializing a BertForSequenceClassification model
        from a BertForSequenceClassification model). """



        print("m2")
        self.fc1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.activ1 = nn.Tanh()
        print("m3")
        self.classifier = nn.Sequential(nn.Linear(args.hidden_size, args.hidden_size),
                                        nn.LayerNorm(args.hidden_size, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(args.hidden_size, args.vocab_size))
    def forward(self, img, input_ids, segment_ids, input_mask):
        h, attn_scores = self.transformer(img, input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc1(h.mean(0).mean(1)))
        logits = self.classifier(pooled_h)
        return logits, attn_scores 
    
    

### train one epoch 
def train_one_epoch(loader, model, optimizer, criterion, device, scaler, args, train_df, idx2ans):

    model.train()
    train_loss = []
    
    PREDS = []
    TARGETS = []
    # print("555555555555555")
    bar = tqdm(loader, leave = False)
    for (img, question_token,segment_ids,attention_mask,target) in bar:
        
        img, question_token,segment_ids,attention_mask,target = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
        question_token = question_token.squeeze(1)
        attention_mask = attention_mask.squeeze(1)
        loss_func = criterion
        optimizer.zero_grad()

        if args.mixed_precision:
            with torch.cuda.amp.autocast(): 
                logits, _ = model(img, question_token, segment_ids, attention_mask)
                loss = loss_func(logits, target)
        else:
            logits, _ = model(img, question_token, segment_ids, attention_mask)
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
            
        pred = logits.softmax(1).argmax(1).detach()
        PREDS.append(pred)
        if args.smoothing:
            TARGETS.append(target.argmax(1))
        else:
            TARGETS.append(target)        

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        bar.set_description('train_loss: %.5f' % (loss_np))
    
    PREDS = torch.cat(PREDS).cpu().numpy()
    TARGETS = torch.cat(TARGETS).cpu().numpy()

    total_acc = (PREDS == TARGETS).mean() * 100.
    closed_acc = (PREDS[train_df['answer_type']=='CLOSED'] == TARGETS[train_df['answer_type']=='CLOSED']).mean() * 100.
    open_acc = (PREDS[train_df['answer_type']=='OPEN'] == TARGETS[train_df['answer_type']=='OPEN']).mean() * 100.

    acc = {'total_acc': np.round(total_acc, 4), 'closed_acc': np.round(closed_acc, 4),\
    'open_acc': np.round(open_acc, 4)}


    return np.mean(train_loss), acc  ### it is important what is returning 

### validate the validation data set 
def validate(loader, model, criterion, device, scaler, args, val_df, idx2ans):
    model.eval()
    val_loss = []

    PREDS = []
    TARGETS = []
    bar = tqdm(loader, leave=False)

    with torch.no_grad():
        for (img, question_token,segment_ids,attention_mask,target) in bar:

            img, question_token,segment_ids,attention_mask,target = img.to(device),\
            question_token.to(device), segment_ids.to(device), attention_mask.to(device),\
            target.to(device)
            question_token = question_token.squeeze(1)
            attention_mask = attention_mask.squeeze(1)


            if args.mixed_precision:
                with torch.cuda.amp.autocast(): 
                    logits, _ = model(img, question_token, segment_ids, attention_mask)
                    loss = criterion(logits, target)
            else:
                logits, _ = model(img, question_token, segment_ids, attention_mask)
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

    total_acc = (PREDS == TARGETS).mean() * 100.
    closed_acc = (PREDS[val_df['answer_type']=='CLOSED'] == TARGETS[val_df['answer_type']=='CLOSED']).mean() * 100.
    open_acc = (PREDS[val_df['answer_type']=='OPEN'] == TARGETS[val_df['answer_type']=='OPEN']).mean() * 100.

    acc = {'total_acc': np.round(total_acc, 4), 'closed_acc': np.round(closed_acc, 4), 'open_acc': np.round(open_acc, 4)}

    # add bleu score code
    total_bleu = calculate_bleu_score(PREDS,TARGETS,idx2ans)
    closed_bleu = calculate_bleu_score(PREDS[val_df['answer_type']=='CLOSED'],TARGETS[val_df['answer_type']=='CLOSED'],idx2ans)
    open_bleu = calculate_bleu_score(PREDS[val_df['answer_type']=='OPEN'],TARGETS[val_df['answer_type']=='OPEN'],idx2ans)

    bleu = {'total_bleu': np.round(total_bleu, 4),  'closed_bleu': np.round(closed_bleu, 4), 'open_bleu': np.round(open_bleu, 4)}

    return val_loss, PREDS, acc, bleu
### final test for testdataset eval()  
def test(loader, model, criterion, device, scaler, args, val_df,idx2ans):

    model.eval()

    PREDS = []
    TARGETS = []

    test_loss = []

    with torch.no_grad():
        try:
            for (img,question_token,segment_ids,attention_mask,target) in tqdm(loader, leave=False):
                img, question_token, segment_ids, attention_mask, target = img.to(device), question_token.to(device), segment_ids.to(device), attention_mask.to(device), target.to(device)
                question_token = question_token.squeeze(1)
                attention_mask = attention_mask.squeeze(1)
                
                if args.mixed_precision:
                    with torch.cuda.amp.autocast(): 
                        logits, _ = model(img, question_token, segment_ids, attention_mask)
                        loss = criterion(logits, target)
                else:
                    logits, _ = model(img, question_token, segment_ids, attention_mask)
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

            total_acc = (PREDS == TARGETS).mean() * 100.
            closed_acc = (PREDS[val_df['answer_type']=='CLOSED'] == TARGETS[val_df['answer_type']=='CLOSED']).mean() * 100.
            open_acc = (PREDS[val_df['answer_type']=='OPEN'] == TARGETS[val_df['answer_type']=='OPEN']).mean() * 100.

            acc = {'total_acc': np.round(total_acc, 4), 'closed_acc': np.round(closed_acc, 4), 'open_acc': np.round(open_acc, 4)}




            return test_loss, PREDS, acc

        except:
            print("this is noiiiiiiiiiiiiiiisssyyyyyyyy\n")   
            return  "Noisy","Noisy","Noisy"
      

### final prediction 
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
                        logits, _ = model(img, question_token, segment_ids, attention_mask)
                else:
                    logits, _ = model(img, question_token, segment_ids, attention_mask)
 
                if i == 0:
                    pred = logits.detach().cpu().numpy()/len(all_models)
                else:
                    pred += logits.detach().cpu().numpy()/len(all_models)
            
            PREDS.append(pred)

    PREDS = np.concatenate(PREDS)

    return PREDS
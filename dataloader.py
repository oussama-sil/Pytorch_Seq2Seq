
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from torchtext.datasets import Multi30k,IWSLT2016,IWSLT2017
# from torchtext.data.utils import get_tokenizer

#! To diasble tensorflow logging errors
import tensorflow as tf
import logging
import os

# Set TensorFlow logging level to suppress warnings
tf.get_logger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#? To print vectore on one line on the console 
terminal_width = os.getenv('COLUMNS',200)
torch.set_printoptions(linewidth=terminal_width)

# from torchtext.legacy.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time

import os



#$ Applies : Tokenization -> 





#* Construct the training dataset 
def preprocess_dataset(dataset,nlp_in,nlp_out,path,label,lemma=False):
        """
            Construct the dataset 
            remove the punctuation ..

            Save the preprocessed dataset, three files will be created :
                -> path/in.txt : file contains the input (original sequences)
                -> path/out.txt : file contains the input (original sequences )
                -> path/vocab_in.txt : file contains the input (after lemmatization)
                -> path/vocab_out.txt : file contains the input (after lemmatization)

            Input :
                dataset : dataset to process (iterable)
                nlp_in : spacy model to process input sequences
                nlp_out : spacy model to process output sequences 
                path : path to folder where to store the files 

        """
        vocab_in = {}
        vocab_out = {}

        print(f" > Preprocessing data ({label})")
        count = 0
        with open(os.path.join(path,f"{label}_in.txt"), 'w', encoding='utf-8') as in_file, open(os.path.join(path,f"{label}_out.txt"), 'w', encoding='utf-8') as out_file :
            for in_,out_ in dataset:
                
                
                #$ Tokenizing the sequences 
                seq_in = nlp_in(in_.lower())
                seq_out = nlp_out(out_.lower())
                
                #$ Removing punctuation 
                seq_in_no_punct= [token for token in seq_in if  not token.is_punct and token.text != ' ' and   token.text != ' '  and   token.text != '	'  ]
                seq_out_no_punct= [token for token in seq_out if  not token.is_punct and token.text != ' ' and   token.text != ' ' and   token.text != '	']

                #$ Writing to the files 
                in_file.write(f"{' '.join(token.text for token in seq_in if  not token.is_punct and token.text != ' ' and   token.text != ' ' and   token.text != '	' )}\n")
                out_file.write(f"{' '.join(token.text for token in seq_out if  not token.is_punct and token.text != ' ' and   token.text != ' ' and   token.text != '	')}\n")

                #$ updating Vocab lists with the tokens 
                for token in seq_in_no_punct:
                    tmp = token.lemma_ if lemma else token.text 
                    if tmp not in vocab_in :
                        vocab_in[tmp] = 1
                    else :
                        vocab_in[tmp] += 1
                for token in seq_out_no_punct:
                    tmp = token.lemma_ if lemma else token.text
                    # print(f'{tmp} {token}')
                    if tmp not in vocab_out:
                        vocab_out[tmp] = 1
                    else:
                        vocab_out[tmp] += 1
                count += 1
                if count % 500 == 0:
                    print(f"\t->Processed sequences  : {str(count):{10}}")

        with open(os.path.join(path,f"{label}_vocab_in.txt"), 'w', encoding='utf-8') as file:            
            sorted_vocab_in = dict(sorted(vocab_in.items(), key=lambda item: item[1], reverse=True))
            for item,freq in sorted_vocab_in.items():
                file.write(f"{item} {freq}\n")

        with open(os.path.join(path,f"{label}_vocab_out.txt"), 'w', encoding='utf-8') as file:
            sorted_vocab_out = dict(sorted(vocab_out.items(), key=lambda item: item[1], reverse=True))
            for item,freq in sorted_vocab_out.items():
                file.write(f"{item} {freq}\n")

        print("End of processing")

class TranslationDataset(Dataset):
    def __init__(self,path_data_in,path_data_out,
                path_vocab_in,path_vocab_out,
                nlp_in,nlp_out,
                max_seq_length=None,vocab_size=None,vocab_min_occ=None,
                ):

        #$ Load file in and out 
        #$ construct the vocab for the dataset (add <s>, </s>)
        #$ construct one hot encoding 

        self.max_seq_length = max_seq_length
        self.nlp_in = nlp_in #spacy object for input sequence 
        self.nlp_out = nlp_out #spacy object for output sequence 


        #$ Loading the data 
        with open(path_data_in, 'r', encoding='utf-8') as file:
            self.x =  [line.strip() for line in file.readlines()]
        with open(path_data_out, 'r', encoding='utf-8') as file:
            self.y =  [line.strip() for line in file.readlines()]

        #$ Loading the vocab 
        with open(path_vocab_in, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            if vocab_size != None:
                lines = lines[:vocab_size]
            self.vocab_in = []
            for line in lines:
                # print("--->"+line+"<---")
                elem,occ = line.strip().split()
                if vocab_min_occ != None and int(occ) >= vocab_min_occ or vocab_min_occ == None :
                    self.vocab_in.append(elem)
            self.vocab_in.append('<s>')
            self.vocab_in.append('</s>')
            self.vocab_in.append('<oov>')
            self.vocab_in.insert(0,'<pad>') #$ so that no element has the index =0 (Keep zeros for padding)


        with open(path_vocab_out, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            if vocab_size != None:
                lines = lines[:vocab_size]
            self.vocab_out = []
            for line in lines:
                elem,occ = line.strip().split()
                if vocab_min_occ != None and  int(occ)  >= vocab_min_occ or vocab_min_occ == None :
                    self.vocab_out.append(elem)
            self.vocab_out.append('<s>')
            self.vocab_out.append('</s>')
            self.vocab_out.append('<oov>')
            self.vocab_out.insert(0,'<pad>') #$ add random token at the beggining so that no element has the index =0
        
        # size of the data 
        self.m = len(self.x)

    def __getitem__(self, index):
        """
            Load one item, if large dataset use loading with openning file..
            Each element : sequence ->Lower(done in preprocessing) -> add <s> and </s> -> Tokenize(Transform)
            Returns :
                Sequence_in (str) :
                Sequence_ou (str) : 
                Sequence_index_in (tensor [seq_length]): list of indices for in seq
                Sequence_index_out (tensor [seq_length]): list of indices for out seq
                Sequence_token_in (tensor [seq_length]): list of tokens for in seq
                Sequence_token_out (tensor [seq_length]): list of tokens for out seq
        """

        #! EROOR : len() counts individual caracters and not words
        #! [:masx..] keeps only the first n caracters and not token 
        # if self.max_seq_length != None and len(self.x[index])>self.max_seq_length:
        #     self.x[index] = self.x[index][:self.max_seq_length]
        # if self.max_seq_length != None and len(self.y[index])>self.max_seq_length:
        #     self.y[index] = self.y[index][:self.max_seq_length]
        x = '<s> '+self.x[index]+' </s>' 
        y = '<s> '+self.y[index]+' </s>'
        x_token,x_ind = self.transform(self.x[index],self.vocab_in,self.nlp_in,reverse=True)
        y_token,y_ind = self.transform(self.y[index],self.vocab_out,self.nlp_out) #reverse to follow the originale paper on sea2sea model

        return x,y,torch.tensor(x_ind),torch.tensor(y_ind),x_token,y_token
    
    def __len__(self):
        return self.m

    def get_data(self):
        return self.x, self.y

    def get_vocab(self):
        """
            Return the vocabs for input and output sequences 
        """
        return self.vocab_in,self.vocab_out

    def transform(self,x,vocab,nlp,lemma=False,reverse=False):
        """
            Transform one sequence for the model 
            Input : x (<s> word1 word2 word3 </s>),vocab,nlp
            Output : tensor([seq_length]) containing list of index of words
        """
        seq = nlp(x)
        # print([token.text for token in seq])
        #$ Tokenize  and Lemmetize  and remove words out of vocab 
        if not reverse:
            if lemma:
                token_lists = ['<s>'] + [token.lemma_ if (token.lemma_ in vocab) else '<oov>'   for token in seq ] + ['</s>']
            else : 
                token_lists = ['<s>'] + [token.text if (token.text in vocab) else '<oov>' for token in seq ] + ['</s>']
        else :
            if lemma:
                token_lists = ['<s>'] + [token.lemma_ if (token.lemma_ in vocab) else '<oov>'   for token in seq ][::-1] + ['</s>']
            else : 
                token_lists = ['<s>'] + [token.text if (token.text in vocab) else '<oov>' for token in seq ][::-1]  + ['</s>']
        
        #$ Return list of indexes of 
        indices_list = [vocab.index(token) for token in token_lists]

        return token_lists,indices_list

    def transform_input_sequence(self,seq,lemma=False):
        """
            #* transform from a raw input sequence to an input of the model 
            
        """    

        # Tokenizing 
        seq_in = self.nlp_in(seq.lower())
        
        #$ Removing punctuation 
        seq_in_no_punct= [token for token in seq_in if  not token.is_punct and token.text != ' ' and   token.text != ' '  and   token.text != '	'  ]

        if lemma:
            token_lists = ['<s>'] + [token.lemma_ if (token.lemma_ in self.vocab_in) else '<oov>'   for token in seq_in_no_punct ][::-1] + ['</s>']
        else : 
            token_lists = ['<s>'] + [token.text if (token.text in self.vocab_in) else '<oov>' for token in seq_in_no_punct ][::-1]  + ['</s>']

        #
        indices_list = [[self.vocab_in.index(token) for token in token_lists]]
        x = torch.tensor(indices_list)
        lengths_x = [x.size()[1]]
        
        
        return x,lengths_x,token_lists



#* Function for the dataloader
def collate_fn(batch):
    """
        This function is useful for handeling sequences of varying length
        => Returns a batch of padded sequences 

        Input : 
            batch : from the dataloader
        returns :
            x_ind_pad,y_ind_pad : input/output padded index sequences in the batch [m,max_length_sequence]
            lengths_x,lengths_y = arrays of lengths of input/output sequences in the batch [m]
            x,y = Input/output sequences 
            x_token,y_token =  Input/output sequences tokenized  
    """

    #$ Batch of data from the dataloader 
    x,y,x_ind,y_ind,x_token,y_token = zip(*batch)

    #? For next layer input 
    #$ Lengths of sequences for packing 
    lengths_x = [x_.size()[0] for x_ in x_ind]
    lengths_y = [y_.size()[0] for y_ in y_ind]
    
    #$ Padding sequences 
    x_ind_pad = pad_sequence(x_ind, batch_first=True)
    y_ind_pad = pad_sequence(y_ind, batch_first=True)
    
    return x_ind_pad,y_ind_pad,lengths_x,lengths_y,x,y,x_token,y_token


def construct_data(dataset_folder="./data",batch_size=4,data_type='train',spacy_in_model ="de_core_news_sm", spacy_out_model='en_core_web_sm'):

    """
        Function that returns the train, validation, and test datasets (dataset + loaders )
    """

    #? Load the full dataset to get the vocab and the list of countries 

    #? Spacy objects 
    spacy_in = spacy.load('de_core_news_sm')
    spacy_out = spacy.load('en_core_web_sm')
    #? Loading dataset 
    dataset = TranslationDataset(
        os.path.join(dataset_folder,f"{data_type}_in.txt"),
        os.path.join(dataset_folder,f"{data_type}_out.txt"),
        os.path.join(dataset_folder,f"{data_type}_vocab_in.txt"),
        os.path.join(dataset_folder,f"{data_type}_vocab_out.txt"),
        spacy_in,spacy_out
    )
    vocab_in,vocab_out = dataset.get_vocab()
    
    #? DataLoaders 
    # next
    data_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)

    return dataset,data_loader,vocab_in,vocab_out,spacy_in,spacy_out




#! TODO : change the ordering of elements in the vocab => Insert <s> </s> at the beginning of the sequence 
#! 

if __name__=="__main__":
    print("=======> Data.py  <======")
    

    #? Constructing the training, validation and test set 
    # spacy_de = spacy.load('de_core_news_sm')
    # spacy_en = spacy.load('en_core_web_sm')

    # # Translation : Gr k-> En
    # train_data, valid_data, test_data = Multi30k(root=".data",language_pair =('de', 'en'))

    # preprocess_dataset(train_data,spacy_de,spacy_en,'.\data','train')
    # preprocess_dataset(valid_data,spacy_de,spacy_en,'.\data','valid')
    # preprocess_dataset(test_data,spacy_de,spacy_en,'.\data','test')



    #? Translation : Gr -> En

    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')


    # train_data, valid_data, test_data = Multi30k(root=".data",language_pair =('de', 'en'))

    # preprocess_dataset(train_data,spacy_de,spacy_en,'.\data','train')
    # preprocess_dataset(valid_data,spacy_de,spacy_en,'.\data','valid')
    # preprocess_dataset(test_data,spacy_de,spacy_en,'.\data','test')


    #? Loading dataset 
    train_dataset = TranslationDataset(
        './data/train_in.txt',
        './data/train_out.txt',
        './data/train_vocab_in.txt',
        './data/train_vocab_out.txt',
        spacy_de,spacy_en,max_seq_length=10
    )

    x,y,x_ind,y_ind,x_token,y_token = next(iter(train_dataset))
    # print(x)
    # print(x_ind)
    # print(x_token)

    # print(y)
    # print(y_ind)
    # print(y_token)

    #? Train dataloader 
    train_loader = DataLoader(dataset=train_dataset,batch_size=3,shuffle=True,collate_fn=collate_fn)
    # x,y,x_ind,y_ind,x_token,y_token = next(iter(train_loader))
    x_ind_pad,y_ind_pad,lengths_x,lengths_y,x,y,x_token,y_token = next(iter(train_loader))
    # print(x_ind_pad)
    # print(lengths_x)
    # print(x)
    # print(x_token)


    #? Example with embedding and padding 
    # padding_idx = 0  # Set your padding index => Indesx for the padding entries 
    # embedding_layer = nn.Embedding(len(train_dataset.get_vocab()[0]), 2, padding_idx=padding_idx)
    # embd = embedding_layer(x_ind_pad)
    # print(embd[0])
    # print(embd[1])
    # print(embd[2])
    # packed_sequence = pack_padded_sequence(embd, lengths_x, batch_first=True, enforce_sorted=False)
    # print(lengths_x)
    # # print(packed_sequence.batch_sizes)
    # print(packed_sequence.data.size())
    # print(packed_sequence.data)



    #* Load data 
    while True:
        user_input = input("Enter a string: ")
        print(train_dataset.transform_input_sequence(user_input))




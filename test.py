import torch
import math 
import time 
import torch.nn as nn


import os 
import time
import sys 
import math


from utils import ConsoleColors
from dataloader import TranslationDataset,collate_fn

import gc # for garbage collector 

import spacy 
from torch.utils.data import Dataset,DataLoader
from model import Seq2Seq
from eval import construct_bleu_score
from utils import step_fct,garbage_collect


def eval(model,data_loader,device,loss_fct,eval_fct):
    """
        Evaluating the model , supposes that ther is only one batch in the validation data 

    """
    print(' Evaluating the model :')
    # Recording the loss and evaluation on the validation data at the end of each epoch 
    model.eval()
    # Computing the loss and evaluating the model 
    with torch.no_grad():
                # Evaluate on one batch of test data
        for  i, (x_ind_pad,y_ind_pad,lengths_x,lengths_y,x,y,x_token,y_token) in enumerate(data_loader):
            start_time = time.time()
                    # Pushing the data to the device
            x_ind_pad,y_ind_pad,lengths_x,lengths_y = x_ind_pad.to(device),y_ind_pad.to(device),lengths_x,lengths_y

                    # Forward through the model
            y_hat,y_pred  =  model(x_ind_pad,lengths_x,y_ind_pad,lengths_y)

                    # Compute the loss
                    # loss = loss_fct(y_hat,y)
                    # y_hat = y_hat[1:].view(-1,y_hat.shape[-1]) # to remove the first token and convert to size [m*(max_seq_length_y-1) ,embedding_out_size]
                    # y_ind_pad = y_ind_pad[1:].view(-1) 
        
            loss = loss_fct(y_hat[1:].view(-1,y_hat.shape[-1]),y_ind_pad[1:].view(-1) ) 

            # Evaluate the model
            eval_ = eval_fct(y_pred,y_ind_pad)

            end_time = time.time()
            eval_time = end_time - start_time # time of a single epoch

            print(f" -> End of epoch,  Loss = {loss.item():.6f} PPL={math.exp(loss.item()):.6f} Eval ={eval_:.6f} , eval time = {eval_time:.2f} seconds ")


def inference(model,seq_in,dataset,vocab_in,vocab_out):
    """
		Input : 
			-> model : the model , output is of shape 
            -> seq_in : input sequence 
    """
    tokens_tensor,tokens = dataset.transform_input_sequence(seq_in)
    


	

	


if __name__ == "__main__":
    print("=======> Train.py <======")

	#* Spacy models to manipulate the text 
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')


    #* Dataset loading 
    train_dataset = TranslationDataset(
            './data/train_in.txt',
            './data/train_out.txt',
            './data/train_vocab_in.txt',
            './data/train_vocab_out.txt',
            spacy_de,spacy_en,max_seq_length=10
        )
    valid_dataset = TranslationDataset(
            './data/valid_in.txt',
            './data/valid_out.txt',
            './data/train_vocab_in.txt',
            './data/train_vocab_out.txt',
            spacy_de,spacy_en,max_seq_length=10
        )


    #* Input and output vocabulary  
    vocab_in,vocab_out = train_dataset.get_vocab()
    
    #* Dataloader   
    train_loader = DataLoader(dataset=train_dataset,batch_size=128,shuffle=True,collate_fn=collate_fn)
    valid_loader = DataLoader(dataset=valid_dataset,batch_size=256,shuffle=False,collate_fn=collate_fn)


	#? Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    #? Loading the model from the checkpoint 
    path = 'checkpoints/checkpoint_4_final.pth'
    loaded_checkpoint = torch.load(path)
    epoch = loaded_checkpoint["epoch"]
    seq2seq = Seq2Seq(len(vocab_in),len(vocab_out),256,256,512,4,0.5,device).to(device)


    #? Evaluating the model  
    criterion = nn.CrossEntropyLoss(ignore_index = 0)
    eval_bleu_score = construct_bleu_score(vocab_out,0,9777,9778)

    # eval(seq2seq,valid_loader,device,criterion,eval_bleu_score)
    seq2seq.load_state_dict(loaded_checkpoint["model_state"])
    seq2seq.eval()
    # eval(seq2seq,valid_loader,device,criterion,eval_bleu_score)
    
	#? Making inference on one example ccc
    while True:
          user_input = input("Enter a string: ")
          x,lengths_x,token_lists = train_dataset.transform_input_sequence(user_input)
        #   print(x)
        #   print(lengths_x)
        #   print(token_lists)
          y_pred = seq2seq.predict(x.to(device),lengths_x,vocab_out.index('<s>'), vocab_out.index('</s>'))
          print(y_pred)
          trad = [vocab_out[indx] for indx in y_pred]
          print(trad)
          
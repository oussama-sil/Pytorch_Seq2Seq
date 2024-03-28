import torch 
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchinfo import summary
from utils import ConsoleColors
import random 

from dataloader import TranslationDataset,collate_fn
import spacy 
from torch.utils.data import Dataset,DataLoader

import tensorflow as tf

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        vocab_in_size,
        vocab_out_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        device,
        max_len=200, # Max size of sequence 
    ):


        self.embedding_size = embedding_size
        self.vocab_in_size = vocab_in_size
        self.vocab_out_size = vocab_out_size
        self.src_pad_idx = src_pad_idx
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layer = num_decoder_layers
        self.forward_expansion =forward_expansion
        self.dropout = dropout

        super(Seq2SeqTransformer, self).__init__()
        self.word_embedding_in = nn.Embedding(vocab_in_size, embedding_size)
        self.position_embedding_in = nn.Embedding(max_len, embedding_size)
        self.word_embedding_out = nn.Embedding(vocab_out_size, embedding_size)
        self.position_embedding_out = nn.Embedding(max_len, embedding_size)

        self.transformer = nn.Transformer(embedding_size,num_heads,num_encoder_layers,num_decoder_layers,forward_expansion,dropout,
                                          batch_first=True)

        # To map the output to [m,trg_length,vocab_out_size]
        self.fc_out = nn.Linear(embedding_size, vocab_out_size)
        self.dropout = nn.Dropout(dropout)
        

        self.device = device 

    def get_model_name(self):
        return "Seq2Seq_Transformer"

    def get_model_characteristic(self):
        return {
            'embedding_size' : self.embedding_size,
            'vocab_in_size': self.vocab_in_size,
            'vocab_out_size': self.vocab_out_size,
            'src_pad_idx' :  self.src_pad_idx ,
            'num_heads' :    self.num_heads ,
            'num_encoder_layers' :        self.num_encoder_layers ,
            'num_decoder_layers' :       self.num_decoder_layers ,
            'forward_expansion' :        self.forward_expansion ,
            'dropout' :        self.dropout,
            'device' :self.device 
        }

    def make_src_mask(self, src):
        """
            Construct the mask for the input sequence 
        """
        src_mask = src== self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)
    
    def forward(self,x,lengths_x,y,lengths_y):
        """
            Input : 
                -> x  : input sequences of size [m,max_seq_length_x]  (indexes of tokens)
                -> y  : output sequence of size [m,max_seq_length_y] (indexes of tokens)
            Output : 
                -> y_hat : predicted sequences (probabilities of each token at each position) of size [m,max_seq_length_y,embedding_out_size] 
                -> y_pred : predicted tokens of size [m,max_seq_length_y]
        """

        m,max_seq_length_x = x.shape
        m,max_seq_length_y = y.shape
        
        # print(f'x.shape {x.shape}' )
        # print(f'y.shape {y.shape}' )
                

        # Positions of the input  [m,max_seq_length_x]
        in_positions = (
                    torch.arange(0, max_seq_length_x)
                    .expand(m,max_seq_length_x)
                    .to(self.device)
                )
        
        # Positions of the output [m,max_seq_length_y]
        out_positions = (
                    torch.arange(0, max_seq_length_y)
                    .expand(m,max_seq_length_y)
                    .to(self.device)
                )

        # Embedding of input sequence [m,max_seq_length_x,embedding_size]
        embed_in = self.dropout(
            (self.word_embedding_in(x) + self.position_embedding_in(in_positions))
        )
        # print(f'embed_in.shape {embed_in.shape}' )
        
        # Embedding of input sequence [m,max_seq_length_y,embedding_size]
        embed_out = self.dropout(
            (self.word_embedding_out(y) + self.position_embedding_out(out_positions))
        )
        # print(f'embed_out.shape {embed_out.shape}' )
        
        src_padding_mask = self.make_src_mask(x) # Mask of size [m,max_seq_length_x]

        trg_mask = self.transformer.generate_square_subsequent_mask(max_seq_length_y).to(
            self.device
        ) # Mask of size [m,max_seq_length_x]
        # print(src_padding_mask)
        # print(trg_mask)

        # print(trg_mask.size())
        
        y_hat = self.transformer(
            embed_in,
            embed_out,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        ) # Output of size [m,max_seq_length_y,embedding_size]
        


        y_hat = self.fc_out(y_hat) # Output of size [m,max_seq_length_y,vocab_out_size]
        y_pred = y_hat.argmax(2)  # Output of size [m,max_seq_length_y]
        

# src_mask.to(torch.float32)


        # print(f'y_hat.size {y_hat.size()}' )
        # print(f'y_pred.size {y_pred.size()}' )
        
        
        return y_hat,y_pred
    


#? Function to initialize the parameters of the model 
    
def params_initializer(alpha):
    def init_weights(m):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -alpha, alpha)
    return init_weights

if __name__=="__main__":
    print("=======> Model.py <======")

    #? Importing the data 
    
    # Spacy models to manipulate the text 
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

    # Loading the data in a custom dataset 
    train_dataset = TranslationDataset(
            './data/train_in.txt',
            './data/train_out.txt',
            './data/train_vocab_in.txt',
            './data/train_vocab_out.txt',
            spacy_de,spacy_en,max_seq_length=10
        )

    # Vocabs 
    vocab_in,vocab_out = train_dataset.get_vocab()

    # Dataloader 
    train_loader = DataLoader(dataset=train_dataset,batch_size=2,shuffle=True,collate_fn=collate_fn)


    # Examples from the dataset 
    train_dataloader_iter = iter(train_loader)
    x_ind_pad,y_ind_pad,lengths_x,lengths_y,x,y,x_token,y_token = next(train_dataloader_iter)


    # Example with the encoder 
    # print('\n\nEncoder')
    # encoder = Encoder(len(vocab_in),256,512,4,0.3)

    # input("Press Enter to continue...")

    # h,c = encoder(x_ind_pad,lengths_x)
    # summary(encoder,input_data =x_ind_pad,device="cpu")


    # Example with the decoder 
    # print('\n\nDecoder')
    # x = torch.randint(0, 90, (32,), dtype=torch.int)
    # decoder = Decoder(len(vocab_out),256,512,4,0.3)
    # x_,h_,c_ = decoder(x,h,c)
    # summary(decoder,input_data =[x,h,c],device="cpu")

    # input("Press Enter to continue...")

    # Example with the full model 
    print('\n\nSeq2Seq')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq2seq = Seq2SeqTransformer(512,len(vocab_in),len(vocab_out),0,8,3,3,4,0.1,device,200).to(device)

    # Initializing the model 
    seq2seq.apply(params_initializer(0.08))

    print(x_ind_pad)
    y_hat,y_pred =  seq2seq(x_ind_pad.to(device),lengths_x,y_ind_pad[:,:-1].contiguous().to(device),lengths_y)
    # summary(seq2seq,input_data =[x_ind_pad.to(device),lengths_x,y_ind_pad[:,:-1].contiguous().to(device),lengths_y],device="cuda")

    # print(seq2seq)

    # print(y_hat.size())
    # print(y_pred.size())


    #? Model size 
    param_size = 0
    nb_params = 0
    for param in seq2seq.parameters():
        nb_params += param.nelement()
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in seq2seq.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Seq2Seq nb params {} model size: {:.3f}MB'.format(nb_params,size_all_mb))
    # input("Press Enter to continue...")
 
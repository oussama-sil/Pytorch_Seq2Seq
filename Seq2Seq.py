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

#? The Encoder
class Encoder(nn.Module):
    def __init__(self,vocab_size, embedding_size,hidden_size,num_layers,dropout) :
        """
            The encoder part of the Seq2seq model
                - vocab_size : number of words in the input vocabulary  (from source sequence)
                - embedding_size : size of embedding 
                - hidden_size : size of hidden state 
                - num_layers : number of layers 
                - dropout : dropout rate 
        """
        super().__init__()

        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers #
        self.hidden_size = hidden_size
        self.dropout = dropout
        #* Embedding layer 
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_size,0)

        #* RNN network 
        self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, dropout = self.dropout,batch_first =True)

        #* Dropout 
        self.dropout = nn.Dropout(dropout)
        

    def forward(self,x,lengths=None):
        """
            Input : x of size [m,max_seq_length] #Padded of indexes 
            Output : 
                -> context_vector (last hidden state) of size  [nb_layers,m,size_hidden_state]
                -> cell state  of size  [nb_layers,m,size_hidden_state]
        """
        # print("--")
        # print(x.size()) # Input size of [m,max_seq_length] 
        # print(lengths) # Lengths of the sequences of size [m]
        # print("--")

        if lengths == None:
            print(ConsoleColors.WARNING + " Error : lengths for input sequences not provided , assuming all sequences of the same length" + ConsoleColors.ENDC)
            lengths = [x.size()[1]]*x.size()[0] 

        #* Embedding of the input 
        #* Input : x of size [m,max_seq_length]
        #* Output : embedding of size [m,max_seq_length,embedding_size]
        embd = self.dropout(self.embedding(x))
        
        # print(embd.size())
        # print(x[0])
        # print(embd[0])

        
        #* packing the sequences for RNN computation 
        packed_sequence = pack_padded_sequence(embd, lengths, batch_first=True, enforce_sorted=False)
        
        
        #* Forward through the LSTM network 
        # * return  :
        # *     output : contains the output for each timestep in last layer of size [m,max_seq_length,size_hidden_state]
        # *     h: final hidden state for each elements in each layers, [nb_layers,m,size_hidden_state]
        # *     c : contains the final cell state for each elements in each layers, [nb_layers,m,size_hidden_state]
        packed_output,(h, c) = self.rnn(packed_sequence) # the output is packed

        # print(packed_output.data.size())
        # print(h.size())
        # print(c.size())
        
        
        #* retrieving the context vectors from the output 
            
        #* Unpacking the output ==> out of size [m,max_seq_length,size_hidden_state]
        output, _ = pad_packed_sequence(packed_output, batch_first=True) #! Output size of [batch_size, seq_length, hidden_state length]

        
        return h, c 


#? The Decoder
class Decoder(nn.Module):
    def __init__(self,vocab_size, embedding_size,hidden_size,num_layers,dropout,apply_softmax=False) :
        """
            The decoder part of the Seq2seq model , performs one single step of decoding 
                - vocab_size : number of words in the input/output vocabulary (from target sequence)
                - embedding_size : size of embedding 
                - hidden_size : size of hidden state 
                - num_layers : number of layers 
                - dropout : dropout rate 
        """
        super().__init__()

        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers #
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.apply_softmax = apply_softmax
        #* Embedding layer for the input from the target sequence 
        self.embedding = nn.Embedding(self.vocab_size,self.embedding_size,0)

        #* RNN network 
        self.rnn = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, dropout = self.dropout,batch_first =True)

        #* Fully connected layers for the decoder to apply the softmax on the output 
        self.fc = nn.Linear(self.hidden_size,self.vocab_size)
        
        #* Dropout 
        self.dropout = nn.Dropout(dropout)
        

    def forward(self,x,hidden, cell):
        """
            Input : 
                -> x  : input token of size [m] #indexes of generated tokens in the previous step 
                -> hidden : hidden states from the previous step of size [nb_layers,m,size_hidden_state]
                -> cell : cell states from the previous step of size [nb_layers,m,size_hidden_state]
            Output : 
                -> predictions of the size [m,max_seq_length, embeding_size]
                -> context_vector (last hidden state) of size  [nb_layers,m,size_hidden_state]
                -> cell state  of size  [nb_layers,m,size_hidden_state]
        """
        # print("--")
        # print(x.size()) # Input size of [m] 
        # print(hidden.size()) # Input size of [nb_layers,m,size_hidden_state]
        # print(cell.size()) # Input size of [nb_layers,m,size_hidden_state]
        
        # print("--")


        #* Embedding of the input 
        #* Input : x of size [m]
        #* Output : embedding of size [m,embedding_size]
        embd = self.dropout(self.embedding(x))
        
        # print(embd.size())
        # print(x)
        # print(embd[0])

        #* Adding a dimension to match the nn.LSTM input dimension ([m,seq_lengths,embedding_size])
        #* output of size [m,1,embedding_size]
        embd = embd.unsqueeze(1)
        
        #* Forward through the LSTM network 
        # * return  :
        # *     output : contains the output for each timestep in last layer of size [m,max_seq_length,size_hidden_state]
        # *     h: final hidden state for each elements in each layers, [nb_layers,m,size_hidden_state]
        # *     c : contains the cell state for each elements in each layers, [nb_layers,m,size_hidden_state]
        output,(h, c) = self.rnn(embd,(hidden, cell)) # the output is packed

        # print(output.size())
        # print(h.size())
        # print(c.size())
        

        #* Forward through the fully connected layers to predict  the next tokens 
        predictions = self.fc(output.squeeze(1))
        if self.apply_softmax :
            predictions = F.softmax(predictions,dim=1)
        
        return predictions,h, c


#? Sea2Seq Model 
    
class Seq2Seq(nn.Module):
    def __init__(self,vocab_in_size,vocab_out_size, embedding_in_size,embedding_out_size,hidden_size,num_layers,dropout,
                device,teacher_forcing_ratio = 0.5,decoding_schema='greedy') :
        """
            The Seq2seq model 
                - vocab_in_size : number of words in the input vocabulary (from target sequence)
                - vocab_out_size : number of words in the output vocabulary (from target sequence)
                - embedding_in_size : size of embedding for input sequence 
                - embedding_out_size : size of embedding for output sequence 
                - hidden_size : size of hidden state 
                - num_layers : number of layers 
                - dropout : dropout rate 
                - device : device where the model whill be trained 
                - teacher_forcing_ratio
                - decoding_schema : decoding schema , only greedy implemented 
        """
        super().__init__()

        self.embedding_in_size = embedding_in_size
        self.vocab_in_size = vocab_in_size
        self.embedding_out_size = embedding_out_size
        self.vocab_out_size = vocab_out_size
        self.num_layers = num_layers # Same for the encoder and decoder 
        self.hidden_size = hidden_size # Same for the encoder and decoder 
        self.dropout = dropout
        self.teacher_forcing_ratio = teacher_forcing_ratio 
        self.decoding_schema  = decoding_schema
        assert decoding_schema in ['greedy'], ConsoleColors.WARNING + f"Error: {decoding_schema} is not a valid decoding schema." + ConsoleColors.ENDC
        
        self.device = device 
        
        #* Encoder of the model 
        self.encoder = Encoder(vocab_in_size, embedding_in_size,hidden_size,num_layers,dropout)
        #* Decoder of the model 
        self.decoder = Decoder(vocab_out_size, embedding_out_size,hidden_size,num_layers,dropout)


    def get_encoder_decoder(self):
        return self.encoder,self.decoder 
    
    def get_model_name(self):
            return "Seq2Seq"
    
    def get_model_characteristic(self):
        return {
            'embedding_in_size' : self.embedding_in_size,
            'vocab_in_size': self.vocab_in_size,
            'embedding_out_size': self.embedding_out_size,
            'vocab_out_size' :  self.vocab_out_size ,
            'num_layers' :    self.num_layers ,
            'hidden_size' :        self.hidden_size ,
            'dropout' :       self.dropout ,
            'teacher_forcing_ratio' :        self.teacher_forcing_ratio ,
            'decoding_schema' :        self.decoding_schema,
            'device' :self.device 
        }

    def forward(self,x,lengths_x,y,lengths_y):
        """
            Input : 
                -> x  : input sequences of size [m,max_seq_length_x]  (indexes of tokens)
                -> y  : output sequence of size [m,max_seq_length_y] (indexes of tokens)
            Output : 
                -> y_hat : predicted sequences (probabilities of each token at each position) of size [m,max_seq_length_y,embedding_out_size] 
                -> y_pred : predicted tokens of size [m,max_seq_length_y]
        """
        # print("--")
        # print(x.size()) # Input size of [m,max_seq_length_x]
        # print(y.size()) #/ Input size of [m,max_seq_length_y] 
        # print("--")

        #* Forward through the encoder 
        hidden,cell = self.encoder(x,lengths_x)
        
        #* Initializing the output tensor 
        m = y.shape[0]
        max_seq_length_y = y.shape[1]
        y_hat = torch.zeros(m,max_seq_length_y,self.vocab_out_size).to(self.device)
        y_pred = torch.zeros(m,max_seq_length_y,dtype=torch.int).to(self.device)
        
        #! The first element of y_hat are zeros 

        # print(y_hat.size()) # output size of [m,max_seq_length_y,embedding_out_size] 

        #* Variable to store the next input to the decoder 
        input = y[:,0] # create reference not a copy 

        #* Case of greedy decoding 
        if self.decoding_schema =='greedy':

            #* Loop max_seq_length_y times through the decoder 
            for i in range(1,max_seq_length_y):
                #* Forward through the decoder 
                pred,hidden,cell= self.decoder(input,hidden,cell)

                #* Storing the prediction at position i 
                y_hat[:,i,:] = pred

                #* The tokens with highest probabilities 
                top1 = pred.argmax(1)

                y_pred[:,i] = top1

                #* Use teacher forcing ? 
                teacher_forcing = random.random() < self.teacher_forcing_ratio
                # print("one step "+str(i)+" "+str(teacher_forcing))

                #* Next input to the model 
                input = y[:,i] if teacher_forcing else top1 
        

        
        return y_hat,y_pred
    
    def predict(self,x,lengths_x,sos_token,eos_token):
        """
        
        """
        #* Forward through the encoder 
        hidden,cell = self.encoder(x,lengths_x)
        
        #* Initializing the output tensor 
        y_pred = [sos_token]
        

        # print(y_hat.size()) # output size of [m,max_seq_length_y,embedding_out_size] 

        #* Variable to store the next input to the decoder 
        input = torch.tensor([sos_token]).to(self.device) # create reference not a copy 
        #* Case of greedy decoding 
        if self.decoding_schema =='greedy':

            while True:
            #* Loop max_seq_length_y times through the decoder 
            # for i in range(1,max_seq_length_y):
                #* Forward through the decoder 
                pred,hidden,cell= self.decoder(input,hidden,cell)

                #* The tokens with highest probabilities 
                top1 = pred.argmax(1)

                y_pred.append(top1[0].item())

                if top1[0]==eos_token:
                    break

                #* Next input to the model 
                input = top1 
        
        return y_pred

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
    train_loader = DataLoader(dataset=train_dataset,batch_size=1,shuffle=True,collate_fn=collate_fn)


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
    device = torch.device('cuda')
    seq2seq = Seq2Seq(len(vocab_in),len(vocab_out),256,256,512,4,0.2,device).to(device)

    # Initializing the model 
    seq2seq.apply(params_initializer(0.08))


    y_hat,y_pred =  seq2seq(x_ind_pad.to(device),lengths_x,y_ind_pad.to(device),lengths_y)
    # summary(seq2seq,input_data =[x_ind_pad.to(device),lengths_x,y_ind_pad.to(device),lengths_y],device="cuda")

    print(seq2seq)

    print(y_hat.size())
    print(y_pred.size())


    #? Model size 
    # param_size = 0
    # for param in seq2seq.parameters():
    #     param_size += param.nelement() * param.element_size()
    # buffer_size = 0
    # for buffer in seq2seq.buffers():
    #     buffer_size += buffer.nelement() * buffer.element_size()

    # size_all_mb = (param_size + buffer_size) / 1024**2
    # print('Seq2Seq model size: {:.3f}MB'.format(size_all_mb))
    # input("Press Enter to continue...")
 
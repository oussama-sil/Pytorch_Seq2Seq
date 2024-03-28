import torch
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
from Seq2Seq import Seq2Seq, params_initializer
from Seq2seq_attention import Seq2SeqAttention
from Seq2Seq_transformer import Seq2SeqTransformer

from eval import construct_bleu_score
from utils import step_fct,garbage_collect

from torchinfo import summary

# Tensorboard 
from torch.utils.tensorboard import SummaryWriter 

# torch.set_num_threads(1)


#! Delete tensorflow warnings :
import tensorflow as tf

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:22"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



def train(model,data,loss_fct,optimizer,nb_epochs,device,tracking_params,eval_fct,checkpoint_options,tensorboard_options,exec_options,clip = 1,transformer=False):
    """
        Train the model for nb_epochs in device
        Input 
            model 
            data (tuple) : dataloader for train,val 
            loss_funct 
            optimizer 
            nb_epochs
            device 
            tracking_params : dict that contains the parameters for tracking the loss and display 
                -> track_loss_train : Bool # track the loss or not on the test data
                -> track_loss_train_steps : if track_loss_train, save the loss each .. steps (batch)
                -> validation : Bool # validation data
                -> debug : print to the console the evolution of the training
                -> debug_loss : print the evolution of the loss in each epoch 
                -> average : record the average loss or the last loss
            eval_fct : a function to evaluate the model (acc, recall...), takes as inpu (y_hat,y) and returns an evaluation of the model
            checkpoint_options: dict that contains the parameters for checkpoint management 
                -> checkpoint : Bool # if True save a check point each 
                -> checkpoint_folder_path : Path to save the checkpoint
                -> checkpoint_epoch : # checkpoint each checkpoint_epoch epochs
            tensorboard_options : dict that contains the parameters for tensorboard 
                -> tensorboard : Bool # if True use tensorboard to print the evolution of the loss and evaluation 
                -> writer : str # path to the log directory
                -> label : label to add in the tensorboard 
            exec_options : dict that contains the function to execute at the end of each epoch, each steps
                -> epoch_fct : function to execute at the end of each epoch 
                -> step_fct : function to execute at the end of each n steps 
                -> nb_steps : ech nb_steps execute step_fct 
        Output 
            loss_train : Evolution of the loss of the train
            loss_val : Evolution of loss on validaton data 
            eval_loss : Evolution of eval metric on validatio data 
    """


    train_loader,val_loader = data
    
    
    nb_batch = len(train_loader) # NB batchs

    # For loss on train data each step
    loss_train = []
    tmp_loss_train = 0 # Accumulating the loss for the display 
    tmp_loss_train_count = 0 # Number of steps accummulated 
    tmp_elem_count = 0 # Number of elements accumulated 
    
    # To accumulate the loss during the epoch => to print the average loss during the epoch and the perplexity 
    epoch_loss = 0
    epoch_elem_count = 0 


    # For loss on validation data
    loss_val = []
    eval_loss = []

    

    # For tensorboard 
    writer = tensorboard_options["writer"] or  None   
    if tensorboard_options["tensorboard"] : 
        pass


        
    print("\n Training the model")
    print(f"\t > Device : {device}")
    print(f"\t > NB_epochs : {nb_epochs}")
    print(f"\t > Batch size : {train_loader.batch_size}")
    print(f"\t > NB Batches : {nb_batch}")
    print(f"""\t > Debug : {'On' if tracking_params["debug"] else 'Off' }""")

    print()


    # Training loop 
    for epoch in range(nb_epochs):
        

        model.train()
        print(f"Epoch :  [{str(epoch+1):{3}} / {nb_epochs}]")

        start_time = time.time() # Time of an epoch 

        for i, (x_ind_pad,y_ind_pad,lengths_x,lengths_y,x,y,x_token,y_token) in enumerate(train_loader):
            
            #! Garbage Collector 
            garbage_collect()

            
            #* Pushing the data to the device 
            x_ind_pad,y_ind_pad,lengths_x,lengths_y = x_ind_pad.to(device),y_ind_pad.to(device),lengths_x,lengths_y


            optimizer.zero_grad() # Set the grads to zero


            if not transformer:
                #* Forward through the model
                y_hat,y_pred  =  model(x_ind_pad,lengths_x,y_ind_pad,lengths_y)
                
                # print(y_hat.size())
                # print(y.size())

                # Compute the loss => the loss function works only on 2d inputs 
                # y_hat of size  [m,max_seq_length_y,embedding_out_size] 
                y_hat = y_hat[:,1:].contiguous().view(-1,y_hat.shape[-1]) # to remove the first token and convert to size [m*(max_seq_length_y-1) ,embedding_out_size]
                y_ind_pad = y_ind_pad[:,1:].contiguous().view(-1) 
            else :
                #* Forward through the model
                y_hat,y_pred  =  model(x_ind_pad,lengths_x,y_ind_pad[:, :-1],lengths_y) #! remove last token , the transformer learns to predict the next one 
                
                # print('--------------------------------------------')
                # print(y_hat.size())
                # print(y_ind_pad.size())


                # # Compute the loss => the loss function works only on 2d inputs 
                # # y_hat of size  [m,max_seq_length_y,embedding_out_size] 
                y_hat = y_hat.view(-1,y_hat.shape[-1]) # y_hat don't contain the first token 
                y_ind_pad = y_ind_pad[:,1:].contiguous().view(-1)
                # print(y_hat.size())
                # print(y_ind_pad.size())
                # print('--------------------------------------------')
            loss = loss_fct(y_hat,y_ind_pad) 

            # Backward step 
            loss.backward() # Compute the gradiants for all the parameters 
            
            # Clipping the gradient to prevent it from exploding (common with RNN)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            optimizer.step() # Update the parameters 
            

            #* Accumulating the loss for the epoch
            epoch_loss += loss.detach().item() * x_ind_pad.size()[0]
            epoch_elem_count += x_ind_pad.size()[0] # Number of elements in the batch 
            
            if i%exec_options['nb_steps']==0 and exec_options['step_fct'] != None:
                print(f'\t ->> End of each {exec_options["nb_steps"]} step fct  <>',end=' ')
                exec_options['step_fct']()

            #* Displaying the average loss each track_loss_train_steps steps  (batch)
            if tracking_params["debug_loss"] :                
                #* Last perplexity 
                if  (i % tracking_params["track_loss_train_steps"] == 0 or i == nb_batch-1):
                    print(f"\t -> Step:{str(i):{6}} ,  Loss = {loss.detach().item() :.6f} , PPL = {math.exp(loss.detach().item()):.6f}")
                    if tensorboard_options["tensorboard"] : 
                        writer.add_scalar(f"""Training Loss -{tensorboard_options["label"]}- (steps)""",loss.detach().item() ,epoch*nb_batch+i)
                        writer.add_scalar(f"""Training PPL -{tensorboard_options["label"]}- (steps)""",math.exp( loss.detach().item()),epoch*nb_batch+i)



            # Recoding the average loss on the train data and displaying it to the console 
            # if tracking_params["track_loss_train"] or tracking_params["debug_loss"] :
            #     tmp_loss_train += epoch_loss
            #     tmp_loss_train_count += 1
                
            #     if i % tracking_params["track_loss_train_steps"] == 0 or i == nb_batch-1:
                    
            #         if tracking_params["debug"] and tracking_params["debug_loss"]:
            #             if tracking_params["average"]:  # Recording the average loss
            #                 print(f"\t -> Step:{str(i):{6}} ,  Loss = {tmp_loss_train / tmp_loss_train_count:.6f} ")
            #                 if tracking_params["track_loss_train"]:
            #                     loss_train.append(tmp_loss_train / tmp_loss_train_count)
            #                     if tensorboard_options["tensorboard"] : 
            #                         writer.add_scalar(f"""Training Loss -{tensorboard_options["label"]}-""",tmp_loss_train / tmp_loss_train_count,epoch*nb_batch+i)
            #             else: # Recording the last loss 
            #                 print(f"\t -> Step:{str(i):{6}} ,  Loss = {epoch_loss:.6f} ")
            #                 if tracking_params["track_loss_train"]:
            #                     loss_train.append(epoch_loss)
            #                     if tensorboard_options["tensorboard"] : 
            #                         writer.add_scalar(f"""Training Loss -{tensorboard_options["label"]}- (steps)""",tmp_loss_train / tmp_loss_train_count,epoch*nb_batch+i)
                    
            #         tmp_loss_train = 0
            #         tmp_loss_train_count = 0

        end_time = time.time()
        training_time = end_time - start_time # time of a single epoch
        
        if tracking_params["track_loss_train"]:
            loss_train.append(epoch_loss/epoch_elem_count)



        # Recording the loss and evaluation on the validation data at the end of each epoch 
        if tracking_params["validation"]:
            print("Evaluating the model on validation")
            model.eval()
            # Computing the loss and evaluating the model 
            with torch.no_grad():
                # Evaluate on one batch of test data
                for  i, (x_ind_pad,y_ind_pad,lengths_x,lengths_y,x,y,x_token,y_token) in enumerate(val_loader):
                    # Pushing the data to the device
                    x_ind_pad,y_ind_pad,lengths_x,lengths_y = x_ind_pad.to(device),y_ind_pad.to(device),lengths_x,lengths_y



                    if not transformer:
                        #* Forward through the model
                        y_hat,y_pred  =  model(x_ind_pad,lengths_x,y_ind_pad,lengths_y)
                        
                        # print(y_hat.size())
                        # print(y.size())
                        loss = loss_fct(y_hat[:,1:].contiguous().view(-1,y_hat.shape[-1]),y_ind_pad[:,1:].contiguous().view(-1) ) 
                        eval_ = eval_fct(y_pred,y_ind_pad)
                    
                    else :
                        #* Forward through the model
                        y_hat,y_pred  =  model(x_ind_pad,lengths_x,y_ind_pad[:, :-1],lengths_y) #! remove last token , the transformer learns to predict the next one 
                        # print('--------------------------------------------')
                        # print(y_hat.size())
                        # print(y.size())
                        # print(y_hat.view(-1,y_hat.shape[-1]))
                        # print(y_ind_pad[:,1:].contiguous().view(-1))
                        # print('--------------------------------------------')

                        loss = loss_fct( y_hat.view(-1,y_hat.shape[-1]),y_ind_pad[:,1:].contiguous().view(-1)) 
                        eval_ = eval_fct(y_pred,y_ind_pad[:,1:])


                    # Forward through the model
                    # y_hat,y_pred  =  model(x_ind_pad,lengths_x,y_ind_pad,lengths_y)

                    # Compute the loss
                    # loss = loss_fct(y_hat,y)
                    # y_hat = y_hat[1:].view(-1,y_hat.shape[-1]) # to remove the first token and convert to size [m*(max_seq_length_y-1) ,embedding_out_size]
                    # y_ind_pad = y_ind_pad[1:].view(-1) 
        

                    # Evaluate the model
                    # eval_ = eval_fct(y_pred,y_ind_pad)


                    loss_val.append(loss.detach().item())
                    eval_loss.append(eval_)

                    if tensorboard_options["tensorboard"] : 
                        writer.add_scalar(f"""Val Loss -{tensorboard_options["label"]}- """,loss.detach().item(),epoch)
                        writer.add_scalar(f"""Val eval  -{tensorboard_options["label"]}- """,eval_,epoch)
                        writer.add_scalar(f"""Val ppl  -{tensorboard_options["label"]}- """,math.exp(loss),epoch)


                    break #! loop only through the first batch of the validation data , set the batch size to the size of validation data 


            if tracking_params["debug"] :
                print(f" -> End of epoch,  Train Loss = {epoch_loss / epoch_elem_count:.6f} PPL={math.exp(epoch_loss/epoch_elem_count):.6f}, Val Loss = {loss.item():.6f} PPL={math.exp(loss.item()):.6f} Eval ={eval_} , t= {training_time:.2f} seconds \n")
                
        elif tracking_params["debug"] :
                print(f" -> End of epoch,   Train Loss = {epoch_loss / epoch_elem_count:.6f} PPL={math.exp(epoch_loss/epoch_elem_count):.6f}, t= {training_time:.2f} seconds\n")
        
        
        if tensorboard_options["tensorboard"] : 
            writer.add_scalar(f"""Training Loss -{tensorboard_options["label"]}- (epochs)""",epoch_loss / epoch_elem_count,epoch)
            writer.add_scalar(f"""Training PPL -{tensorboard_options["label"]}- (epochs)""",math.exp(epoch_loss/epoch_elem_count),epoch)


        if exec_options['epoch_fct'] != None:
            print(' ->> End of epoch fct <>',end=' ')
            exec_options['epoch_fct']()
        
        epoch_loss = 0
        epoch_elem_count = 0
        

        #? Checkpoints 
        if checkpoint_options["checkpoint"] and (epoch+1) % checkpoint_options["checkpoint_epoch"] == 0: 
            # Saving the model, optimizer and epoch 
            checkpoint ={
                "epoch" : epoch, #current epoch
                "characteristic" : model.get_model_characteristic(),
                "model_state" : model.state_dict(),
                "optimizer_state" : optimizer.state_dict(),
            }
            torch.save(checkpoint,os.path.join(checkpoint_options["checkpoint_folder_path"],f"checkpoint_{model.get_model_name()}_{epoch+1}.pth"))

        if checkpoint_options["checkpoint"] and epoch == nb_epochs-1: # Last checkpoint 
            # Saving the model, optimizer and epoch 
            checkpoint ={
                "epoch" : epoch, #current epoch
                "characteristic" : model.get_model_characteristic(),
                "model_state" : model.state_dict(),
                "optimizer_state" : optimizer.state_dict()
            }
            torch.save(checkpoint,os.path.join(checkpoint_options["checkpoint_folder_path"],f"checkpoint_{model.get_model_name()}_{epoch+1}_final.pth"))

    
    return loss_train,loss_val,eval_loss


# from memory_profiler import profile

# @profile
def work():
    
    # import tracemalloc

    # # Start tracing memory
    # tracemalloc.start()

    #? Data 

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
    
    train_dataset_size_mb = sys.getsizeof(train_dataset) / (1024 ** 2)
    valid_dataset_size_mb = sys.getsizeof(valid_dataset) / (1024 ** 2)

    print(f"Size of train dataset {train_dataset_size_mb:.5f} MB")
    print(f"Size of valid dataset {valid_dataset_size_mb:.5f} MB")



    #* Input and output vocabulary  
    vocab_in,vocab_out = train_dataset.get_vocab()
    
    #* Dataloader   
    train_loader = DataLoader(dataset=train_dataset,batch_size=16,shuffle=True,collate_fn=collate_fn)
    valid_loader = DataLoader(dataset=valid_dataset,batch_size=16,shuffle=False,collate_fn=collate_fn)
    

    #* For the train function 
    data = (train_loader,valid_loader)


    #? Device to train on GPU or CPU 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(torch.cuda.get_device_properties(device))



    #? Seq2Seq model 
    #* Model 
    
    #* Seq2Seq RNN
    # transformer = False
    # seq2seq = Seq2Seq(len(vocab_in),len(vocab_out),256,256,512,4,0.5,device).to(device)

    #* Seq2Seq RNN + Attention 
    # transformer = False
    # seq2seq = Seq2SeqAttention(len(vocab_in),len(vocab_out),300,300,1024,1,0.0,device).to(device)
    
    #* Seq2Seq Transformer  
    transformer = True
    seq2seq = Seq2SeqTransformer(128,len(vocab_in),len(vocab_out),0,8,6,6,4,0.1,device,200)
    
    #! Add to reduce the model size  .to(dtype=torch.bfloat16)
    
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

    # input("Model CReated")


    seq2seq = seq2seq.to(device)

    torch.cuda.empty_cache()
    gc.collect()
    # input("MOdel Moved")
    
    # Printing the size of the model 
    param_size = 0
    for param in seq2seq.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in seq2seq.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

    # model_size_mb = sum(sys.getsizeof(param) for param in seq2seq.state_dict().values()) / (1024.0 ** 2)
    # print(f"Size of the entire model: {model_size_mb:.4f} MB")

    #* Initializing the model 
    seq2seq.apply(params_initializer(0.1))
    print(seq2seq)
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

    #* Summary of the model 
    train_dataloader_iter = iter(train_loader)
    x_ind_pad,y_ind_pad,lengths_x,lengths_y,x,y,x_token,y_token = next(train_dataloader_iter)

    y_hat,y_pred =  seq2seq(x_ind_pad.to(device),lengths_x,y_ind_pad.to(device),lengths_y)
    # summary(seq2seq,input_data =[x_ind_pad.to(device),lengths_x,y_ind_pad.to(device),lengths_y],device="cuda")



    

    #? Loss function 
    criterion = nn.CrossEntropyLoss(ignore_index = 0)

    #? Eval function 
    eval_bleu_score = construct_bleu_score(vocab_out,0,9777,9778)

    #? Optimizer 
    optimizer = torch.optim.Adam(seq2seq.parameters(),lr=3e-4)

    #? Training params 
    nb_epochs = 1
    tracking_params ={
            "track_loss_train" : False, # Save loss on train data
            "track_loss_train_steps" : 20, # save loss each. .. step
            "validation" : True, # Save loss on valid data 
            "debug" : True, #
            "debug_loss" : True, # Dislay the evolution of loss during an epoch 
            "average":True
        }

    eval_fct = eval_bleu_score

    checkpoint_options = {
        "checkpoint" : True,
        "checkpoint_folder_path" : "./checkpoints",
        "checkpoint_epoch":3
    }
    
    writer = SummaryWriter("runs/seq2seq")
    tensorboard_options = {"tensorboard" : True, 
                "writer" : writer,
                "label" : "Seq2Seq" }
    

    exec_options = {
        "epoch_fct":step_fct,
        "step_fct":step_fct,
        "nb_steps":20
    } 

    loss_train,loss_val,eval_loss = train(seq2seq,data,criterion,optimizer,nb_epochs,device,tracking_params,eval_fct,checkpoint_options,
                                        tensorboard_options,exec_options,transformer=transformer)

    print(loss_train,loss_val,eval_loss)

    writer.close() 


if __name__ == "__main__":
    print("=======> Train.py <======")

    work()




    # def sizeof_fmt(num, suffix='B'):
    #     ''' 
    #         by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified
    #     '''
    #     for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
    #         if abs(num) < 1024.0:
    #             return "%3.1f %s%s" % (num, unit, suffix)
    #         num /= 1024.0
    #     return "%.1f %s%s" % (num, 'Yi', suffix)

    # for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
    #                         locals().items())), key= lambda x: -x[1])[:10]:
    #     print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
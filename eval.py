from torchtext.data.metrics import bleu_score

def construct_bleu_score(vocab,pad_indx,sos_indx,eos_indx,n_gram=4):
    """
        Bleu score evaluation function constructor 
            Inputs :
                vocab -> [int] : Vocabulary 
                pad_indx -> int : index of the padding token in the vocabulary 
                sos_indx -> int : index of the start of sequence token in the vocabulary 
                eos_indx -> int : index of the end of sequence token in the vocabulary 
                n_gram -> int :  maximum of n-gram to use 
            Output :
                eval_bleu_score -> function : function that computes the bleu score 
    """
    def eval_bleu_score(y_pred, y_target):
        """
            Computes the bleu score 
            Inputs :
                y_pred -> [[int]] : predicted outputs (indexes of tokens) of size [m,max_seq_length]
                y_target -> [[int]] : target outputs (indexes of tokens) of size [m,max_seq_length]
            Output :
                bleu score -> int 
        """
        candidate = [] 
        for m in range(y_pred.size()[0]):
            tmp_str = []
            for token in range(y_pred.size()[1]):
                if y_pred[m,token].item() not in [pad_indx,sos_indx,eos_indx]: # pad , sos and eos tokens are excluded in the computation 
                    tmp_str.append(vocab[y_pred[m,token].item()])
            candidate.append(tmp_str)
    
        reference = [] 
        for m in range(y_target.size()[0]):
            tmp_str = []
            for token in range(y_target.size()[1]):
                if y_target[m,token].item() not in [pad_indx,sos_indx,eos_indx]: 
                    tmp_str.append(vocab[y_target[m,token].item()])
            reference.append([tmp_str])
        return bleu_score(candidate, reference,max_n=n_gram)

    return eval_bleu_score 
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import editdistance as ed

def CreateOnehotVariable(input_x, encoding_dim=63) :
    if type(input_x) is Variable :
        input_x = input_x.data
    input_type = type(input_x)
    batch_size = input_x.size(0)
    time_steps = input_x.size(1)
    input_x = input_x.unsqueeze(2).type(torch.LongTensor)
    onehot_x = Variable(torch.LongTensor(batch_size, time_steps, encoding_dim).zero_().scatter_(-1,input_x, 1)).type(input_type)

    return onehot_x

def TimeDistributed(input_module, input_x) :
    batch_size = input_x.size(0)
    time_steps = input_x.size(1)
    reshaped_x = input_x.contiguous().view(-1, input_x.size(-1))
    output_x = input_module(reshaped_x)
    return output_x.view(batch_size, time_steps, -1)


# LetterErrorRate function
# Merge the repeated prediction and calculate editdistance of prediction and ground truth
def LetterErrorRate(pred_y, true_y, data):
    ed_accumalate = []
    for p, t in zip(pred_y, true_y):
        compressed_t = [w for w in t if (w != 1 and w != 0)]

        compressed_p = []
        for p_w in p:
            if p_w == 0:
                continue
            if p_w == 1:
                break
            compressed_p.append(p_w)
        ed_accumalate.append(ed.eval(compressed_p, compressed_t) / len(compressed_t))
    return ed_accumalate


def label_smoothing_loss(pred_y, true_y, label_smoothing=0.1):
    # Self defined loss for label smoothing
    # pred_y is log-scaled and true_y is one-hot format padded with all zero vector
    assert pred_y.size() == true_y.size()
    seq_len = torch.sum(torch.sum(true_y, dim=-1), dim=-1, keepdim=True)

    # calculate smoothen label, last term ensures padding vector remains all zero
    class_dim = true_y.size()[-1]
    smooth_y = ((1.0 - label_smoothing) * true_y + (label_smoothing / class_dim)) * torch.sum(true_y, dim=-1,
                                                                                              keepdim=True)

    loss = - torch.mean(torch.sum((torch.sum(smooth_y * pred_y, dim=-1) / seq_len), dim=-1))

    return loss

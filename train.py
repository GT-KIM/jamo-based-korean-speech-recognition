import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


import torch
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import random
import Levenshtein as Lev
from utils import *
from las_old.model import *
from model import EncoderRNN, DecoderRNN, Seq2Seq
from data_loader.loader_utils import *
from data_loader.unicode import *
from dataset import AudioDataLoader, SpectrogramDataset, BucketingSampler


parser = argparse.ArgumentParser(description='LAS_train')
parser.add_argument('--rootpath', type=str, default="/sde1/speech_recognition/")
parser.add_argument('--dataset_path', type=str, default="/sde1/speech_recognition/Data/")

parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=8)
#---------------
parser.add_argument('--encoder_size', type=int, default=512)
parser.add_argument('--encoder_layers', type=int, default=3)
parser.add_argument('--dropout_rate', type=float, default=0.3)
parser.add_argument('--no-bidirectional',dest='bidirectional', action='store_false', default=True)
parser.add_argument('--rnn_type', type=str, default='lstm')
parser.add_argument('--max_len', type=int, default=1000)
parser.add_argument('--decoder_size', type=int, default=512)
parser.add_argument('--decoder_layers', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--load_model', type=bool, default=True)
parser.add_argument('--mode', type=str, default="train")

parser.add_argument('--save_folder', type=str, default="/sde1/speech_recognition/model/")
parser.add_argument('--model_path', type=str, default="/sde1/speech_recognition/model/las.pth")
parser.add_argument('--best_model_path', type=str, default="/sde1/speech_recognition/model/las_best.pth")
parser.add_argument('--finetune', type=bool, default=True)

parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--seed', type=int, default=81)
parser.add_argument('--max_norm', type=int, default=400)
parser.add_argument('--teacher_forcing', type=float, default=1.0)
parser.add_argument('--learning_anneal', type=float,default=1.1)
args = parser.parse_args()

def char_distance(ref, hyp) :
    ref = ref.replace(' ', '')
    hyp = hyp.replace(' ', '')

    dist = Lev.distance(hyp, ref)
    length = len(ref.replace(' ', ''))

    return dist, length

def get_distance(ref_labels, hyp_labels) :
    ref_labels = ref_labels.cpu().numpy()
    hyp_labels = hyp_labels.cpu().numpy()

    total_dist = 0
    total_length = 0
    transcripts = []
    for i in range(len(ref_labels)) :
        ref_sentence = list()
        for j in ref_labels[i]:
            ref_sentence.append(j)
            if j == 71:
                break
        ref_sentence = np.array(ref_sentence)

        hyp_sentence = list()
        for j in hyp_labels[i]:
            hyp_sentence.append(j)
            if j == 71:
                break
        hyp_sentence = np.array(hyp_sentence)

        ref = join_jamos(label_to_jamo(ref_sentence))
        hyp = join_jamos(label_to_jamo(hyp_sentence))
        #print('{hyp}\n{ref}'.format(hyp=hyp, ref=ref))
        transcripts.append('{hyp}\t{ref}'.format(hyp=hyp, ref=ref))

        if len(ref_sentence) >= len(hyp_sentence):
            temp = np.zeros(len(ref_sentence))
            temp[:len(hyp_sentence)] = hyp_sentence
            hyp_sentence = temp
        else:
            temp = np.zeros(len(hyp_sentence))
            temp[:len(ref_sentence)] = ref_sentence
            ref_sentence = temp

        dist = np.count_nonzero(ref_sentence - hyp_sentence)

        length = len(ref_sentence)
        total_dist += dist
        total_length += length
    return total_dist, total_length, transcripts


def train(model, data_loader, criterion, optimizer, device, epoch,
          max_norm=400, teacher_forcing_ratio = 1) :
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0

    model.train()
    for i_batch, (data) in enumerate(data_loader) :
        batch_x, batch_y, feat_lengths, script_lengths = data
        optimizer.zero_grad()

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        feat_lengths = feat_lengths.to(device)

        src_len = batch_y.size(1)
        target = batch_y[:, 1:]

        logit = model(batch_x, feat_lengths, batch_y, teacher_forcing_ratio=teacher_forcing_ratio)

        logit = torch.stack(logit, dim=1).to(device)
        y_hat = logit.max(-1)[1]

        loss = criterion(logit.contiguous().view(-1,logit.size(-1)), target.contiguous().view(-1))
        total_loss += loss.item()
        total_num += sum(feat_lengths).item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        dist, length, _ = get_distance(target, y_hat)
        total_dist += dist
        total_length += length
        cer = float(dist / length) * 100

        total_sent_num += batch_y.size(0)
        if i_batch % 1000 == 0 and i_batch != 0 :
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(state, args.model_path)

        print('Epoch: [{0}][{1}/{2}]\t Loss {loss:.4f}\t Cer {cer:.4f}'.format((epoch + 1),(i_batch+1), len(train_sampler), loss=loss, cer=cer))

    return total_loss / total_num, (total_dist / total_length) * 100

def evaluate(model, data_loader, criterion, device, save_output=False):
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0
    transcripts_list = []

    model.eval()
    with torch.no_grad() :
        for i_batch, (data) in tqdm(enumerate(data_loader), total=len(data_loader)) :
            batch_x, batch_y, feat_lengths, script_lengths = data

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            feat_lengths = feat_lengths.to(device)

            src_len = batch_y.size(1)
            target = batch_y[:, 1:]

            logit = model(batch_x, feat_lengths, None, teacher_forcing_ratio=0.0)
            logit = torch.stack(logit, dim=1).to(device)
            y_hat = logit.max(-1)[1]

            logit = logit[:,:target.size(1),:]
            loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
            total_loss += loss.item()
            total_num += sum(feat_lengths).item()

            dist, length, transcript = get_distance(target, y_hat)
            cer = float(dist / length) * 100

            total_dist += dist
            total_length += length
            if save_output == True :
                transcripts_list.append(transcript)
            total_sent_num += target.size(0)
    aver_loss = total_loss / total_num
    aver_cer = float(total_dist / total_length) * 100
    return aver_loss, aver_cer, transcripts_list

if __name__ == "__main__" :

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    SOS_token = 70
    EOS_token = 71
    PAD_token = 0

    device = torch.device('cuda' if args.cuda else 'cpu')

    batch_size = args.batch_size

    train_dataset = SpectrogramDataset(dataset_path = args.dataset_path, data_list = args.rootpath + "train_list.csv")
    train_sampler = BucketingSampler(train_dataset, batch_size=batch_size)
    train_loader = AudioDataLoader(train_dataset, num_workers=4, batch_sampler=train_sampler)

    test_dataset= SpectrogramDataset(dataset_path = args.dataset_path, data_list = args.rootpath + "valid_list.csv")
    test_loader = AudioDataLoader(test_dataset, num_workers=4, batch_size=1)

    input_size = 80
    enc = EncoderRNN(input_size, args.encoder_size, n_layers=args.encoder_layers,
                     dropout_p = args.dropout_rate, bidirectional = args.bidirectional,
                     rnn_cell=args.rnn_type, variable_lengths=False)
    dec = DecoderRNN(72, args.max_len, args.decoder_size, args.encoder_size,
                     SOS_token, EOS_token, n_layers=args.decoder_layers,
                     rnn_cell=args.rnn_type, dropout_p = args.dropout_rate, bidirectional_encoder=args.bidirectional)
    model = Seq2Seq(enc,dec)
    train_model = nn.DataParallel(model)

    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)

    optim_state=None
    if args.load_model :
        print("Loading checkpoint %s" % args.model_path)
        state = torch.load(args.model_path)
        train_model.load_state_dict(state['model'])
        print("Model loaded")

        if not args.finetune :
            optim_state = state['optimizer']

    train_model = train_model.to(device)
    optimizer = optim.Adam(train_model.parameters(), lr=args.lr, weight_decay = 1e-5)
    if optim_state is not None :
        train_model.load_state_dict(optim_state)

    criterion = nn.CrossEntropyLoss(reduction='mean').to(device)

    print(model)
    print("Number of parameters: %d" % Seq2Seq.get_param_size((model)))


    if args.mode != "train" :
        test_loss, test_cer, transcripts_list = evaluate(model, test_loader, criterion, device,
                                                         save_output=True)
        for line in transcripts_list :
            print(line)
    else :
        best_cer = 1e10
        begin_epoch = 0

        for epoch in range(begin_epoch, args.epochs) :
            train_loss, train_cer = train(train_model, train_loader, criterion, optimizer, device,
                                          epoch, args.max_norm, args.teacher_forcing)

            cer_list = []
            test_loss, test_cer, transcripts_list = evaluate(model, test_loader, criterion, device, save_output=True)
            for line in transcripts_list:
                print(line)
            test_log = 'Test Summary Epoch : [{0}]\tAverage Loss {loss:.3f}\tAverage CER{cer:.3f}\t'.format(
                epoch + 1, loss = test_loss, cer=test_cer)
            print(test_log)

            cer_list.append(test_cer)

            if best_cer > cer_list[0] :
                print("Found better validated model, saving to %s" % args.model_path)
                state = {
                    'model' : model.state_dict(),
                    'optimizer' : optimizer.state_dict()
                }
                torch.save(state, args.best_model_path)
                best_cer = cer_list[0]

            for g in optimizer.param_groups :
                g['lr'] = g['lr'] / args.learning_anneal
            print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))
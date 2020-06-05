import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"


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
import librosa

parser = argparse.ArgumentParser(description='LAS_train')
parser.add_argument('--rootpath', type=str, default="/sde1/speech_recognition/")
parser.add_argument('--custompath', type=str, default="./data/")

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
parser.add_argument('--mode', type=str, default="test")

parser.add_argument('--save_folder', type=str, default="/sde1/speech_recognition/model/")
parser.add_argument('--model_path', type=str, default="/sde1/speech_recognition/model/las.pth")
parser.add_argument('--best_model_path', type=str, default="/sde1/speech_recognition/model/las_best.pth")
parser.add_argument('--finetune', type=bool, default=True)

parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--seed', type=int, default=81)
parser.add_argument('--max_norm', type=int, default=400)
parser.add_argument('--teacher_forcing', type=float, default=1.0)
parser.add_argument('--learning_anneal', type=float,default=1.1)

parser.add_argument('--sample_rate', type=int, default=16000)
parser.add_argument('--n_fft', type=int, default=1024)
parser.add_argument('--window_size', type=int, default=.05)
parser.add_argument('--window_stride', type=int, default=.025)

args = parser.parse_args()

def load_custom_data(args) :
    datalist = os.listdir(args.custompath)
    input_data = list()
    for dataname in datalist :
        speech_data, _ = librosa.load(args.custompath + dataname, sr=16000)
        speech_data = speech_data / np.max(speech_data)
        mel_data = get_mel_feature(speech_data, args)
        mel_data, _ = librosa.magphase(mel_data)

        text = os.path.splitext(dataname)[0]
        filtered_data = sentence_filter(text)
        jamo = split_syllables(filtered_data)
        label = np.array(jamo_to_label(jamo))
        input_data.append([mel_data, filtered_data, label])
    return input_data

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
        print('{hyp}\n{ref}'.format(hyp=hyp, ref=ref))
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


def demo(model, test_data, criterion, device, save_output=False):
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0
    transcripts_list = []

    model.eval()
    with torch.no_grad() :
        for i in range(len(test_data)) :
            batch_x = test_data[i][0][np.newaxis,np.newaxis,:,:]
            batch_y = test_data[i][1]
            target = test_data[i][2][np.newaxis, :]
            feat_lengths = np.array([batch_x.shape[3]])

            batch_x = torch.from_numpy(batch_x)
            batch_x = batch_x.to(device)

            feat_lengths = torch.from_numpy(feat_lengths)
            feat_lengths = feat_lengths.to(device)

            target = target[:,1:]
            target = torch.from_numpy(target)
            target = target.to(device)
            logit = model(batch_x, feat_lengths, None, teacher_forcing_ratio=0.0)
            logit = torch.stack(logit, dim=1).to(device)
            y_hat = logit.max(-1)[1]

            dist, length, transcript = get_distance(target, y_hat)
            #cer = float(dist / length) * 100

            total_dist += dist
            total_length += length
            if save_output == True :
                transcripts_list.append(transcript)
            total_sent_num += target.size(0)
    aver_loss = 0#total_loss / total_num
    aver_cer = 0#float(total_dist / total_length) * 100
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

    test_data = load_custom_data(args)

    if args.mode != "train" :
        test_loss, test_cer, transcripts_list = demo(model, test_data, criterion, device,
                                                         save_output=True)
        for line in transcripts_list :
            print(line)
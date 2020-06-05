import os
import math
import torch
import librosa
import numpy as np
import scipy.signal
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

class SpectrogramDataset(Dataset):
    def __init__(self, dataset_path, data_list):
        super(SpectrogramDataset, self).__init__()
        """
        Dataset loads data from a list contatining wav_name, transcripts, speaker_id by dictionary.
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds.
        :param data_list: List of dictionary. key : "wav", "text", "speaker_id"
        :param char2index: Dictionary mapping character to index value.
        :param sos_id: Start token index.
        :param eos_id: End token index.
        :param normalize: Normalized by instance-wise standardazation.
        """
        self.data_list = pd.read_csv(data_list)
        self.size = len(self.data_list)
        self.dataset_path = dataset_path

    def __getitem__(self, index):
        name = self.data_list.iloc[index, 0]
        mel_path = os.path.join(self.dataset_path,'mel', name)
        mel_spectrogram =np.load(mel_path)
        label_path = os.path.join(self.dataset_path,'label', name)
        label = np.load(label_path)
        return torch.from_numpy(mel_spectrogram), torch.from_numpy(label)

    def __len__(self):
        return self.size

def _collate_fn(batch):
    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    seq_lengths = [s[0].size(1) for s in batch]
    target_lengths = [len(s[1]) for s in batch]

    max_seq_size = max(seq_lengths)
    max_target_size = max(target_lengths)

    feat_size = batch[0][0].size(0)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, 1, feat_size, max_seq_size)
    targets = torch.zeros(batch_size, max_target_size).to(torch.long)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        seqs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    seq_lengths = torch.IntTensor(seq_lengths)
    return seqs, targets, seq_lengths, target_lengths

class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class BucketingSampler(Sampler):
    def __init__(self, data_source, batch_size=1):
        """
        Samples batches assuming they are in order of size to batch similarly sized samples together.
        """
        super(BucketingSampler, self).__init__(data_source)
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i:i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)

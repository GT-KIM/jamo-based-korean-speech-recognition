import librosa
from librosa.display import specshow
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd

import argparse
from tqdm import tqdm, trange
from data_loader.loader_utils import *

parser = argparse.ArgumentParser(description='LAS_preprocess')
parser.add_argument('--rootpath', type=str, default="/media/super/Samsung_T5/Speech Recognition/")
parser.add_argument('--savepath', type=str, default="/sde1/speech_recognition/Data/")
parser.add_argument('--sample_rate', type=int, default=16000)
parser.add_argument('--n_fft', type=int, default=1024)
parser.add_argument('--window_size', type=int, default=.05)
parser.add_argument('--window_stride', type=int, default=.025)
parser.add_argument('--feature_extract', type=bool, default=False)
parser.add_argument('--train_test_split', type=bool, default=True)
args = parser.parse_args()

if __name__ == "__main__" :

    rootpath = args.rootpath
    savepath = args.savepath

    if args.feature_extract :
        idx = 0
        for (path, dir, files) in os.walk(rootpath):
            for filename in files:
                ext = os.path.splitext(filename)[-1]
                if ext == '.pcm':
                    tgt_path = "%s%s/" % (savepath,"pcm")
                    if not os.path.isdir(tgt_path):
                        os.makedirs(tgt_path)

                    featurename = filename[:-4] + ".npy"
                    feature_path = "%s%s/" % (savepath,"mel")
                    if not os.path.isdir(feature_path):
                        os.makedirs(feature_path)

                    shutil.copy("%s/%s" % (path, filename), "%s/%s" % (tgt_path, filename))

                    speech_data = np.memmap("%s/%s" % (path, filename), dtype='h', mode='r')
                    speech_data = speech_data / np.max(speech_data)
                    mel_data = get_mel_feature(speech_data, args)
                    mel_data, _ = librosa.magphase(mel_data)
                    np.save("%s/%s" % (feature_path, featurename), mel_data)
                    #specshow(librosa.amplitude_to_db(mel_data), cmap=cm.jet)
                    #plt.show()

                elif ext == '.txt':
                    src = "%s/%s" % (path, filename)
                    tgt_path = "%s%s/" % (savepath,"text")
                    if not os.path.isdir(tgt_path):
                        os.makedirs(tgt_path)

                    filename_label = filename[:-4] + ".npy"
                    label_path = "%s%s/" % (savepath,"label")
                    if not os.path.isdir(label_path):
                        os.makedirs(label_path)

                    f = open(src, 'r', encoding='cp949')
                    w = open("%s/%s" % (tgt_path, filename), 'w', encoding='utf-8')
                    raw_data = f.readline()
                    filtered_data = sentence_filter(raw_data)
                    w.write(filtered_data)
                    jamo = split_syllables(filtered_data)
                    label = jamo_to_label(jamo)
                    f.close()
                    w.close()
                    np.save("%s/%s" % (label_path, filename_label), label)
                idx += 1
                if idx % 115000 == 0 :
                    print("%d percent finished"%int(idx/11500))

    if args.train_test_split :
        total_len = 622545
        split_idx = np.arange(1, total_len+1)
        np.random.shuffle(split_idx)

        train_dict = {'name': []}
        test_dict = {'name': []}
        for i in range(total_len):
            if i < int(total_len * 0.98):
                train_dict['name'].append("KsponSpeech_" + '{0:06}'.format(split_idx[i]) + ".npy")
            else:
                test_dict['name'].append("KsponSpeech_" + '{0:06}'.format(split_idx[i]) + ".npy")

        train_df = pd.DataFrame(train_dict)
        test_df = pd.DataFrame(test_dict)

        train_df.to_csv(savepath + "../train_list.csv", encoding='utf-8', index=False)
        test_df.to_csv(savepath + "../test_list.csv", encoding='utf-8', index=False)
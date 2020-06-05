#-*- coding:utf-8 -*-
import os
import shutil
import re
import sys
import numpy as np
import librosa
import torch
from data_loader.unicode import *
def filenum_padding(filenum):
    if filenum < 10:
        return '00000' + str(filenum)
    elif filenum < 100:
        return '0000' + str(filenum)
    elif filenum < 1000:
        return '000' + str(filenum)
    elif filenum < 10000:
        return '00' + str(filenum)
    elif filenum < 100000:
        return '0' + str(filenum)
    else:
        return str(filenum)

def get_path(path, fname, filenum, format):
    return path + fname + filenum + format


def bracket_filter(sentence):
    new_sentence = str()
    flag = False

    for ch in sentence:
        if ch == '(' and flag == False:
            flag = True
            continue
        if ch == '(' and flag == True:
            flag = False
            continue
        if ch != ')' and flag == False:
            new_sentence += ch
    return new_sentence


def special_filter(sentence):
    SENTENCE_MARK = ['?', '!']
    NOISE = ['o', 'n', 'u', 'b', 'l']
    EXCEPT = ['b','/', '+', '*', '-', '@', '$', '^', '&', '[', ']', '=', ':', ';', '.', ',', '\n']
    ENGLISH = {'A' : '에이','B' : '비', 'C':'씨', 'D':'디', 'E':'이','F':'에프',
               'G' : '쥐', 'H':'에이치', 'I':'아이', 'J' : '제이', 'K':'케이',
               'L':'엘','M':'엠', 'N':'엔','O':'오', 'P':'피', 'Q':'큐','R':'알',
               'S':'에스','T':'티','U':'유','V':'브이', 'W':'더블유', 'X' : '엑스',
               'Y':'와이', 'Z':'지'}
    english = {'a' : '에이','b' : '비', 'c':'씨', 'd':'디', 'e':'이','f':'에프',
               'g' : '쥐', 'h':'에이치', 'i':'아이', 'j' : '제이', 'k':'케이',
               'l':'엘','m':'엠', 'n':'엔','o':'오', 'p':'피', 'q':'큐','r':'알',
               's':'에스','t':'티','u':'유','v':'브이', 'w':'더블유', 'x' : '엑스',
               'y':'와이', 'z':'지'}
    number = {'0' : '영', '1' : '일', '2' : '이', '3' : '삼', '4' : '사', '5' : '오',
              '6' : '육', '7' : '칠', '8' : '팔', '9' : '구'}
    new_sentence = str()
    for idx, ch in enumerate(sentence):
        if ch not in SENTENCE_MARK:
            if idx + 1 < len(sentence) and ch in NOISE and sentence[idx + 1] == '/':
                continue
        if ch == '%' :
            new_sentence += u'퍼센트'
        elif ch == '#':
            new_sentence += u'샾'
        elif ch in ENGLISH :
            new_sentence += ENGLISH[ch]
        elif ch in english:
            new_sentence += english[ch]
        elif ch in number:
            new_sentence += number[ch]
        elif ch not in EXCEPT:
            new_sentence += ch
    pattern = re.compile(r'\s\s+')  # re : 정규식 패턴 더듬는 패턴을 '\s\s+'로 표현해놓은거 같은데 이거 제거하려고 쓰는듯
    new_sentence = re.sub(pattern, ' ', new_sentence.strip())
    return new_sentence


def sentence_filter(sentence):
    return special_filter(bracket_filter(sentence))

def sentence_to_target(sentence, char2id):
    target = ""
    for ch in sentence:
        target += (str(char2id[ch]) + ' ')
    return target[:-1]

def target_to_sentence(target, id2char):
    sentence = ""
    targets = target.split()

    for n in targets:
        sentence += id2char[int(n)]
    return sentence

def jamo_to_label(jamo):
    SPECIAL_LIST = [' ', '!', '?']
    CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ',
                     'ㅣ']
    JONGSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                     'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    ES_LIST = ['<sos>', '<eos>']

    LIST = [*SPECIAL_LIST, *CHOSUNG_LIST,*JUNGSUNG_LIST,*JONGSUNG_LIST, *ES_LIST]
    char_to_idx = {ch:i for i, ch in enumerate(LIST)}

    result_label = [char_to_idx[x] for x in jamo]
    label = [len(LIST) - 2,*result_label, len(LIST)-1]

    return label

def label_to_jamo(label):
    SPECIAL_LIST = [' ', '!', '?'] #3
    CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'] # 19
    JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ',
                     'ㅣ'] # 21
    JONGSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                     'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'] # 27
    ES_LIST = ['<eos>', '<sos>']

    LIST = [*SPECIAL_LIST, *CHOSUNG_LIST, *JUNGSUNG_LIST, *JONGSUNG_LIST, *ES_LIST]
    idx_to_char = {i: ch for i, ch in enumerate(LIST)}
    result_jamo = [idx_to_char[x] for x in label]

    return result_jamo

def get_mel_feature(wav, args) :
    spectrogram = librosa.stft(wav, n_fft=args.n_fft,
                                   hop_length=int(args.sample_rate*args.window_stride),
                                   win_length=int(args.sample_rate*args.window_size))
    mel_spectrogram = librosa.feature.melspectrogram(S=spectrogram, n_mels=80)
    mel_spectrogram = np.log1p(mel_spectrogram)
    mean = np.mean(mel_spectrogram)
    std = np.std(mel_spectrogram)
    mel_spectrogram -= mean
    mel_spectrogram /= std

    return mel_spectrogram



"""
https://github.com/neotune/python-korean-handler
"""
def text_to_label(test_keyword):
    # 유니코드 한글 시작 : 44032, 끝 : 55199
    BASE_CODE, CHOSUNG, JUNGSUNG = 44032, 588, 28
    # 초성 리스트. 00 ~ 18
    CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    CHOSUNG_LABEL = np.arange(len(CHOSUNG_LIST))
    # 중성 리스트. 00 ~ 20
    JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ',
                     'ㅣ']
    JUNGSUNG_LABEL = np.arange(len(JUNGSUNG_LIST)) + len(CHOSUNG_LIST)
    # 종성 리스트. 00 ~ 27 + 1(1개 없음)
    JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                     'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    JONGSUNG_LABEL = np.arange(len(JONGSUNG_LIST)) + len(CHOSUNG_LIST) + len(JUNGSUNG_LIST)

    LIST = [*CHOSUNG_LIST,*JUNGSUNG_LIST, *JONGSUNG_LIST]
    LABEL = [*CHOSUNG_LABEL, *JUNGSUNG_LABEL, *JONGSUNG_LABEL]
    print(LIST)
    print(LABEL)

    split_keyword_list = list(test_keyword)
    # print(split_keyword_list)

    result = list()
    result_label = list()
    for keyword in split_keyword_list:
        # 한글 여부 check 후 분리
        if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', keyword) is not None:
            char_code = ord(keyword) - BASE_CODE
            char1 = int(char_code / CHOSUNG)
            result.append(CHOSUNG_LIST[char1])
            # print('초성 : {}'.format(CHOSUNG_LIST[char1]))
            char2 = int((char_code - (CHOSUNG * char1)) / JUNGSUNG)
            result.append(JUNGSUNG_LIST[char2])
            # print('중성 : {}'.format(JUNGSUNG_LIST[char2]))
            char3 = int((char_code - (CHOSUNG * char1) - (JUNGSUNG * char2)))
            if not char3 == 0:
                result.append(JONGSUNG_LIST[char3])
            # print('종성 : {}'.format(JONGSUNG_LIST[char3]))
        else:
            result.append(keyword)
    # result
    print("".join(result))
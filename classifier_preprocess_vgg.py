from pathlib import Path
from tqdm import tqdm
from scipy.spatial import distance
from model import vgg

import pickle
import random
import numpy as np
import librosa
import argparse
import sys
import os

def load_wav(vid_path, sr, mode='train'):
    wav, sr_ret = sf.read(vid_path)
    extended_wav = np.append(wav, wav)
    if np.random.random() < 0.3:
        extended_wav = extended_wav[::-1]
    return extended_wav

def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T

def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=250, mode='train'):
    #print("starting loading a datum")
    #t1 = timelib.time()
    wav = load_wav(path, sr=sr, mode=mode)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    if mode == 'train':
        if time > spec_len:
            randtime = np.random.randint(0, time-spec_len)
            spec_mag = mag_T[:, randtime:randtime+spec_len]
        else:
            spec_mag = np.pad(mag_T, ((0, 0), (0, spec_len - time)), 'constant')
    else:
        spec_mag = mag_T
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    #print("finished loading a datum", timelib.time() - t1)
    return (spec_mag - mu) / (std + 1e-5)

def load_embed(path):
    specs = load_data(path, win_length=params['win_length'], sr=params['sampling_rate'],
                         hop_length=params['hop_length'], n_fft=params['nfft'],
                         spec_len=params['spec_len'], mode='eval')
    specs = np.expand_dims(np.expand_dims(specs, 0), -1)
    v = vgg_model.predict(specs)
    return v[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesses audio file for training purpose, extract feature by using pretrained model VGG",
    )
    parser.add_argument("-d", "--train_dir", type=Path, default="datasets_root/dataset/", help=\
        "Path to the directory containing your zalo training dataset (in which contains folders with label like '272-M-26', '632-M-27',...")
    parser.add_argument("-mvgg", "--models_dir_vgg", type=Path, default="encoder/saved_models/", help=\
        "Path to directory containing pretrained weights of VGG model")
    parser.add_argument("-o", "--out_dir", type=Path, default='train_data_vgg.pickle', help=\
        "Data filename serialized by Pickle, default: 'train_data_vgg.pickle'")

    args = parser.parse_args()

    vgg_params = {'dim': (257, None, 1),
          'nfft': 512,
          'spec_len': 250,
          'win_length': 400,
          'hop_length': 160,
          'n_classes': 5994,
          'sampling_rate': 16000,
          'normalize': True,
    }
    class VGGArgs:
        batch_size = 16
        net = 'resnet34s'
        ghost_cluster = 2
        vlad_cluster = 8
        bottleneck_dim = 512
        aggregation_mode = 'gvlad'
        loss='softmax'
    vgg_args = VGGArgs()
    vgg_model = vgg.vggvox_resnet2d_icassp(input_dim=vgg_params['dim'],
                                            num_class=vgg_params['n_classes'],
                                            mode='eval', args=vgg_args)
    vgg_model.load_weights(f'{args.models_dir_vgg}/vgg.h5', by_name=True)
    for path in tqdm(Path(args.train_dir).rglob('*.wav')):
        file = str(path)
        path = file.split('/')
        label = path[-2]
        label_list[label].append(file)
        if file not in data:
            data[file] = load_embed(file, encoder)

    with open(args.out_dir, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"Done! Data saved to {args.out_dir}")
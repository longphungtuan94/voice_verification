import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import distance
from model import vgg

import pickle
import random
import numpy as np
import soundfile as sf
import librosa
import argparse
import sys
import os

graph = None

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

def get_embed(path, vgg_model, params):
    global graph
    specs = load_data(path, win_length=params['win_length'], sr=params['sampling_rate'],
                         hop_length=params['hop_length'], n_fft=params['nfft'],
                         spec_len=params['spec_len'], mode='eval')
    specs = np.expand_dims(np.expand_dims(specs, 0), -1)
    with graph.as_default():
        v = vgg_model.predict(specs)
    return v[0]

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

def load_model_vgg(model_dir):
    global vgg_params
    global graph
    graph = tf.get_default_graph()
    vgg_model = vgg.vggvox_resnet2d_icassp(input_dim=vgg_params['dim'],
                                            num_class=vgg_params['n_classes'],
                                            mode='eval', args=vgg_args)
    vgg_model.load_weights(f'{model_dir}/vgg.h5', by_name=True)
    return vgg_model
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesses audio file for training purpose, extract feature by using pretrained model VGG",
    )
    parser.add_argument("-d", "--train_dir", type=Path, default="datasets_root/zalo/dataset", help=\
        "Path to the directory containing your zalo training dataset (in which contains folders with label like '272-M-26', '632-M-27',...")
    parser.add_argument("-mvgg", "--models_dir_vgg", type=Path, default="encoder/saved_models/", help=\
        "Path to directory containing pretrained weights of VGG model")
    parser.add_argument("-o", "--out_dir", type=Path, default='train_data_vgg.pickle', help=\
        "Data filename serialized by Pickle, default: 'train_data_vgg.pickle'")

    args = parser.parse_args()

    # load model
    vgg_model = load_model_vgg(args.models_dir_vgg)

    # extract feature and save
    data = {}
    for path in tqdm(Path(args.train_dir).rglob('*.wav')):
        file = str(path)
        data[file] = get_embed(file, vgg_model, vgg_params)

    with open(args.out_dir, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"Done! Data saved to {args.out_dir}")
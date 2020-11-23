from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.modelutils import check_model_paths
from encoder import inference as encoder
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import distance

import pickle
import random
import numpy as np
import librosa
import argparse
import sys
import os

def get_embed(fn, encoder):
    original_wav, sampling_rate = librosa.load(fn)
    preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    embed = encoder.embed_utterance(preprocessed_wav)
    return embed

def load_model_voice_clone(model_dir):
    encoder.load_model(Path(f'{model_dir}/pretrained.pt'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesses audio file for training purpose, extract feature by using pretrained model Real-Time-Voice-Cloning",
    )
    parser.add_argument("-d", "--train_dir", type=Path, default="datasets_root/zalo/dataset/", help=\
        "Path to the directory containing your zalo training dataset (in which contains folders with label like '272-M-26', '632-M-27',...")
    parser.add_argument("-mvc", "--models_dir_voice_clone", type=Path, default="encoder/saved_models/", help=\
        "Path to directory containing pretrained weights of Real-Time-Voice-Cloning model")
    parser.add_argument("-o", "--out_dir", type=Path, default='train_data_voice_clone.pickle', help=\
        "Data filename serialized by Pickle, default: 'train_data_voice_clone.pickle'")
    args = parser.parse_args()

    # load model
    load_model_voice_clone(args.models_dir_voice_clone)

    # extract feature and save
    data = {}
    for path in tqdm(Path(args.train_dir).rglob('*.wav')):
        file = str(path)
        data[file] = get_embed(file, encoder)

    with open(args.out_dir, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"Done! Data saved to {args.out_dir}!")
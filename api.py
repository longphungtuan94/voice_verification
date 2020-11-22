from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path
from scipy.spatial import distance

from model.network import build_model
from model.network import predict_verification
from model.network import load_weight
from encoder import inference as voice_clone_encoder
from classifier_preprocess_voice_clone import get_embed as get_embed_voice_clone
from classifier_preprocess_vgg import load_model_vgg
from classifier_preprocess_vgg import get_embed as get_embed_vgg
from classifier_preprocess_vgg import vgg_params

import numpy as np
import json
import os
AUDIO_STORAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)),"audio_storage")

if not os.path.isdir(AUDIO_STORAGE):  
    os.makedirs(AUDIO_STORAGE)

app = Flask(__name__)
voice_clone_encoder.load_model(Path(f'encoder/saved_models/pretrained.pt'))
clf = build_model()
load_weight(clf, 'weights/classifier.h5')
vgg_model = load_model_vgg('encoder/saved_models/')

def get_embed_final(path): 
    # concate of voice_clone embedding and vgg embedding
    global vgg_model
    e1 = get_embed_voice_clone(path, voice_clone_encoder)    
    e2 = get_embed_vgg(path, vgg_model, vgg_params)
    return np.concatenate((e2, e1))

def predict(audio_1, audio_2):
    global clf
    e1 = get_embed_final(audio_1)
    e2 = get_embed_final(audio_2)
    d1 = distance.euclidean(e1, e2)
    d2 = distance.cosine(e1, e2)
    batch = np.array([np.concatenate((e1, e2, np.array([d1, d2])), axis=0)])
    return np.argmax(predict_verification(clf, batch)[0])

@app.route("/api/predict", methods=['POST'])
def api_predict():
    assert 'audio_1' in request.files and 'audio_2' in  request.files, "audio not found!"
    audio_file_1 = request.files['audio_1'] # Required
    audio_file_2 = request.files['audio_2'] # Required
    if audio_file_1:
        filename_1 = secure_filename(audio_file_1.filename)
        # Save audio in audio_storage, path: audio_storage/filename_1
        audio_file_1.save(os.path.join(AUDIO_STORAGE, filename_1)) 
        

    if audio_file_2:
        filename_2 = secure_filename(audio_file_2.filename)
        # Save audio in audio_storage, path: audio_storage/filename_2
        audio_file_2.save(os.path.join(AUDIO_STORAGE, filename_2)) 

    #### code start
    label = predict(os.path.join(AUDIO_STORAGE, filename_1), os.path.join(AUDIO_STORAGE, filename_2))
    #### code end

    return jsonify(dict(label=int(label)))

if __name__ == '__main__':    
    app.run(host='0.0.0.0', port='6677', debug=True, use_reloader=False)
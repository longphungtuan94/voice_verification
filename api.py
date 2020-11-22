from flask import Flask
from werkzeug.utils import secure_filename

from encoder import inference as voice_clone_encoder
from classifier_preprocess_vgg import load_model_vgg

import json
import os
AUDIO_STORAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)),"audio_storage")

if not os.path.isdir(AUDIO_STORAGE):  
    os.makedirs(AUDIO_STORAGE)

app = Flask(__name__)

def load_model():
    voice_clone_encoder.load_model(Path(f'encoder/saved_models/pretrained.pt'))
    vgg_model = load_model_vgg('encoder/saved_models/')
    return voice_clone_encoder, vgg_model

voice_clone_encoder, vgg_model = load_model()

@app.route("/api/predict", methods=['POST'])
def api_predict():
    audio_file_1 = request.files[​'audio_1'​] ​# Required
    audio_file_2 = request.files[​'audio_2'​] ​# Required
    if​ audio_file_1:
        filename_1 = secure_filename(audio_file_1.filename)
        ​# Save audio in audio_storage, path: audio_storage/filename_1
        audio_file_1.save(os.path.join(AUDIO_STORAGE, filename_1)) 
        

    if​ audio_file_2:
        filename_2 = secure_filename(audio_file_2.filename)
        ​# Save audio in audio_storage, path: audio_storage/filename_2
        audio_file_2.save(os.path.join(AUDIO_STORAGE, filename_2)) 

    #### code start

    #### code end

    label = ​None ​​# Must be 0 or 1​
    return​ jsonify(label=label)

if​ __name__ == ​'__main__'​:    
    app.run(host=​'0.0.0.0'​, port=​'6677'​, debug=​True, use_reloader=False​)
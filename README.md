# Zalo Voice Verification Challenge

## Encoder:
The Encoder is a module of Corentin Jemine's awesome [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) repository. Please visit his repository for more in depth information.

### Preprocess data
You will need one of the following datasets (ideally all):
- [LibriSpeech](http://www.openslr.org/12/): train-other-500 (extract as `LibriSpeech/train-other-500`)
- [VoxCeleb1](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html): Dev A - D as well as the metadata file (extract as `VoxCeleb1/wav` and `VoxCeleb1/vox1_meta.csv`)
- [VoxCeleb2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html): Dev A - H (extract as `VoxCeleb2/dev`)
- For the zalo voice verification challenge: Zalo dataset (extract as `zalo/dataset`)

Create a `datasets_root` directory and extract the above datasets.

Run the following scripts to preprocess the datasets:

- `python encoder_preprocess.py datasets_root -d librispeech_other` 
- `python encoder_preprocess.py datasets_root -d voxceleb1`
- `python encoder_preprocess.py datasets_root -d voxceleb2`
- `python encoder_preprocess.py datasets_root -d zalo`

### Train encoder
- `python encoder_train.py checkpoint_name datasets_root/SV2TTS/encoder/ --no_visdom`

If you want to use visdom:
- Start visdom on a terminal with command `visdom` then run:
- `python encoder_train.py checkpoint_name datasets_root/SV2TTS/encoder/`

Checkpoint is saved in `encoder/saved_models`

## Classifier
### Preprocess data
Run the following scripts to preprocess the datasets using pretrained model:

- `python classifier_preprocess_vgg.py `
- `python classifier_preprocess_voice_clone.py `

These commands will create 2 pickle file containing preprocessed data to train classifier model faster, default is `train_data_vgg.pickle` and `train_data_voice_clone.pickle`.

### Train classifier
After preprocessing data we need to train the classifier:

> `python classifier_train.py `

This will train and create the best checkpoint at `weights/classifier.h5`.

## Testing
You can run a sample by first creating the predict API with:

> `python api.py`

Then edit the path in `api_test.py` and run it to test the model:

> `python api_test.py`

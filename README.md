# Zalo Voice Verification Challenge

## Encoder:
The Encoder is a module of Corentin Jemine's awesome [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) repository. Please visit his repository for more in depth information

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

### Train
- `python encoder_train.py checkpoint_name datasets_root/SV2TTS/encoder/ --no_visdom`

If you want to use visom, please start the visdom server first then run:
- `python encoder_train.py checkpoint_name datasets_root/SV2TTS/encoder/`

Checkpoint is saved in `encoder/saved_models`

### Inference

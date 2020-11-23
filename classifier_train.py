import keras
import pickle
import argparse
import numpy as np

from glob import glob
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path

from sklearn.model_selection import train_test_split

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

from model.network import build_model
from model.generator import Generator
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build and train voice verification model with preprocessed data",
    )
    parser.add_argument("-d", "--train_dir", type=Path, default="datasets_root/zalo/dataset/", help=\
        "Path to the directory containing your zalo training dataset (in which contains folders with label like '272-M-26', '632-M-27',...")
    parser.add_argument("-dvc", "--data_voice_clone", type=str, default="train_data_voice_clone.pickle", help=\
        "Path to data preprocessed and pickled by Real-Time-Voice-Cloning model")
    parser.add_argument("-dgg", "--data_vgg", type=str, default="train_data_vgg.pickle", help=\
        "Path to data preprocessed and pickled by VGG model")
    parser.add_argument("-o", "--output", type=str, default="weights/classifier.h5", help=\
        "Path to save the final model's weight after training")
    parser.add_argument("-e", "--epochs", type=int, default=1000, help=\
        "Total number of epochs to train")
    parser.add_argument("-es", "--early_stopping", type=int, default=100, help=\
        "Number of epoch for early stopping")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help=\
        "Batch size")
    args = parser.parse_args()

    # load preprocessed data
    with open(args.data_vgg, 'rb') as handle:
        data_vgg = pickle.load(handle)
    with open(args.data_voice_clone, 'rb') as handle:
        data = pickle.load(handle)

    # create training data
    label_list = defaultdict(list)
    for path in tqdm(Path(args.train_dir).rglob('*.wav')):
        file = str(path)
        path = file.split('/')
        label = path[-2]
        label_list[label].append(file)
    X = []
    y = []
    for k, v in label_list.items():
        for label in v:
            if label not in data or label not in data_vgg:
                print(label)
                continue
            x = np.concatenate((data_vgg[label], data[label]))
            X.append(x)
            y.append(k)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=69, 
                                                        shuffle=True)
    train_generator = Generator(X_train, y_train)
    val_generator = Generator(X_test, y_test)

    # build and train classification model
    model = build_model()
    print(model.summary())
    model.compile(optimizer=Adam(1e-4), 
        loss='categorical_crossentropy', metrics=['accuracy'])
    callbacks_list = [ModelCheckpoint(args.output, monitor='val_loss', verbose=1, save_weights_only=False, save_best_only=True, mode='min'),
                    EarlyStopping(patience=args.early_stopping, monitor="val_loss")]
    model.fit_generator(
        train_generator.generate_data(args.batch_size), 
        validation_data=val_generator.generate_data(args.batch_size), 
        epochs=args.epochs,
        callbacks=callbacks_list,
        verbose=1,
        validation_steps=int(len(y_test)/(args.batch_size)),
        steps_per_epoch=int(len(y_train)/(args.batch_size)))
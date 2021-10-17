import numpy as np
from pathlib import Path
import random
import os
import librosa
import pickle

import librosa.feature as libf
from .dataset import dataset_initializer


class Preprocessor:
    def __init__(self, sr=22050, frame_size=100, frame_shift=20, n_mfcc=40, n_chroma=40,
                 n_fft=2048, hop_length=512, roll_perc=0.85, batch_size=32, split=0.2):
        self.sr = sr
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.roll_perc = roll_perc
        self.batch_size = batch_size
        self.split = split
        self.seed = 211

        self.n_features = self.n_mfcc + self.n_chroma + 7

        self.default_dataset_name = f"dataset_{self.sr}_{self.frame_size}_{self.frame_shift}_{self.n_mfcc}_" \
                                    f"{self.n_chroma}_{self.n_fft}_{self.hop_length}_{self.roll_perc}"

    def _extract_effects_features(self, effects_signal):
        padded = np.pad(effects_signal, self.n_fft // 2, mode="reflect")
        framed = librosa.util.frame(padded, self.n_fft, self.hop_length)

        frame_max = framed.max(axis=0)[:, None]
        frame_var = framed.var(axis=0)[:, None]
        return np.concatenate((frame_max, frame_var), axis=1)

    def extract_data(self, signal, sr):
        args = {"sr": sr, "n_fft": self.n_fft, "hop_length": self.hop_length}

        mfcc = np.abs(libf.mfcc(signal, n_mfcc=self.n_mfcc, **args)).T
        zcrs = libf.zero_crossing_rate(signal, frame_length=self.n_fft, hop_length=self.hop_length).T
        spectral_centr = libf.spectral_centroid(signal, **args).T
        rolloff = libf.spectral_rolloff(signal, roll_percent=self.roll_perc, **args).T
        chroma = libf.chroma_stft(signal, **args, n_chroma=self.n_chroma).T

        harm, perc = librosa.effects.hpss(signal)
        harm_features = self._extract_effects_features(harm)
        perc_features = self._extract_effects_features(perc)

        features = np.concatenate((mfcc, zcrs, spectral_centr, rolloff,
                                   chroma, harm_features, perc_features), axis=1)

        return features

    def preprocess_file(self, song_path):
        signal, sr = librosa.load(song_path, sr=self.sr)
        signal, _ = librosa.effects.trim(signal)

        raw_data = self.extract_data(signal, sr)
        framed_song = librosa.util.frame(raw_data, self.frame_size, self.frame_shift, axis=0)
        return framed_song

    def add_songs_to_dataset(self, files, subdir, genre, save_root, dataset_name, ):
        rng = np.random.default_rng(self.seed)
        Path(os.path.join(save_root, dataset_name)).mkdir(parents=True, exist_ok=True)
        save_filename = os.path.join(save_root, dataset_name, f"{genre}.npy")

        table = None
        for f in files:
            filename = os.path.join(subdir, f)
            song_data = self.preprocess_file(filename)
            if table is None:
                table = song_data
            else:
                table = np.concatenate((table, song_data), axis=0)

        rng.shuffle(table)
        np.save(save_filename, table)

    def create_dataset(self, data_root, save_root):

        for subdir, dirs, files in os.walk(data_root):
            if len(dirs) == 0:
                genre = os.path.split(subdir)[1]

                random.shuffle(files)
                n = len(files)
                test_files = files[:int(n * self.split)]
                val_files = files[int(n * self.split):2 * int(n * self.split)]
                train_files = files[2 * int(n * self.split):]

                self.add_songs_to_dataset(test_files, subdir, genre, save_root, "test")
                self.add_songs_to_dataset(val_files, subdir, genre, save_root, "val")
                self.add_songs_to_dataset(train_files, subdir, genre, save_root, "train")

            print(f"Done with {subdir}")

    def get_dataset(self, root_dir):

        train = dataset_initializer(root_dir, "train", self.batch_size, self.seed, (self.frame_size, self.n_features))
        val = dataset_initializer(root_dir, "val", self.batch_size, self.seed, (self.frame_size, self.n_features))
        test = dataset_initializer(root_dir, "test", self.batch_size, self.seed, (self.frame_size, self.n_features))

        return train, val, test

    def save_parameters(self, folder=os.path.join("Data", "stored")):
        kwargs = {"sr": self.sr, "frame_size": self.frame_size, "frame_shift": self.frame_shift,
                  "n_mfcc": self.n_mfcc, "n_chroma": self.n_chroma, "n_fft": self.n_fft,
                  "hop_length": self.hop_length, "roll_perc": self.roll_perc, "batch_size": self.batch_size,
                  "split": self.split}

        path = os.path.join(folder, f"preprocessor.pickle")

        with open(path, "wb") as f:
            pickle.dump(kwargs, f)

    @staticmethod
    def load_parameters(path):
        with open(path, "rb") as f:
            kwargs = pickle.load(f)

        return Preprocessor(**kwargs)


def start_preprocessing(data_folder, save_folder, force, sample_rate, frame_size, frame_shift, n_mfcc,
                        n_chroma, n_fft, hop_length, roll_perc, batch_size, split):
    pr = Preprocessor(sr=sample_rate, frame_size=frame_size,
                      frame_shift=frame_shift, n_mfcc=n_mfcc,
                      n_chroma=n_chroma, n_fft=n_fft,
                      hop_length=hop_length, roll_perc=roll_perc,
                      batch_size=batch_size, split=split)

    try:
        if force:
            # Force re-preprocessing
            raise FileNotFoundError

        pr.get_dataset(save_folder)
    except FileNotFoundError:

        pr.create_dataset(data_folder, save_folder)

    pr.save_parameters(save_folder)

import os
import keras
import time

from .preprocess import Preprocessor


class Trainer:
    def __init__(self, preprocessor: Preprocessor, n_neurons=200, dropout=0.1, recurrent_dropout=0.1):
        self.pr = preprocessor
        self.n_neurons = n_neurons
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout

        self.logdir = os.path.join(os.curdir, "logs")

    def _get_run_logdir(self):
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(self.logdir, run_id)

    def train(self, pr_dir):
        train, val, test = self.pr.get_dataset(pr_dir)

        norm_layer = keras.layers.Normalization()
        norm_layer.adapt(train.map(lambda x, y: x))

        model = keras.models.Sequential([
            norm_layer,
            keras.layers.LSTM(self.n_neurons, return_sequences=True, input_shape=[None, self.pr.n_features],
                              dropout=self.dropout, recurrent_dropout=self.recurrent_dropout),
            keras.layers.LSTM(self.n_neurons, dropout=self.dropout, recurrent_dropout=self.recurrent_dropout),
            keras.layers.Dense(10, activation="softmax")
        ])

        tensorboard_cb = keras.callbacks.TensorBoard(self._get_run_logdir())
        earlystopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min",
                                                         patience=3,
                                                         restore_best_weights=True)

        callbacks = [tensorboard_cb, earlystopping_cb]

        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
        model.fit(train, epochs=50, validation_data=val, callbacks=callbacks)

        model.evaluate(test)

        return model


def start_training(model_path, preprocessor_root, n_neurons, dropout, recurrent_dropout):
    pr = Preprocessor.load_parameters(os.path.join(preprocessor_root, f"preprocessor.pickle"))
    tr = Trainer(pr, n_neurons, dropout, recurrent_dropout)

    model = tr.train(preprocessor_root)
    model.save(model_path)

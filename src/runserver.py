from flask import Flask
from flask_restful import Api, Resource, abort
from secrets import token_hex
import tensorflow as tf
import numpy as np
import keras
import requests
import os
import sys
import pytube as pt
import tempfile
from .preprocess import Preprocessor
from .dataset import N_CLASS, CLASS_TO_LABEL

MAX_VIDEO_LENGTH = 600

app = Flask(__name__)
api = Api(app)


def get_yt(youtube_id):
    full_url = f"https://www.youtube.com/watch?v={youtube_id}"
    r = requests.get(f"https://www.youtube.com/oembed?url={full_url}")
    if r.status_code == 400:
        abort(400, message=f"URL {full_url} is invalid")

    yt = pt.YouTube(full_url)
    if yt.length >= MAX_VIDEO_LENGTH:
        abort(400, message=f"Video is longer than 10 minutes")

    return yt


class Model(Resource):
    def __init__(self, model, preprocessor, debug):
        self.model = model
        self.pr = preprocessor
        self.debug = debug

    def predict(self, path):
        song_data = self.pr.preprocess_file(path)

        predictions = self.model.predict(song_data)
        best_class = tf.math.argmax(predictions, axis=1)
        votes_for_each_class = [tf.reduce_sum(tf.cast(best_class == i, tf.int32)).numpy()
                                for i in range(N_CLASS)]

        winner = CLASS_TO_LABEL[np.argmax(votes_for_each_class)]
        prob_distr = votes_for_each_class/np.sum(votes_for_each_class)

        label_distr = {
            CLASS_TO_LABEL[i]: f"{p*100:.2f}%" for i, p in enumerate(prob_distr)
        }

        return winner, label_distr

    def get(self, youtube_id):
        yt = get_yt(youtube_id).streams.filter(only_audio=True).order_by("abr").last()
        filename = f"{token_hex(8)}.mp4"
        with tempfile.TemporaryDirectory() as root:
            yt.download(root, filename=filename)

            path_to_song = os.path.join(root, filename)

            if self.debug:
                print(path_to_song, file=sys.stderr)

            label, label_distr = self.predict(path_to_song)
        return {"prediction": label, "distribution": label_distr}


def start_server(model_path, preprocess_path, debug):
    model = keras.models.load_model(model_path)
    preprocessor = Preprocessor.load_parameters(preprocess_path)

    api.add_resource(Model, "/<string:youtube_id>", resource_class_kwargs={
        "model": model,
        "preprocessor": preprocessor,
        "debug": debug
    })

    app.run(debug=True, port=7070)

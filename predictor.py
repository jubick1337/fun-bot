import keras.backend as K
import numpy as np
import tensorflow as tf
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


class Predictor:
    def _create_model(self):
        model = load_model('model.h5')
        return model

    def _encode_text(tokenizer, lines, length):
        encoded = tokenizer.texts_to_sequences(lines)
        padded = pad_sequences(encoded, maxlen=length, padding='post')
        return padded

    def _create_tokenizer(lines):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer

    def predict(self, text):
        encoded_text = self._encode_text(self.tokenizer, [text], 72)
        p = self.model.predict(encoded_text)
        return np.argmax(p)

    def __init__(self):
        self.sess = tf.Session()
        self.graph = tf.get_default_graph()
        self.model = self._create_model()
        with open('tokenzier', 'rb') as f:
            self.tokenizer = pickle.load(f)
        self.model = self._create_model()

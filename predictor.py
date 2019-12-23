import keras.backend as K
import numpy as np
import tensorflow as tf
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


class Predictor:
    def _recall_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def _precision_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def _f1_m(self, y_true, y_pred):
        precision = self._precision_m(y_true, y_pred)
        recall = self._recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def _create_model(self):
        model = load_model('model.h5', custom_objects=
        {'f1_m': self._f1_m})
        return model

    def _encode_text(self, tokenizer, lines, length):
        encoded = tokenizer.texts_to_sequences(lines)
        padded = pad_sequences(encoded, maxlen=length, padding='post')
        return padded

    def predict(self, text):
        encoded_text = self._encode_text(self.tokenizer, [text], 72)
        p = self.model.predict(encoded_text)
        return p

    def __init__(self):
        with open('tokenzier', 'rb') as f:
            self.tokenizer = pickle.load(f)
        self.model = self._create_model()

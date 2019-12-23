from tensorflow.python.keras.backend import set_session
import keras.backend as K
import tensorflow as tf
import pickle
from keras.models import Sequential
from keras.layers import *
from keras.preprocessing.sequence import pad_sequences

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

    def _create_model(self, length, vocab_size):
        model = Sequential()
        model.add(Embedding(vocab_size, 50, input_shape=(length,)))
        model.add(Conv1D(filters=16, kernel_size=2, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(filters=32, kernel_size=2, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy', self._f1_m])
        return model

    def _encode_text(self, tokenizer, lines, length):
        encoded = tokenizer.texts_to_sequences(lines)
        padded = pad_sequences(encoded, maxlen=length, padding='post')
        return padded

    def predict(self, text):
        K.clear_session()
        global sess
        global graph
        encoded_text = self._encode_text(self.tokenizer, [text], 72)
        with self.graph.as_default():
            set_session(self.sess)
            p = self.model.predict(encoded_text)
            return p

    def __init__(self):
        with open('tokenizer', 'rb') as f:
            self.tokenizer = pickle.load(f)
        self.sess = tf.Session()
        self.graph = tf.get_default_graph()
        set_session(self.sess)
        self.model = self._create_model(72, 107695)
        self.model.load_weights('model_weights.h5')

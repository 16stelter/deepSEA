import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation


class SimpleMlp:
    def __init__(self):
        self.model = Sequential([
            Dense(32, input_shape=16, activation='softsign'),
            Dense(32, activation='softsign'),
            Dense(32, activation='softsign'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, x, y):
        self.model.fit(x, y, epochs=10,
                       batch_size=100,
                       validation_split=0.2)

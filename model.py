import tensorflow as tf 

from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional, Flatten, Input, Reshape, RepeatVector, TimeDistributed, GRU


class NDAEModel:
    def __init__(self):
        self.embedding_dim = 41

    def build_model(self):
        input_layer = Input(shape = (1, self.embedding_dim))

        encode1_lstm1 = GRU(units = 64, 
                    activation = 'relu',
                    kernel_regularizer= regularizers.l2(0.00),
                    return_sequences=True)(input_layer)

        encode1_lstm2 = GRU(units = 32, 
                    activation = 'relu',
                    return_sequences=False)(encode1_lstm1)

        repeat = RepeatVector(1)(encode1_lstm2)

        encode2_lstm1 = GRU(units = 64, 
                    activation = 'relu',
                    return_sequences=True)(repeat)

        encode2_lstm2 = GRU(units = 32,
                    activation = 'relu',
                    return_sequences=False)(encode2_lstm1)

        output = Dense(units = 23, activation='softmax')(encode2_lstm2)

        model = Model(inputs=input_layer, outputs = output)

        model.summary()

        model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

        return model
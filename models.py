from keras.layers import Dense, Input, LSTM, Embedding, Dropout, GlobalMaxPooling1D, Bidirectional, Conv1D, concatenate
from keras.models import Model


def YoonKim(sentence_length, vocab_size, embedding_dim, embedding_matrix, n_filters, trainable):
    """
    TODO: adapter a la parametrisation des filtres (taille 7 a rendre possible pour rapport final)
    """
    # input
    inp = Input(shape=(sentence_length,))
    # embedding
    emb = Embedding(vocab_size, embedding_dim, input_length=sentence_length,
                    weights=[embedding_matrix], trainable=trainable)(inp)

    # Specify each convolution layer and their kernel siz i.e. n-grams
    conv_3 = Conv1D(filters=n_filters, kernel_size=3, activation='relu')(emb)
    pool_3 = GlobalMaxPooling1D()(conv_3)

    conv_4 = Conv1D(filters=n_filters, kernel_size=4, activation='relu')(emb)
    pool_4 = GlobalMaxPooling1D()(conv_4)

    conv_5 = Conv1D(filters=n_filters, kernel_size=5, activation='relu')(emb)
    pool_5 = GlobalMaxPooling1D()(conv_5)

    # Gather all convolution layers
    x = concatenate([pool_3, pool_4, pool_5], axis=1)
    x = Dropout(0.1)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.1)(x)
    outp = Dense(6, activation='sigmoid')(x)

    # # load pre-trained model from disk
    # model = load_nnet(MODEL_NAME)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # ready for .fit() !
    return (model)


def Bidirectional_LSTM(sentence_length, vocab_size, embedding_dim, embedding_matrix, trainable):
    # input
    inp = Input(shape=(sentence_length,))
    # embedding
    x = Embedding(vocab_size, embedding_dim, input_length=sentence_length,
                  weights=[embedding_matrix], trainable=trainable)(inp)
    # LSTM
    x = Bidirectional(LSTM(60, return_sequences=True, name='lstm_layer'))(x)
    # max pooling 1D
    x = GlobalMaxPooling1D()(x)
    # dropout 1
    x = Dropout(0.1)(x)
    # dense 1
    x = Dense(50, activation="relu")(x)
    # dropout 2
    x = Dropout(0.1)(x)
    # dense 1
    outp = Dense(6, activation="sigmoid")(x)

    # # load pre-trained model from disk
    # model = load_nnet(MODEL_NAME)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # ready to .fit() !
    return (model)

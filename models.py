from keras.layers import Dense, Input, LSTM, Embedding, Dropout, GlobalMaxPooling1D, Bidirectional, Conv1D, concatenate
from keras.models import Model


# TODO : add spatial dropout and/or batch norm in yoon_kim()
# TODO : stack LSTM layers in bidirectional_lstm()


def yoon_kim(sentence_length=200, vocab_size=30000,
             n_filters=100, filters_sizes=(3, 5, 7),
             embedding_dim=150, embedding_matrix=None, train_embeddings=True):
    # input
    inp = Input(shape=(sentence_length,))
    # embedding
    emb = Embedding(vocab_size, embedding_dim, input_length=sentence_length, trainable=train_embeddings,
                    weights=[embedding_matrix] if embedding_matrix is not None else None)(inp)

    # Specify each convolution layer and their kernel size i.e. n-grams
    conv_layers, pool_layers = [None] * len(filters_sizes), [None] * len(filters_sizes)
    for i_layer, filter_size in enumerate(filters_sizes):
        conv_layers[i_layer] = Conv1D(filters=n_filters, kernel_size=filter_size, activation='relu')(emb)
        pool_layers[i_layer] = GlobalMaxPooling1D()(conv_layers[i_layer])

    # Gather all convolution layers
    x = concatenate([pool for pool in pool_layers], axis=1)
    # x = Dropout(0.1)(x)
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


def bidirectional_lstm(sentence_length=200, vocab_size=30000,
                       embedding_dim=150, embedding_matrix=None, train_embeddings=True):
    # input
    inp = Input(shape=(sentence_length,))
    # embedding
    emb = Embedding(vocab_size, embedding_dim, input_length=sentence_length, trainable=train_embeddings,
                    weights=[embedding_matrix] if embedding_matrix is not None else None)(inp)
    # LSTM
    x = Bidirectional(LSTM(60, return_sequences=True, name='lstm_layer'))(emb)
    # max pooling 1D
    x = GlobalMaxPooling1D()(x)
    # dropout 1
    # x = Dropout(0.1)(x)
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

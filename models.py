from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GlobalMaxPooling1D, Bidirectional, Conv1D, concatenate
from keras.models import Model

def YoonKim(sentence_length,emb_dim,emb_matrix,n_filters,trainableBool):
    """
    TODO: adapter à la paramétrisation des filtres (taille 7 à rendre possible pour rapport final)
    """

    SENTENCE_LENGTH = sentence_length
    EMBEDDING_DIM = emb_dim
    embedding_matrix = emb_matrix
    N_FILTERS = n_filters

    # input
    inp = Input(shape=(SENTENCE_LENGTH, ))
    # embedding
    emb = Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=SENTENCE_LENGTH,
                    weights=[embedding_matrix], trainable=trainableBool)(inp)

    # Specify each convolution layer and their kernel siz i.e. n-grams
    conv_3 = Conv1D(filters=N_FILTERS, kernel_size=3, activation='relu')(emb)
    pool_3 = GlobalMaxPooling1D()(conv_3)

    conv_4 = Conv1D(filters=N_FILTERS, kernel_size=4, activation='relu')(emb)
    pool_4 = GlobalMaxPooling1D()(conv_4)

    conv_5 = Conv1D(filters=N_FILTERS, kernel_size=5, activation='relu')(emb)
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

    # modèle prêt au .fit() !
    return(model)

def Bidirectional_LSTM(sentence_length,emb_dim,emb_matrix,trainableBool):

    SENTENCE_LENGTH = sentence_length
    EMBEDDING_DIM = emb_dim
    embedding_matrix = emb_matrix

    # input
    inp = Input(shape=(SENTENCE_LENGTH, ))
    # embedding
    x = Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=SENTENCE_LENGTH,
                    weights=[embedding_matrix], trainable=trainableBool)(inp)
    # LSTM
    x = Bidirectional(LSTM(60, return_sequences=True,name='lstm_layer'))(x)
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

    # modèle prêt au .fit() !
    return(model)

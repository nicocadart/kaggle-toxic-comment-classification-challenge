# -*- coding: utf-8 -*-

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import nltk
from sklearn.model_selection import train_test_split

from tools import *
from models import *
from embeddings import *

from bert_serving.client import BertClient

# Les paramètres doivent correspondre à ce qui a été précisé au lancement du serveur de la librairie
# (attention au path du modèle BERT pré-entraîné !)
# Par exemple, dans un autre screen: "bert-serving-start -pooling_strategy REDUCE_MEAN -model_dir ./bert_as_service/uncased_L-12_H-768_A-12/ -max_seq_len 40 -num_worker=1"

MAX_SEQ_LENGTH = 40
MODEL_NAME = "BertSentence40_LSTM_clean_singleDense"
NUM_CLASSES = 6
N_EPOCHS = 2

SPLIT_VALID_RATIO = 0.10
SPLIT_RANDOM_SEED = 0

CONT_EMBEDDING_DIM = 768
INPUT_SHAPE = (CONT_EMBEDDING_DIM,)

# load raw string data
data_train, y_train_all, data_test, id_test\
    = load_data()

# NETTOYAGE DES DONNÉES
params = {'clean': True,
          'lower': False,
          'lemma': False,
          'stop_words': True}

CommentCleaner(**params)

data_train = transform_dataset(data_train, transformer=CommentCleaner, kwargs=params)
data_test = transform_dataset(data_test, transformer=CommentCleaner, kwargs=params)

# PADDING DES DONNÉES
paramsPadder = {'maxlen': MAX_SEQ_LENGTH+1,
                'padval': "<pad>"}

data_train = transform_dataset(data_train, transformer=CommentPadder, kwargs=paramsPadder)
data_test = transform_dataset(data_test, transformer=CommentPadder, kwargs=paramsPadder)


print("Beginning train-valid split")
X_train, X_valid, y_train, y_valid = train_test_split(data_train, y_train_all,
                                                      test_size=SPLIT_VALID_RATIO,
                                                      random_state=SPLIT_RANDOM_SEED)
print("End train-valid split")

#######################################
## SHORTEN DATA SETS FOR QUICK TRIAL ##
#######################################

# X_train = X_train[0:50000]
# y_train = y_train[0:50000]
# X_valid = X_valid[0:5000]
# y_valid = y_valid[0:5000]

######################################
## GET BERT EMBEDDINGS FOR COMMENTS ##
######################################

# https://github.com/hanxiao/bert-as-service
bc = BertClient(check_length=False)

print("Beginning valid BERT encoding")
#(Nbr valid sentences, emb size)
all_valid_embeddings = bc.encode(X_valid)
print(all_valid_embeddings.shape)
print("End valid BERT encoding")

print("Beginning train BERT encoding")
#(Nbr train sentences, emb size)
all_train_embeddings = bc.encode(X_train)
print(all_train_embeddings.shape)
print("End train BERT encoding")

print("Beginning test BERT encoding")
#(Nbr test sentences, emb size)
all_test_embeddings = bc.encode(data_test)
print(all_test_embeddings.shape)
print("End test BERT encoding")

######################
## RESHAPING INPUTS ##
######################

nbr_train_samples = all_train_embeddings.shape[0]
all_train_embeddings = all_train_embeddings.reshape(nbr_train_samples,
                                                    CONT_EMBEDDING_DIM,
                                                    1)
print("final train emb shape for input:",all_train_embeddings.shape)
nbr_valid_samples = all_valid_embeddings.shape[0]
all_valid_embeddings = all_valid_embeddings.reshape(nbr_valid_samples,
                                                    CONT_EMBEDDING_DIM,
                                                    1)
print("final valid emb shape for input:",all_valid_embeddings.shape)
nbr_test_samples = all_test_embeddings.shape[0]
all_test_embeddings = all_test_embeddings.reshape(nbr_test_samples,
                                                    CONT_EMBEDDING_DIM,
                                                    1)
print("final test emb shape for input:",all_test_embeddings.shape)

###########
## MODEL ##
###########

# main input (comments)
inp = Input(shape=(CONT_EMBEDDING_DIM,1))
x = SpatialDropout1D(0.1)(inp)
x = Bidirectional(LSTM(60, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = Conv1D(60, kernel_size=3, padding="valid")(x)
# pooling
max_pool = GlobalMaxPooling1D()(x)
avg_pool = GlobalAveragePooling1D()(x)
x = concatenate([max_pool, avg_pool])
# dense 1
# x = Dense(50, activation="relu")(x)
# dropout 1
# x = Dropout(0.1)(x)
# final dense
outp = Dense(NUM_CLASSES, activation="sigmoid")(x)

# build final model
model = Model(inp, outputs=outp)

model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

RocAuc = RocAucEvaluation(validation_data=(all_valid_embeddings, y_valid))

hist = model.fit(all_train_embeddings, y_train,
                 epochs=N_EPOCHS,
                 validation_data=(all_valid_embeddings, y_valid),
                 callbacks=[RocAuc])

# final model evaluation
y_train_pred = model.predict(all_train_embeddings, batch_size=64)
train_score = evaluate(y_train, y_train_pred)
print("ROC-AUC score on train set : {:.4f}".format(train_score))

y_valid_pred = model.predict(all_valid_embeddings, batch_size=64)
valid_score = evaluate(y_valid, y_valid_pred)
print("ROC-AUC score on validation set : {:.4f}".format(valid_score))

# predict
y_test_pred = model.predict(all_test_embeddings, batch_size=64, verbose=2)

# write submission file
submission(y_test_pred, id_test, name=MODEL_NAME)

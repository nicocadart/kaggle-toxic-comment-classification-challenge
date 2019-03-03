import numpy as np
np.random.seed(42)
from keras.preprocessing import text, sequence

from tools import *
from embeddings import *
from models import *

import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
from keras import backend as K
import keras.layers as layers
from keras.models import Model, load_model
from keras.engine import Layer
import numpy as np

# Initialize session
sess = tf.Session()
K.set_session(sess)

###################
## LOAD RAW DATA ##
###################

# load raw string data
data_train, y_train_all, data_test, id_test = load_data()

# Create datasets (Only take up to 150 words for memory)
train_text = [' '.join(t.split()[0:50]) for t in data_train]
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
train_label = y_train_all

test_text = [' '.join(t.split()[0:50]) for t in data_test]
test_text = np.array(test_text, dtype=object)[:, np.newaxis]

######################################################################################
## FONCTION PERMETTANT LE CHARGEMENT DES EMBEDDINGS À PARTIR DU MODÈLE PRE ENTRAINE ##
######################################################################################

def ELMoEmbedding(x):
    # on va chercher le modèle sur tfhub.dev, on y trouve également des explications et exemples
    elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
    # "default": fixed mean-pooling of all contextualized word representations with shape [batch_size, 1024]
    return elmo_model(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

###########
## MODEL ##
###########

input_text = layers.Input(shape=(1,), dtype=tf.string)
embed_seq = layers.Lambda(ELMoEmbedding, output_shape=(1024,))(input_text)
x = Dense(256,activation ="relu")(embed_seq)
preds = Dense(6,activation="sigmoid")(x)

model = Model(input_text,preds)
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

model.fit(train_text,train_label,epochs=2,batch_size=64)

# predict
y_test_pred = model.predict(test_text, batch_size=64)

# write submission file
MODEL_NAME = "elmo_sentence50_elmo2_dense256"
submission(y_test_pred, id_test, name=MODEL_NAME)

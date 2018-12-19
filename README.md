# kaggle-toxic-comment-classification-challenge


Kaggle Challenge : Toxic Comment Classification, for course OPT7 of master AIC at Université Paris-Sud.

Autors : M. Bauw, N. Cadart, B. Sarthou

Date : February 2019

## Premières idées en vrac

- TF IDF va être utile car des termes grossiers reviennent tout le temps et ça vire les mots inutiles
- 3 niveaux de tâches: preprocess, quantification/description, classification
- Première classification binaire clean/pas clean ie est-ce qu'une classe au moins est à 1, puis multiclasse pour savoir le(s)quel(s) sont à 1
- **Première difficulté:** classification multiclasse non exclusive

https://nlp.stanford.edu/IR-book/html/htmledition/classification-with-more-than-two-classes-1.html

On pourrait donc partir sur 6 classifieurs binaires indépendants - AU FINAL NON (?)

- **Deuxième difficulté:** déséquilibre des classes, utiliser une loss adaptée augmentant comme il faut le poids de la classe sous représentée
- **Troisième difficulté:** pauvreté

### Première solution

LSTM + max pooling + denses avec du dropout sur chaque dense, code dispo avec Keras

https://www.kaggle.com/sbongo/for-beginners-tackling-toxic-using-keras

Commencer par ça puis passer sous PyTorch ?

### TODO

- lire les kernels pour se retrouver dans le paysage des solutions et prétraitements
- implémenter le baseline (kernel Kaggle - cf. url ci-dessus LSTM, pooling etc)
- implémenter le modèle d'Allauzen
- implémenter le modèle GRU (kernel Kaggle bidirectional GRU)
- implémenter une baseline sans embeddings/qui ne soit pas du deep learning


## Plus d'idées

- Utiliser des embeddings déjà entraînés. Voir [GloVe](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)


## Embedding + LSTM + 2 FC

Voir [notebook](kernels/embeddings_lstm_dense.ipynb).
- Accuracy de 98.25% sur validation set (98.31 sur train)
- 2 epochs seulement, mais stagne au delà du premier tiers de la deuxième epoch
- split train/validation ratio de 0.1
- dropout (p=0.1) pour chaque couche dense
- dimension des embeddings de 128
- longueur max/padding : 200

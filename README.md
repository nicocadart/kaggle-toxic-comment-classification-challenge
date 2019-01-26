# kaggle-toxic-comment-classification-challenge


Kaggle Challenge : Toxic Comment Classification, for course OPT7 of master AIC at Université Paris-Sud.

Autors : M. Bauw, N. Cadart, B. Sarthou

Date : February 2019


## Done

- lire les kernels pour se retrouver dans le paysage des solutions et prétraitements
- implémenter le baseline (kernel Kaggle - cf. url ci-dessus LSTM, pooling etc)
- implémenter le modèle de Yoon Kim
- essayer les embeddings pré-entrainés (Glove, word2vec)


### TODO/idées

- tester les GRU plutôt que LSTM
- implémenter une baseline sans embeddings/qui ne soit pas du deep learning
- stacker des LSTM/convolutions
- tester les embeddings contextuels
- tester des entrées auxiliaires (nb de majuscules, ponctuation, longueur des commentaires, ...)


## A propos des embeddings pré-entrainés

Le but est d'utiliser des embeddings pré-entrainés sur différents corpus, notemment:
- [GloVe entraîné sur Twitter](https://nlp.stanford.edu/projects/glove/) (dimensions 25, 50, 100, 200)
- [GloVe entraîné sur Wikipedia 2014 + Gigaword 5](https://nlp.stanford.edu/projects/glove/) (dimensions 50, 100, 200, 300)
- [word2vec entraîné sur Google News](https://code.google.com/archive/p/word2vec/) (dimension 300)
- fasttext (non encore disponible par la version actuelle du code)

Pour charger les poids pré-entraînés d'une couche Keras Embeddings, il suffit d'appeler :
```
# Load GloVe pre-trained embeddings
EMBEDDING_DIM = 200  # several embeddings sizes depending on source : 25, 50, 100, 200, 300 
EMBEDDING_SOURCE = 'glove_wikipedia'  # {'glove_twitter', 'glove_wikipedia', 'word2vec_googlenews'}

embeddings_matrix = embeddings.load_pretrained_embeddings(tokenizer.word_index, 
                                                          VOCAB_SIZE, 
                                                          EMBEDDING_DIM, 
                                                          EMBEDDING_SOURCE)
```
Puis de donner cette matrice de poids directement en paramètre de la couche Keras:
```
emb = Embedding(vocab_size, embedding_dim, input_length=sentence_length, 
                weights=[embedding_matrix])(input)
```

## Modèles de réseaux disponibles

2 réseaux sont actuellement disponibles, le modèle de Yoon Kim et un modèle de LSTM bidirectionnel. Ils peuvent être chargés facilement grâce aux fonctions `models.yoon_kim()` et `models.bidirectional_lstm()`.
# kaggle-toxic-comment-classification-challenge

Kaggle Challenge : Toxic Comment Classification, for course OPT7 of master AIC at Université Paris-Sud.

Autors : M. Bauw, N. Cadart, B. Sarthou

Date : December 2018 - February 2019

- [kaggle-toxic-comment-classification-challenge](#kaggle-toxic-comment-classification-challenge)
  - [Description et installation](#description-et-installation)
  - [A propos des embeddings pré-entrainés](#a-propos-des-embeddings-pré-entrainés)
  - [A propos des embeddings contextuels](#a-propos-des-embeddings-contextuels)
  - [Done](#done)
  - [TODO](#todo)

Ce repo rassemble la plupart du code utilisé dans le cadre du [Toxic Comment Classification Kaggle Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). Il contient les outils et notebooks qui ont permis de générer les résultats, avec quelques exemples de modèles testés. En revanche, la très grande majorité des classifieurs testés n'est pas représentée ici. En effet, ces fichiers diffèrent seulement de quelques paramètres, et ont donc été générés en local afin d'être exécutés sur d'autres machines distantes pour plus de rapidité.

## Description et installation

Utilitaires :
- `tools.py` définit de nombreuses fonctions permettant de charger, nettoyer ou convertir rapidement les données, de simplifier la gestion (chargement ou sauvegarde) des classifieurs, ou bien d'évaluer les résultats.
- `embeddings.py` définit quelques fonctions utilitaires pour charger directement une matrice de poids d'embeddings pré-entraînés à partir de plusieurs datasets disponibles.
- `models.py`définit quelques exemples de classifieurs types utilisés afin de faciliter leur définition (résaux de neurones, classifiers "standards", model mix).

Ces outils nécessitent pour fonctionner Keras et Scikit-Learn. Ils nécessitent de plus la création de certains répertoires :
- `data/` : répertoire contenant a minima les données textuelles à classifier (fichiers `train.csv` et `test.csv`, téléchargeables sur [la page de la compétition Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)). Les fichiers `.csv` contenant les prédictions seront enregistrés ici.
- `models/` : répertoire où seront sauvegardés automatiquement tous les modèles (NN) testés, afin de pouvoir les réutiliser facilement sans relancer l'apprentissage
- `embeddings/` : répertoire contenant les fichiers d'embeddings pré-entraînés qui seront chargés automatiquement par la fonction `embeddings.load_pretrained_embeddings()`. Voir [A propos des embeddings pré-entrainés](#a-propos-des-embeddings-pré-entrainés).

Notebooks :
- `Data Preprocessing for Toxic Comments.ipynb` : introduit le chargement, l'aperçu, le nettoyage et la conversion des données.
- `models_testing.ipynb` : donne des exemples d'utilisation de réseaux de neurones dans le cadre de ce challenge en utilisant les outils présentés précédemment.
- `models_not_nn_testing.ipynb` : idem que `models_testing.ipynb` mais pour des classifieurs non réseaux de neurones.
- `contextual.ipynb`
- `Models mix.ipynb` :  Permet d'entraîner une méthode d'ensemble (pondération des résultats) à partir de prédicitons d'autres modèles déjà entraînés (voir plus bas)

## A propos des embeddings pré-entrainés

Le but est d'utiliser des embeddings pré-entrainés sur différents corpus, notemment:
- [GloVe entraîné sur Twitter](https://nlp.stanford.edu/projects/glove/) (dimensions 25, 50, 100, 200)
- [GloVe entraîné sur Wikipedia 2014 + Gigaword 5](https://nlp.stanford.edu/projects/glove/) (dimensions 50, 100, 200, 300)
- [word2vec entraîné sur Google News](https://code.google.com/archive/p/word2vec/) (dimension 300)
- [fastText entraîné sur Common Crawl](https://fasttext.cc/docs/en/english-vectors.html) (dimension 300)

Pour charger les poids pré-entraînés d'une couche Keras Embeddings, il suffit d'appeler :
```
import embeddings

EMBEDDING_DIM = 200  # several embeddings sizes depending on source : 25, 50, 100, 200, 300
EMBEDDING_SOURCE = 'glove_wikipedia'  # {'glove_twitter', 'glove_wikipedia', 'word2vec_googlenews', 'fasttext_crawl'}

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
## A propos du model mix
Pour rajouter un modèle, il faut générer une prédiction sur *l'ensemble* de la base d'apprentissage (train+val, variable `y_train_all` dans la plupart des notebooks) et d'enregistrer la prédiction avec la fonction `save_pred`. Il faut également enregistrer la prédiction sur la base de test, avec le même nom de modèle que pour l'apprentissage, auquel on ajoute `_test`.
Enfin, le notebook `Model mix` permet de spécifier le nom des modèles que l'on veut "mixer", il s'occupe de les charger, d'afficher le score pour chaque modèle, le résultat initial (pondération uniforme) puis le score final, et enfin d'écrire un fichier de soumission pour kaggle avec les fichiers test. 

## A propos des embeddings contextuels

TODO

## Done

- implémenter une baseline (LSTM + pooling)
- implémenter le modèle de Yoon Kim (convolutions)
- utiliser des embeddings pré-entrainés (Glove, Word2Vec, FastText, Freebase)
- tester des entrées auxiliaires (nb de majuscules, ponctuation, longueur des commentaires, ...)
- tester les GRU plutôt que LSTM (plus léger, mais a plus de mal à exploiter le0 slongues dépendances)
- implémenter des modèles qui ne soient pas du deep learning
- stacker des LSTM
- enchainer LSTM puis convolution (exemples [ici](https://www.kaggle.com/fizzbuzz/bi-lstm-conv-layer-lb-score-0-9840), [là](https://www.kaggle.com/eashish/bidirectional-gru-with-convolution) ou encore [là](https://www.kaggle.com/tunguz/bi-gru-cnn-poolings-gpu-kernel-version))
- tester les embeddings contextuels
- augmentation synthétique des données

## TODO

- optimisation des paramètres des modèles déjà existants (taille vocabulaire, longueurs phrases, pré-traitement, dimension et initialisation des embeddings, LSTM/GRU, tailles et nombre de couches, ...)
- réduire l'overfitting (dropout spatial/récurrent/normal, plus de dropout et d'epochs, régularisation, ...)
- étudier plus en détail l'influence du prétraitement (nettoyage des données, lower(), stop_words, ...)
- essayer plus de model mix
- plus de preprocessing (exemples [ici](https://www.kaggle.com/larryfreeman/toxic-comments-code-for-alexander-s-9872-model) ou [là](https://www.kaggle.com/fizzbuzz/toxic-data-preprocessing))
- capsule net?
- approfondir l'utilisation des embeddings contextuels

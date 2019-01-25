import numpy as np
from gensim.models import KeyedVectors  # to load word2vec pretrained embeddings


def load_pretrained_embeddings(word_index, vocab_size, emb_dim=200, source='glove_twitter'):
    """
    Build embeddings layer weights from pre-trained word vectors.
    :param word_index: dictionary mapping our words to their numerical index (sorted from most to least frequent)
    :param vocab_size: number of most frequent words we want to consider
    :param emb_dim: dimension of the embeddings vectors
    :param source: the pre-trained word vectors dataset, in {'glove_twitter', 'glove_wikipedia', 'word2vec_googlenews'}
    :return: the embeddings weights to give to keras layer
    """
    # parse file name
    if source == 'glove_twitter':
        if emb_dim not in {25, 50, 100, 200}:
            raise ValueError("'glove_twitter' embeddings are only available for dimensions 25, 50, 100 or 200")
        embeddings_file = 'embeddings/glove.twitter.27B.{}d.txt'.format(emb_dim)

    elif source == 'glove_wikipedia':
        if emb_dim not in {50, 100, 200, 300}:
            raise ValueError("'glove_wikipedia' embeddings are only available for dimensions 50, 100, 200 or 300")
        embeddings_file = 'embeddings/glove.6B.{}d.txt'.format(emb_dim)

    elif source == 'word2vec_googlenews':
        if emb_dim != 300:
            raise ValueError("'word2vec_googlenews' embeddings are only available for dimension 300")
        embeddings_file = 'embeddings/GoogleNews-vectors-negative300.txt'

    else:
        raise ValueError("Unknown pre-trained embeddings source. "
                         "Must be in {'glove_twitter', 'glove_wikipedia', 'word2vec_googlenews'}")

    # parse embeddings file and update embedding matrix
    word_vectors = dict()
    with open(embeddings_file) as file:
        for i_line, line in enumerate(file):
            # display loading every 1000 words
            if i_line % 1000 == 0:
                print('Loading word vector {}...'.format(i_line + 1), end="\r")

            # read new word and vector
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype=np.float32)

            # save word only if it is part of the most frequent words of our vocabulary
            index = word_index.get(word)
            if (index is not None) and (index < vocab_size) and (coefs.size == emb_dim):
                word_vectors[word] = coefs

    print('Number of pre-trained word vectors in database       : {}'.format(i_line + 1))
    print("Number of our words with a pre-trained embedding     : {}".format(len(word_vectors)))
    print("Percentage of our words with a pre-trained embedding : {:.3f}%".format(100 * len(word_vectors) / vocab_size))

    # Init a weight matrix for words in training docs matching pre-trained embeddings statistics
    # get mean and std values of pre-trained embeddings
    all_embs = np.stack(word_vectors.values())
    emb_mean, emb_std = np.mean(all_embs, axis=0), np.std(all_embs, axis=0)
    del all_embs
    # init matrix to embeddings statistical distribution
    embedding_matrix = np.random.normal(emb_mean, emb_std, (vocab_size, emb_dim))

    # Loop on words in our documents to update known word vectors
    for word, index in word_index.items():
        # if word isn't enough used in our documents, ignore it
        if index >= vocab_size:
            continue
        # otherwise, fill embedding matrix with pre-trained vector corresponding to this word
        embedding_vector = word_vectors.get(word)
        if embedding_vector is not None:
            embedding_matrix[index, :] = embedding_vector

    return embedding_matrix


def load_word2vec_embeddings_index(path_to_emb_file):
    """
    load embeddings file and build embeddings index

    TODO: adapter a la selection de differents fichiers sources/dim associees
    """
    google_model = KeyedVectors.load_word2vec_format(path_to_emb_file, binary=True, limit=100000)
    google_model_words = list(google_model.wv.vocab)
    print("Number of words in pre trained word2vec:", len(google_model_words))

    embeddings_index_google = dict()
    for googleRetainedWord in google_model_words:
        embeddings_index_google[googleRetainedWord] = google_model[googleRetainedWord]
    print('Loaded %s word vectors.' % len(embeddings_index_google))

    del google_model  # FREE RAM ! END MEMORY GULAG !

    return (embeddings_index_google)


def init_word2vec_embeddings(embeddings_index_google, emb_dim, vocab_size):
    """
    si probleme de RAM, charger les donnees train/test et les libraires Keras
    apres avoir lance cette fonction ! (bref, liberez delivrez la RAM)

    TODO: adapter à la selection de differents fichiers sources/dim associees
    """
    # get mean and std values of pre-trained embeddings
    all_embs_google = np.stack(embeddings_index_google.values())

    del embeddings_index_google  # will have to recreate it when calling load_word2vec_embeddings()

    emb_mean_google, emb_std_google = np.mean(all_embs_google, axis=0), np.std(all_embs_google, axis=0)

    del all_embs_google  # FREE RAM ! END MEMORY GULAG !

    # init matrix to embeddings distribution
    embedding_matrix_google = np.random.normal(emb_mean_google, emb_std_google, (vocab_size, emb_dim))

    # on retourne la matrice qui DOIT ENCORE ETRE COMPLETEE PAR LE TOKENIZER
    # (operation impossible avant car cela implique de surcharger la RAM avec le tokenizer)
    return (embedding_matrix_google)


def load_word2vec_embeddings(embedding_matrix_google, embeddings_index_google, tokenizer, vocab_size):
    """
    adapter a la selection de differents fichiers sources/dim associees

    embedding_matrix_google is result from init_word2vec_embeddings
    """
    word_with_goggle_emb = 0

    for word, index in tokenizer.word_index.items():
        if index > vocab_size - 1:  # determine a quel point on s'interesse aux mots moins importants d'apres Glove
            continue
        else:
            try:
                embedding_vector_google = embeddings_index_google[
                    word]  # on va chercher le mot dans word2vec embeddings
            except KeyError:
                embedding_vector_google = None
            if embedding_vector_google is not None:
                word_with_goggle_emb += 1
                embedding_matrix_google[index, :] = embedding_vector_google

    print("Number of words with a word2vec embedding:", word_with_goggle_emb)
    print("Percentage of words with a word2vec embedding:", word_with_goggle_emb / vocab_size)

    # on retourne la matrice qui sera un parametre de la fonction Embedding de Keras
    return (embedding_matrix_google)

import numpy as np
from gensim.models import KeyedVectors  # to load word2vec pretrained embeddings


def load_glove_embeddings(emb_dim, vocab_size, path_to_emb_file, tokenizer):
    """
    1) create complete embeddings matrix to compute statistical distribution
    2) with stat. distribution initialize properly emb. matrix to avoid emb. sharing
    3) update init. emb. matrix with emb. associated with our corpus' vocabulary

    TODO: adapter à la selection de differents fichiers sources/dim associees
    """
    embeddings_index = dict()
    with open(path_to_emb_file) as file:
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            # print(coefs.shape)  # le code est bien compatible avec le format renvoye par word2vec
            embeddings_index[word] = coefs
    print('Loaded %s word vectors.' % len(embeddings_index))

    # Create a weight matrix for words in training docs

    # get mean and std values of pre-trained embeddings
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = np.mean(all_embs, axis=0), np.std(all_embs, axis=0)
    # init matrix to embeddings statistical distribution
    embedding_matrix = np.random.normal(emb_mean, emb_std, (vocab_size, emb_dim))

    # loop on words in our documents

    word_with_glove_emb = 0

    for word, index in tokenizer.word_index.items():
        # if word isn't enough used in documents, ignore it
        if index >= vocab_size:
            continue
        # otherwise, fill embedding matrix with pre-trained vector corresponding to this word
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            word_with_glove_emb += 1
            embedding_matrix[index, :] = embedding_vector

    print("Number of words with a GloVe embedding:", word_with_glove_emb)
    print("Percentage of words with a GloVe embedding:", word_with_glove_emb / vocab_size)

    # on retourne la matrice qui sera un parametre de la fonction Embedding de Keras
    return (embedding_matrix)


def load_word2vec_embeddings_index(path_to_emb_file):
    """
    load embeddings file and build embeddings index

    TODO: adapter à la selection de differents fichiers sources/dim associees
    """
    google_model = KeyedVectors.load_word2vec_format(path_to_emb_file, binary=True)
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

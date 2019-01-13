import numpy as np
from gensim.models import KeyedVectors # to load word2vec pretrained embeddings

def load_GloVe_embeddings(emb_dim,vocab_size,pathToEmbFile,tokenizer):
    """
    1) create complete emebeddings matrix to compute statistical distribution
    2) with stat. distribution initialize properly emb. matrix to avoid emb. sharing
    3) update init. emb. matrix with emb. associated with our corpus' vocabulary

    TODO: adapter à la sélection de différents fichiers sources/dim associées
    """
    VOCAB_SIZE = vocab_size
    EMBEDDING_DIM = emb_dim
    
    embeddings_index = dict()
    with open(pathToEmbFile) as file:
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            #print(coefs.shape) # le code est bien compatible avec le format renvoyé par word2vec
            embeddings_index[word] = coefs
    print('Loaded %s word vectors.' % len(embeddings_index))

    # Create a weight matrix for words in training docs

    # get mean and std values of pre-trained embeddings
    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = np.mean(all_embs, axis=0), np.std(all_embs, axis=0)
    # init matrix to embeddings statistical distribution
    embedding_matrix = np.random.normal(emb_mean, emb_std, (VOCAB_SIZE, EMBEDDING_DIM))

    # loop on words in our documents

    wordWithGloveEmb = 0

    for word, index in tokenizer.word_index.items():
        # if word isn't enough used in documents, ignore it
        if index >= VOCAB_SIZE:
            continue
        # otherwise, fill embedding matrix with pre-trained vector corresponding to this word
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            wordWithGloveEmb += 1
            embedding_matrix[index, :] = embedding_vector

    print("Number of words with a GloVe embedding:",wordWithGloveEmb)
    print("Percentage of words with a GloVe embedding:",wordWithGloveEmb/VOCAB_SIZE)

    # on retourne la matrice qui sera un paramètre de la fonction Embedding de Keras
    return(embedding_matrix)

def load_word2vec_embeddings_index(pathToEmbFile):
    """
    load embeddings file and build embeddings index

    TODO: adapter à la sélection de différents fichiers sources/dim associées
    """

    google_model = KeyedVectors.load_word2vec_format(pathToEmbFile, binary=True)
    google_model_words = list(google_model.wv.vocab)
    print("Number of words in pre trained word2vec:",len(google_model_words))

    embeddings_index_google = dict()
    for googleRetainedWord in google_model_words:
        embeddings_index_google[googleRetainedWord] = google_model[googleRetainedWord]
    print('Loaded %s word vectors.' % len(embeddings_index_google))

    del google_model # FREE RAM ! END MEMORY GULAG !

    return(embeddings_index_google)

def init_word2vec_embeddings(embeddings_index_google,emb_dim,vocab_size):
    """
    si problème de RAM, charger les données train/test et les libraires Keras
    après avoir lancé cette fonction ! (bref, libérez délivrez la RAM)

    TODO: adapter à la sélection de différents fichiers sources/dim associées
    """

    EMBEDDING_DIM_GOOGLE = emb_dim
    VOCAB_SIZE = vocab_size

    # get mean and std values of pre-trained embeddings
    all_embs_google = np.stack(embeddings_index_google.values())
    
    del all_embs_google
    
    emb_mean_google, emb_std_google = np.mean(all_embs_google, axis=0), \
                                        np.std(all_embs_google, axis=0)

    del all_embs_google # FREE RAM ! END MEMORY GULAG !

    # init matrix to embeddings distribution
    embedding_matrix_google = np.random.normal(emb_mean_google, emb_std_google,\
                                               (VOCAB_SIZE, EMBEDDING_DIM_GOOGLE))

    # on retourne la matrice qui DOIT ENCORE ÊTRE COMPLÉTÉE PAR LE TOKENIZER
    # (opération impossible avant car cela implique de surcharger la RAM avec le tokenizer)
    return(embedding_matrix_init)

def load_word2vec_embeddings(embedding_matrix_google,embeddings_index_google,tokenizer):
    """
    adapter à la sélection de différents fichiers sources/dim associées

    embedding_matrix_google is result from init_word2vec_embeddings
    """

    wordWithGoogleEmb = 0

    for word, index in tokenizer.word_index.items():
        if index > VOCAB_SIZE - 1: # détermine à quel point on s'intéresse aux mots moins importants d'après Glove
            continue
        else:
            try:
                embedding_vector_google = embeddings_index_google[word] # on va chercher le mot dans word2vec embeddings
            except KeyError:
                embedding_vector_google = None
            if embedding_vector_google is not None:
                wordWithGoogleEmb += 1
                embedding_matrix_google[index,:] = embedding_vector_google

    print("Number of words with a word2vec embedding:",wordWithGoogleEmb)
    print("Percentage of words with a word2vec embedding:",wordWithGoogleEmb/VOCAB_SIZE)

    # on retourne la matrice qui sera un paramètre de la fonction Embedding de Keras
    return(embedding_matrix_google)

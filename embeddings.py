import numpy as np


def load_pretrained_embeddings(word_index, vocab_size, emb_dim=200, source='glove_twitter', norm_init=True):
    """
    Build embeddings layer weights from pre-trained word vectors.
    :param word_index: dictionary mapping our words to their numerical index (sorted from most to least frequent)
    :param vocab_size: number of most frequent words we want to consider
    :param emb_dim: dimension of the embeddings vectors
    :param source: the pre-trained word vectors dataset to use to build embedding matrix, in
                   {'glove_twitter', 'glove_wikipedia', 'word2vec_googlenews', 'fasttext_crawl'}
    :param norm_init: if True, init emb matrix with pre-trained vectors normal distribution, otherwise zeros.
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
        embeddings_file = 'embeddings/GoogleNews-vectors-negative300.bin'

    elif source == 'fasttext_crawl':
        if emb_dim != 300:
            raise ValueError("'fasttext_crawl' embeddings are only available for dimension 300")
        embeddings_file = 'embeddings/crawl-300d-2M.vec'

    else:
        raise ValueError("Unknown pre-trained embeddings source. "
                         "Must be in {'glove_twitter', 'glove_wikipedia', 'word2vec_googlenews'}")

    # parse embeddings file
    if source.split('_')[0] in {"glove", "fasttext"}:
        word_vectors, n_emb = load_txt_embeddings(embeddings_file, word_index, vocab_size, emb_dim)
    elif source.split('_')[0] == "word2vec":
        word_vectors, n_emb = load_bin_embeddings(embeddings_file, word_index, vocab_size, emb_dim)

    print('Number of pre-trained word vectors in database       : {}'.format(n_emb))
    print("Number of our words with a pre-trained embedding     : {}".format(len(word_vectors)))
    print("Percentage of our words with a pre-trained embedding : {:.3f}%".format(100 * len(word_vectors) / vocab_size))

    # Init a weight matrix for words in training docs matching pre-trained embeddings statistics
    # get mean and std values of pre-trained embeddings
    if norm_init:
        all_embs = np.stack(word_vectors.values())
        emb_mean, emb_std = np.mean(all_embs, axis=0), np.std(all_embs, axis=0)
        del all_embs
        # init matrix to embeddings statistical distribution
        embedding_matrix = np.random.normal(emb_mean, emb_std, (vocab_size, emb_dim))
    # Init a weight matrix for words in training docs with zeros
    else:
        embedding_matrix = np.zeros((vocab_size, emb_dim))

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


def load_txt_embeddings(embeddings_file, word_index, vocab_size, emb_dim):
    """
    Get GloVe or FastText pre-trained word vectors matching to our words of interest.
    The input file is a text file (with an optionnal header) of format : <word> <dim1> ... <dimN>.
    """
    word_vectors = {}
    with open(embeddings_file, 'r') as file:

        # remove header for fasttext .vec file
        if embeddings_file.split('.')[-1] == "fasttext":
            _ = file.readline()

        for i_line, line in enumerate(file):
            # display loading every 1000 words
            if i_line % 1000 == 0:
                print('Loading word vector {}...'.format(i_line), end="\r")

            # read word and vector
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype=np.float32)

            # save word only if it is part of the most frequent words of our vocabulary
            index = word_index.get(word)
            if (index is not None) and (index < vocab_size) and (coefs.size == emb_dim):
                word_vectors[word] = coefs
        n_emb = i_line + 1

    return word_vectors, n_emb


def load_bin_embeddings(embeddings_file, word_index, vocab_size, emb_dim):
    """
    Get Word2Vec pre-trained word vectors matching to our words of interest.
    The input file is a txt/binary file (with a txt header) of format : <word> <dim1> ... <dimN>.
    """
    word_vectors = {}
    with open(embeddings_file, "rb") as file:
        # read and use header
        header = file.readline()
        n_emb, emb_dim = map(int, header.split())
        vector_binary_len = np.dtype('float32').itemsize * emb_dim

        # parse file
        for i_line in range(n_emb):
            # display loading every 1000 words
            if i_line % 1000 == 0:
                print('Loading word vector {}...'.format(i_line), end="\r")

            # read word
            word = ''
            while True:
                ch = file.read(1)
                if ch == b' ':
                    break
                if ch != '\n':
                    word += ch.decode('cp437')

            # read vector
            coefs = np.frombuffer(file.read(vector_binary_len), dtype='float32')

            # save word only if it is part of the most frequent words of our vocabulary
            index = word_index.get(word)
            if (index is not None) and (index < vocab_size) and (coefs.size == emb_dim):
                word_vectors[word] = coefs

    return word_vectors, n_emb

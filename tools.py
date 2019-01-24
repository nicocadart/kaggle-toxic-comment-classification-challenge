import csv
import numpy as np
import re  # for regex

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import Callback
from keras.models import model_from_json
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


##########################################
########### EVALUATION ###################
##########################################


class RocAucEvaluation(Callback):
    """
    Keras callback to be called after each epoch to compute ROC-AUC score on validation set.
    """

    def __init__(self, validation_data=(), interval=1):
        """
        Object constructor
        :param validation_data: tuple (X_valid, y_valid
        :param interval: callback called after each 'interval' epochs
        """
        super(Callback, self).__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("epoch: {:d} - val_roc_auc: {:.4f}".format(epoch + 1, score))


def evaluate(y_true, y_pred):
    """
    @brief: Compute column wise ROC AUC on prediction/truth samples

    @param:
        y_true: ndarray, (n_samples, n_classes), binary label indicators for each class
        y_pred: ndarray, (n_samples, n_classes), prediction probability for each class

    @return:
        roc_auc_score: column-wise mean of roc auc on each label (non weighted)
    """
    return roc_auc_score(y_true, y_pred, average='macro')


def submission(y, id_list, name, dir='data/', list_classes=CLASSES):
    """
    @brief: Takes multi-class prediction probability and output a csv file ready for
                kaggle submission

    @param:
        y: ndarray, (n_samples, n_classes), prediction probability for each class
        id: list of string, id of comment to be used in submission
        name: string, name of submission to be append to the filename
        dir: string, path to save folder
        list_classes: list, list of string naming each class
    """
    with open(dir + name + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['id'] + list_classes)

        for (i, id) in enumerate(id_list):
            writer.writerow([id] + list(y[i]))


##########################################
########### DATA MANAGEMENT ##############
##########################################


def load_data(path='data/', list_classes=CLASSES):
    """
    @brief:
        Load data and get comment, labels for train, test

    @param:
        path: string, path to folder
        list_classes: list, list of string naming each class

    @return:
        data_train: list of comments, used for train
        y_train: ndarray of labels (categorical encoding)
        data_test: list of comments, used for test
        id_test: list of id, used for output
    """
    train_set = pd.read_csv(path + 'train.csv')
    test_set = pd.read_csv(path + 'test.csv')

    y_train = train_set[list_classes].values

    data_train = list(train_set["comment_text"])
    data_test = list(test_set["comment_text"])

    id_test = list(test_set["id"])

    return data_train, y_train, data_test, id_test


def load_nnet(name, dir="models/"):
    """
    Load a pre-trained keras model from disk.
    :param name: path of the model, without the extension 'json' or 'h5'
    :param dir: directory where is located the model
    :return: the keras model to compile
    """
    # load json and create model
    with open("{}{}.json".format(dir, name), "r") as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("{}{}.h5".format(dir, name))
    return model


def save_nnet(model, name, dir="models/"):
    """
    Save a keras model to disk.
    :param model: the trained model
    :param name: path of the model, without the extension 'json' or 'h5'
    :param dir: directory where to store the model
    """
    # serialize model to JSON
    model_json = model.to_json()
    with open("{}{}.json".format(dir, name), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("{}{}.h5".format(dir, name))


#########################################
########### PRE-PROCESSING ##############
#########################################


def clean_comment(comment, lower=True, lemma=True, stop_words=True):
    """
    @brief: This function receives comments and returns clean word-list
    (from https://www.kaggle.com/jagangupta/stop-the-s-toxic-comments-eda)

    @param:
        comment: string, sentence to be cleaned

    @return:
        clean_comment: string, cleaned sentence
    """
    # Convert to lower case , so that Hi and hi are the same
    if lower:
        comment = comment.lower()

    # remove \n
    comment = re.sub("\\n", " ", comment)
    # remove leaky elements like ip, user
    comment = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "", comment)
    # removing usernames
    comment = re.sub("\[\[.*\]", "", comment)
    # remove any non alphanum, digit character
    clean_comment = re.sub("\W+", " ", comment)

    # If processing has to be on words
    if lemma or stop_words:

        # Split the sentences into words
        tokenizer = TweetTokenizer(reduce_len=True)
        words = tokenizer.tokenize(clean_comment)

        # remove stop_words
        if stop_words:
            eng_stopwords = set(stopwords.words("english"))
            words = [w for w in words if w not in eng_stopwords]

        # reduce words to lemma
        if lemma:
            lem = WordNetLemmatizer()
            words = [lem.lemmatize(word, "v") for word in words]

        clean_comment = " ".join(words)

    return clean_comment


def transform_dataset(dataset, func=clean_comment, kwargs={}):
    """
    @brief: apply transform func on a dataset of comments

    @param:
        dataset: iterable of strings, dataset of comments to be transformed
        func: function, to be applied on a string to be transformed
        kwargs: other params for 'func'

    @return:
        transform_dataset: iterable of strings, transformed comments
    """
    transform_dataset = dataset.copy()
    print_every = int(np.ceil(len(dataset) / 1000))

    for (i_txt, txt) in enumerate(dataset):

        if i_txt % print_every == 0:
            print('Transformation: {:.2f}%    '.format(100 * (i_txt + 1) / len(dataset)), end='\r')

        transform_dataset[i_txt] = func(txt, **kwargs)

    print('Transformation: 100%       ')
    return transform_dataset


def pad_comment(comment, maxlen=200, join_bool=True, padval="<pad>"):
    """
    @brief: limit lengths of comments to len_padding words

    @param:
        comment: string, sentence to be padded with padval
        maxlen: int, length of all sentences

    @return:
        if join=True, return a single string with the padded comment
        else, return a list of tokens of len maxlen
    """
    tokenizer = TweetTokenizer(reduce_len=True)
    # Split the sentences into words
    words = tokenizer.tokenize(comment)

    if len(words) < maxlen:
        pad_words = words + [padval] * (maxlen - len(words))
    else:
        pad_words = words[:maxlen]

    if join_bool:
        clean_comment = " ".join(pad_words)
    else:
        clean_comment = pad_words

    return clean_comment


def encode(data_train, data_test, vectorizer=TfidfVectorizer()):
    """
    @brief: encode textual data to numerical, and apply padding if necessary

    @param:
        data_train: list of comments, used for train
        data_test: list of comments, used for testing
        vectorizer: object, should have a method fit and transform (see sklearn doc)

    @return:
        X_train: scipy.sparse, (n_samples_train, max_features)
        X_test: scipy.sparse, (n_samples_test, max_features)
     """
    print('ENCODING: Fitting vectorizer to data')
    vectorizer.fit(data_train + data_test)

    print('ENCODING: transforming data to numerical')
    X_train = vectorizer.transform(data_train)
    X_test = vectorizer.transform(data_test)

    return X_train, X_test


class TokenVectorizer(Tokenizer):
    """
    Tokenize a dataset and create numpy padded representation.
    Each word is replaced by its integer vocanulary index, and the sequence is optionally padded with zeros.
    Wrapper for the Tokenizer object of Keras, with padding added
    """

    def __init__(self, max_len=-1, max_features=30000, **kwargs):
        super().__init__(num_words=max_features, **kwargs)
        self.max_len = max_len
        self.max_features = max_features

    def fit(self, text):
        return self.fit_on_texts(text)

    def transform(self, text):
        list_tokens = self.texts_to_sequences(text)
        if self.max_len > 0:
            return pad_sequences(list_tokens, maxlen=self.max_len, truncating='post')
        else:
            return list_tokens


if __name__ == '__main__':
    data_train, y_train, data_test, id_test = load_data()

    #####################
    ###### ENCODING #####
    #####################

    # -------
    ### CBOW

    # Create a CBOW vectorizer for english words, without accent,
    ## limiting the vocabulary to 30000 words max.

    count_vectorizer = CountVectorizer(analyzer='word', stop_words='english',
                                       strip_accents='unicode', max_features=30000)

    # -------
    ### Hash

    # Create a CBOW vectorizer for english words, without accent. No limit on vocab size

    hash_vectorizer = HashingVectorizer(analyzer='word', stop_words='english',
                                        strip_accents='unicode')

    # -------
    ### TFIDF

    # Create a TFIDF vectorizer for english words, (only unigrams), limiting the vocabulary to
    # 30000 words max.and filtering words with frequency under 10.
    ## Remove accents, and using idf for filtering, with smoothing to avoid zero division

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', analyzer='word', ngram_range=(1, 1),
                                       min_df=10, max_features=30000,
                                       strip_accents='unicode', use_idf=1, smooth_idf=1,
                                       sublinear_tf=1)

    # ------------
    ### Tokenizer

    ## Tokenize the corpus, with only the 30000 most commons tokens, and pad the sentences to 200
    tokens_vectorizer = TokenVectorizer(max_len=200, max_features=30000)

    X_train, X_test = encode(data_train, data_test, vectorizer=tokens_vectorizer)

    # Vocabulary can be extracted from the vectorizer object (if tdidf or count)
    # print(count_vectorizer.get_feature_names())

    y_test = np.ones((X_test.shape[0], y_train.shape[1]))
    submission(y_test, id_test)

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
        :param validation_data: tuple (X_valid, y_valid)
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


def load_data(path='data/', language='', list_classes=CLASSES):
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
    train_set = pd.read_csv(path + 'train'+ language +'.csv')
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
    :return: the keras model ready to fit
    """
    # load json and create model
    with open("{}{}.json".format(dir, name), "r") as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("{}{}.h5".format(dir, name))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
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


class CommentLength:
    """
    Object returning the number of words of each comment.
    """
    def transform(self, comment):
        # get comment length (taking care of empty comments to avoid division by 0)
        return max(len(comment.split()), 1)


class CharCounter:
    """
    Object returning the number of occurences of a set of chars in each comment.
    """
    def __init__(self, chars_set={'!', '?'}, divide_by_len=False):
        self.chars_set = chars_set
        self.divide_by_len = divide_by_len

    def transform(self, comment):
        # get comment length (taking care of empty comments to avoid division by 0)
        if self.divide_by_len:
            comment_length = max(len(comment.split()), 1)
            char_count = sum(char in self.chars_set for char in comment) / comment_length
        else:
            char_count = sum(char in self.chars_set for char in comment)
        return char_count


class UppercaseWordsCounter:
    """
    Object returning number of words in capital letters in a comment.
    """
    def __init__(self, divide_by_len=False):
        self.divide_by_len = divide_by_len
        self.re_I = re.compile('(\s*)I(\s*)')

    def transform(self, comment):
        # lower 'I' in comment to ignore it
        comment_without_I = self.re_I.sub('i', comment)
        # count number of words in capital letters
        if self.divide_by_len:
            words = comment_without_I.split()
            count = sum(map(str.isupper, words)) / max(len(words), 1)
        else:
            count = sum(map(str.isupper, comment_without_I.split()))
        return count


class CommentCleaner:
    """
    This object is used to clean commments.
    """
    def __init__(self, clean=True, lower=True, lemma=True, stop_words=True):
        # save config
        self.clean = clean
        self.lower = lower
        self.lemmatize = lemma
        self.remove_stop_words = stop_words

        # init regexp objects
        self.re_remove_newline = re.compile("\\n")
        self.re_remove_leaks = re.compile("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
        self.re_remove_usernames = re.compile("\[\[.*\]")
        self.re_remove_nonalphadigit = re.compile("\W+")  # TODO : check if it isn't better to keep punctation

        # init other processing objects
        self.tokenizer = TweetTokenizer(reduce_len=True)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def transform(self, comment):
        """
        @brief: This function receives comments and returns clean word-list
        (from https://www.kaggle.com/jagangupta/stop-the-s-toxic-comments-eda)

        @param:
            comments: list of strings, sentences to be cleaned

        @return:
            clean_comments: list of strings, cleaned sentences
        """
        # Remove some noisy chars
        if self.clean:
            # remove \n
            comment = self.re_remove_newline.sub(" ", comment)
            # remove leaky elements like ip, user
            comment = self.re_remove_leaks.sub("", comment)
            # removing usernames
            comment = self.re_remove_usernames.sub("", comment)
            # remove any non alphanum, digit character
            comment = self.re_remove_nonalphadigit.sub(" ", comment)

        # Convert to lower case , so that Hi and hi are the same
        if self.lower:
            comment = comment.lower()

        # If processing has to be on words
        if self.lemmatize or self.remove_stop_words:

            # Split the sentences into words
            words = self.tokenizer.tokenize(comment)

            # remove stop_words
            if self.remove_stop_words:
                words = [w for w in words if w not in self.stop_words]

            # reduce words to lemma
            if self.lemmatize:
                words = [self.lemmatizer.lemmatize(word, "v") for word in words]

            comment = " ".join(words)

        return comment


class CommentPadder:
    """
    Object used to pad comments.
    """
    def __init__(self, maxlen=200, join_bool=True, padval="<pad>"):
        """

        :param maxlen:  int, length of all output sentences
        :param join_bool: if join=True, return a single string with the padded comment
        :param padval: the string used to pad shorter sequences
        """
        # save config
        self.maxlen = maxlen
        self.padval = padval
        self.join_bool = join_bool

        # init other processing objects
        self.tokenizer = TweetTokenizer(reduce_len=True)

    def transform(self, comment):
        """
        Limit lengths of comments to len_padding words
        :param comment: string, sentence to be truncated or padded with padval to length maxlen
        :return: if join=True, return a single string with the padded comment,  else, a list of tokens of len maxlen
        """
        # Split the sentences into words
        words = self.tokenizer.tokenize(comment)

        if len(words) < self.maxlen:
            pad_words = words + [self.padval] * (self.maxlen - len(words))
        else:
            pad_words = words[:self.maxlen]

        if self.join_bool:
            clean_comment = " ".join(pad_words)
        else:
            clean_comment = pad_words

        return clean_comment


def transform_dataset(dataset, transformer=CommentCleaner, kwargs={}, n_prints=1000):
    """
    @brief: apply transform func on a dataset of comments

    @param:
        dataset: iterable of strings, dataset of comments to be transformed
        transformer: object with a transform() method to be applied on a string to be transformed
        kwargs: init params to transformer

    @return:
        transform_dataset: iterable of strings, transformed comments
    """
    transform_dataset = dataset.copy()
    print_every = int(np.ceil(len(dataset) / n_prints))
    transformer = transformer(**kwargs)

    for (i_txt, txt) in enumerate(dataset):

        if i_txt % print_every == 0:
            print('Transformation: {:.2f}%    '.format(100 * (i_txt + 1) / len(dataset)), end='\r')

        transform_dataset[i_txt] = transformer.transform(txt)

    print('Transformation: 100%       ')
    return transform_dataset


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
    Each word is replaced by its integer vocabulary index, and the sequence is optionally padded with zeros.
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

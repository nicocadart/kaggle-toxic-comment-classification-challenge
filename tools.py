import csv
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import re    #for regex
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


##########################################
########### EVALUATION ###################
##########################################

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


def submission(y, id_list, name, path='data/', list_classes=CLASSES):
    """ @brief: Takes multi-class prediction probability and output a csv file ready for
                kaggle submission
        @param:
            y: ndarray, (n_samples, n_classes), prediction probability for each class
            id: list of string, id of comment to be used in submission
            name: string, name of submission to be append to the filename
            path: string, path to save folder
            list_classes: list, list of string naming each class
    """

    with open(path + name + '.csv', 'w') as csvfile:

        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['id']+list_classes)

        for (i, id) in enumerate(id_list):
            writer.writerow([id] + list(y[i]))

    print('SUBMISSION WRITTEN')

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
    print('LOADING DATA')
    train_set = pd.read_csv(path + 'train.csv')
    test_set = pd.read_csv(path + 'test.csv')

    y_train = train_set[list_classes].values

    data_train = list(train_set["comment_text"])
    data_test = list(test_set["comment_text"])

    id_test = list(test_set["id"])

    return data_train, y_train, data_test, id_test


def encode(data_train, data_test, vectorizer=TfidfVectorizer):
    """ @brief: encode textual data to numerical, and apply padding if necessary

        @param:
            data_train: list of comments, used for train
            data_test: list of comments, used for testing
            vectorizer: class, should have a method fit and transform (see sklearn doc)
            padding: int, if 0 no padding, else, data is completed such as each sample is
                            the same length

        @return:
            X_train: scipy.sparse, (n_samples_train, max_features)
            X_test: scipy.sparse, (n_samples_test, max_features)

     """
    print('ENCODING: Fitting vectorizer to data')
    vectorizer.fit(data_train + data_test)

    print('ENCODING: transforming data to numerical')
    X_train =  vectorizer.transform(data_train)
    X_test = vectorizer.transform(data_test)

    return X_train, X_test


class TokenVectorizer:
    """ Tokenize a dataset and create numpy padded representation.
        Wrapper for the Tokenizer object of Keras, with padding added """

    def __init__(self, max_len=200, max_features=30000):

        self.max_len = max_len
        self.max_features = max_features
        self.tokenizer = Tokenizer(num_words=self.max_features)

    def fit(self, text):
        return self.tokenizer.fit_on_texts(text)

    def transform(self, text):
        list_tokens = self.tokenizer.texts_to_sequences(text)
        return pad_sequences(list_tokens, maxlen=self.max_len)


#########################################
########### PRE-PROCESSING ##############
#########################################


def clean_comment(comment):
    """
    @brief: This function receives comments and returns clean word-list
    (from https://www.kaggle.com/jagangupta/stop-the-s-toxic-comments-eda)

    @param:
        comment: string, sentence to be cleaned

    @return:
        clean_comment: string, cleaned sentence
    """

    eng_stopwords = set(stopwords.words("english"))

    lem = WordNetLemmatizer()
    tokenizer = TweetTokenizer()

    #Convert to lower case , so that Hi and hi are the same
    comment = comment.lower()
    #remove \n
    comment = re.sub("\\n", " ", comment)
    # remove leaky elements like ip,user
    comment = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "", comment)
    #removing usernames
    comment = re.sub("\[\[.*\]", "", comment)

    #Split the sentences into words
    words = tokenizer.tokenize(comment)

    # remove stop_words
    words = [lem.lemmatize(word, "v") for word in words]
    words = [w for w in words if not w in eng_stopwords]

    clean_comment = " ".join(words)

    # remove any non alphanum, digit character
    clean_comment = re.sub("\W+"," ",clean_comment)

    return(clean_comment)


def pad_comment(comment, maxlen=200, join_bool=True):
    """
        @brief: limit lengths of comments to len_padding words

        @param:
            comment: string, sentence to be padded with '<pad>'
            maxlen: int, lenght of all sentences

        @return:
            if join=True, return a single string with the padded comment
            else, return a list of tokens of len maxlen
    """

    tokenizer=TweetTokenizer()
    #Split the sentences into words
    words=tokenizer.tokenize(comment)

    if len(words) < maxlen:
        pad_words = words + ['<pad>'] * (maxlen - len(words))

    elif len(words) > maxlen:
        pad_words = words[:maxlen]

    if join_bool:
        clean_comment = " ".join(pad_words)
    else:
        clean_comment = pad_words

    return clean_comment


def transform_dataset(dataset, func=clean_comment, name='Standard cleaning', kwargs={}):
    """
        @brief: apply transform func on a dataset of comments

        @param:
            dataset: iterable of strings, dataset of comments to be transformed
            func: function, to be applied on a string to be transformed
            name: string

        @return:
            transform_dataset: iterable of strings, transformed comments
    """

    transform_dataset = dataset.copy()

    for (i_txt, txt) in enumerate(dataset):

        print('Transformation: {}%'.format(round(100*i_txt/len(dataset), 2)), end='\r')
        transform_dataset[i_txt] = func(txt, **kwargs)

    return transform_dataset


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

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', analyzer='word', ngram_range=(1,1),
                                       min_df=10, max_features=30000,
                                       strip_accents='unicode', use_idf=1,smooth_idf=1,
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

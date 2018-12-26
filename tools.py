import csv
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


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


def submission(y, id_list, path='data/submission.csv', list_classes=CLASSES):
    """ @brief: Takes multi-class prediction probability and output a csv file ready for
                kaggle submission
        @param:
            y: ndarray, (n_samples, n_classes), prediction probability for each class
            id: list of string, id of comment to be used in submission
            path: string, path where to save the csv file
            list_classes: list, list of string naming each class

    """

    with open(path, 'w') as csvfile:

        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['id']+list_classes)

        for (i, id) in enumerate(id_list):
            writer.writerow([id] + list(y[i]))

    print('SUBMISSION WRITTEN')




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


    X_train, X_test = encode(data_train, data_test, vectorizer=count_vectorizer)

    # Vocabulary can be extracted from the vectorizer object (if tdidf or count)
    # print(count_vectorizer.get_feature_names())


    y_test = np.ones((X_test.shape[0], len(CLASSES)))

    submission(y_test, id_test)

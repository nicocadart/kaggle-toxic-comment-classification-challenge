from keras.layers import Dense, Input, LSTM, GRU, Embedding, Bidirectional, Conv1D, concatenate
from keras.layers import Dropout, SpatialDropout1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.models import Model

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression

from scipy.optimize import minimize
from scipy import sparse
import numpy as np

from tools import evaluate


# TODO : add spatial dropout and/or batch norm
# TODO : stack LSTM layers in bidirectional_lstm()
# TODO : add regularization to limit over-fitting
# TODO : add normalization and dense layer after auxiliary input?

##########################################
########### NEURAL NETS ##################
##########################################

def yoon_kim(sentence_length=200, vocab_size=30000,
             n_filters=100, filters_sizes=(3, 5, 7),
             embedding_dim=150, embedding_matrix=None, train_embeddings=True,
             aux_input_dim=None):
    """
    Compile a Keras nnet model. The returned model is a convolutional net adapted to NLP, inspired from Yoon Kim aticle.
    :param sentence_length: fixed length of our truncated/padded numerical sentences.
    :param vocab_size: dimension of our vocabulary set.
    :param n_filters: number of kernels trained by each parallel conv layer.
    :param filters_sizes: kernel sizes of each parallel conv layer.
    :param embedding_dim: dimension of word vectors.
    :param embedding_matrix: the initial weights to give to embedding layer.
    :param train_embeddings: True if embedding layer is trainable or not.
    :param aux_input_dim: dimension of an auxiliary input added in the last dense part of the nnet.
    :return: the compiled keras model, ready to fit()
    """
    # input
    main_input = Input(shape=(sentence_length,))
    # embedding
    x = Embedding(vocab_size, embedding_dim, input_length=sentence_length, trainable=train_embeddings,
                  weights=[embedding_matrix] if embedding_matrix is not None else None)(main_input)

    # Specify each convolution layer and their kernel size i.e. n-grams
    conv_layers, pool_layers = [None] * len(filters_sizes), [None] * len(filters_sizes)
    for i_layer, filter_size in enumerate(filters_sizes):
        conv_layers[i_layer] = Conv1D(filters=n_filters, kernel_size=filter_size, activation='relu')(x)
        pool_layers[i_layer] = GlobalMaxPooling1D()(conv_layers[i_layer])

    # Gather all convolution layers
    x = concatenate([pool for pool in pool_layers], axis=1)

    # auxiliary input
    if aux_input_dim:
        aux_input = Input(shape=(aux_input_dim,), name='aux_input')
        # merge all inputs
        x = concatenate([x, aux_input])

    # x = Dropout(0.1)(x)
    # x = Dense(50, activation='relu')(x)
    # x = Dropout(0.1)(x)
    outp = Dense(6, activation='sigmoid')(x)

    # build final model
    model = Model(inputs=[main_input, aux_input] if aux_input_dim else main_input, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # ready for .fit() !
    return model


def bidirectional_lstm(sentence_length=200, vocab_size=30000,
                       # lstm_sizes=(60,),
                       embedding_dim=150, embedding_matrix=None, train_embeddings=True,
                       aux_input_dim=None):
    """
    Compile a Keras nnet model. The returned model is a RNN with bidirectionnal LSTM layers.
    :param sentence_length: fixed length of our truncated/padded numerical sentences.
    :param vocab_size: dimension of our vocabulary set.
    :param embedding_dim: dimension of word vectors.
    :param embedding_matrix: the initial weights to give to embedding layer.
    :param train_embeddings: True if embedding layer is trainable or not.
    :param aux_input_dim: dimension of an auxiliary input added in the last dense part of the nnet.
    :return: the compiled keras model, ready to fit()
    """
    # main input (comments)
    main_input = Input(shape=(sentence_length,), name='main_input')

    # embedding
    x = Embedding(vocab_size, embedding_dim, input_length=sentence_length, trainable=train_embeddings,
                  weights=[embedding_matrix] if embedding_matrix is not None else None)(main_input)
    x = SpatialDropout1D(0.1)(x)

    # LSTM layers
    # lstm_layers = [None] * len(lstm_sizes)
    # for i_layer, layer_size in enumerate(lstm_sizes):
    #     lstm_layers[i_layer] = Bidirectional(LSTM(layer_size, return_sequences=True))(x)
    #     x = lstm_layers[i_layer]
    x = Bidirectional(LSTM(60, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)

    # pooling
    max_pool = GlobalMaxPooling1D()(x)
    avg_pool = GlobalAveragePooling1D()(x)
    x = concatenate([max_pool, avg_pool])

    # auxiliary input
    if aux_input_dim:
        aux_input = Input(shape=(aux_input_dim,), name='aux_input')
        # merge all inputs
        x = concatenate([x, aux_input])

    # dense 1
    x = Dense(50, activation="relu")(x)
    # dropout 1
    x = Dropout(0.1)(x)

    # final dense
    outp = Dense(6, activation="sigmoid")(x)

    # build final model
    model = Model(inputs=[main_input, aux_input] if aux_input_dim else main_input, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # ready to .fit() !
    return model


def bidir_lstm_conv(sentence_length=200, vocab_size=30000,
                    embedding_dim=150, embedding_matrix=None, train_embeddings=True,
                    aux_input_dim=None):
    """
    Compile a Keras nnet model. The returned model is a mix of LSTM followed by convolutions layers.
    :param sentence_length: fixed length of our truncated/padded numerical sentences.
    :param vocab_size: dimension of our vocabulary set.
    :param embedding_dim: dimension of word vectors.
    :param embedding_matrix: the initial weights to give to embedding layer.
    :param train_embeddings: True if embedding layer is trainable or not.
    :param aux_input_dim: dimension of an auxiliary input added in the last dense part of the nnet.
    :return: the compiled keras model, ready to fit()
    """
    # main input (comments)
    main_input = Input(shape=(sentence_length,), name='main_input')

    # embedding
    x = Embedding(vocab_size, embedding_dim, input_length=sentence_length, trainable=train_embeddings,
                  weights=[embedding_matrix] if embedding_matrix is not None else None)(main_input)
    x = SpatialDropout1D(0.1)(x)

    x = Bidirectional(LSTM(60, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = Conv1D(60, kernel_size=3, padding="valid")(x)
    # add activation (ReLU) layer ?

    # pooling
    max_pool = GlobalMaxPooling1D()(x)
    avg_pool = GlobalAveragePooling1D()(x)
    x = concatenate([max_pool, avg_pool])

    # auxiliary input
    if aux_input_dim:
        aux_input = Input(shape=(aux_input_dim,), name='aux_input')
        # merge all inputs
        x = concatenate([x, aux_input])

    # dense 1
    # x = Dense(50, activation="relu")(x)
    # dropout 1
    # x = Dropout(0.1)(x)

    # final dense
    outp = Dense(6, activation="sigmoid")(x)

    # build final model
    model = Model(inputs=[main_input, aux_input] if aux_input_dim else main_input, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # ready to .fit() !
    return (model)


##########################################
########### STANDARD CLASSIFIERS #########
##########################################


class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1, solver='liblinear'):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs
        self.solver = solver


    def predict(self, x):
        """
        Give prediction with learned model on given data

        :param x: scipy sparse matrix, (n_samples, n_features)

        :return prediction: ndarray (n_samples, n_classes), binary prediction
                            for each class
        """
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))


    def predict_proba(self, x):
        """
        Give class probability prediction with learned model on given data

        :param X: scipy sparse matrix, (n_samples, n_features)

        :return prediction: ndarray (n_samples, n_classes), probability prediction
                            for each class given by the model
        """

        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))


    def fit(self, x, y):
        """
        Learn the NBSVM model on x according to y, compute prior and fit
         LogisticRegression regression on biased data

        :param x: scipy sparse matrix, (n_samples, n_features)
        :param y: numpy array, (n_samples,)

        :return model: the learned model
        """
        # Check that X and y have correct shape
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            idx = np.where(y == y_i)
            p = x[idx].sum(0)
            return (p + 1) / ((y == y_i).sum() + 1)

        self._r = sparse.csr_matrix(np.log(pr(x, 1, y) / pr(x, 0, y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs, solver=self.solver).fit(x_nb, y)
        return self


class OneVAllClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_classes, clf=NbSvmClassifier, params={}):

        self.models = []
        self.n_classes = n_classes

        if params != {}:
            for i_class in range(self.n_classes):
                param_clf = {}
                for (param, param_val) in params.items():
                    assert (len(param_val) == self.n_classes)
                    param_clf[param] = param_val[i_class]
                self.models.append(clf(**param_clf))
        else:
            for i_class in range(self.n_classes):
                self.models.append(clf())


    def fit(self, X, y):
        """
        Learn the 1vsAll model on x according to y, by fitting a model of the
            specified estimator for each class.

        :param x: scipy sparse matrix, (n_samples, n_features)
        :param y: numpy array, (n_samples,)

        :return model: the learned model
        """

        assert (y.shape[1] == self.n_classes)

        for i_class in range(self.n_classes):
            print('Fitting model {}:'.format(i_class))
            self.models[i_class].fit(X, y[:, i_class])

        return self


    def predict_proba(self, X):
        """
        Give class probability prediction with learned model on given data

        :param X: scipy sparse matrix, (n_samples, n_features)

        :return prediction: ndarray (n_samples, n_classes), probability prediction
                            for each class given by each 1vAll model
        """

        y_pred = np.ones((X.shape[0], self.n_classes))

        for i_class in range(self.n_classes):
            y_pred[:, i_class] = self.models[i_class].predict_proba(X)[:, 1]

        return y_pred


    def predict(self, X):
        """
        Give prediction with learned model on given data

        :param X: scipy sparse matrix, (n_samples, n_features)

        :return prediction: ndarray (n_samples, n_classes), binary prediction
                            for each class given by each 1vAll model
        """

        y_pred = np.ones((X.shape[0], self.n_classes))

        for i_class in range(self.n_classes):
            y_pred[:, i_class] = self.models[i_class].predict(X)

        return y_pred



##########################################
########### MODEL MIX ####################
##########################################


def model_mix(y_preds, y_true):
    """
    Load prediction for different models and compute optimal weights for model mix
    :param y_preds: list of tuple (str, ndarray), each tuple has the name of a
                    model and a ndarray of predicted proba for each class
    :param y_true: ndarray (n_samples, n_classes), should have the same shape as all y_preds

    :return optimal_weights: weights for ponderation between model prediction,
                             optimized for those models
     """
    names, y_pred_list = zip(*y_preds)

    for i in range(len(names)):
        assert (y_pred_list[i].shape == y_true.shape)
        print('Prediction score on {}: {:.4f}'.format(names[i], evaluate(y_true, y_pred_list[i])))

    # --------------------------------
    #  Find ensemble learning weights
    # --------------------------------

    # We want to minimize the logloss of the global prediction
    def score_func(weights, func=evaluate):
        import time
        final_prediction = 0
        for weight, prediction in zip(weights, y_pred_list):
            final_prediction += weight * prediction
        return -func(y_true, final_prediction)

    # Uniform initialisation
    init_weights = np.ones((len(y_pred_list),)) / len(y_pred_list)
    print('Initial mix score: {:.4f}'.format(-score_func(init_weights)))
    # Weights are in range [0; 1] and must sum to 1
    constraint = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    bounds = [(0, 1)] * len(y_pred_list)
    # Compute best weights (method chosen with the advice of Kaggle kernel)
    res = minimize(score_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraint)
    optimal_weights = res['x']

    print('Model mix prediction on train: {:.4f}'.format(-score_func(optimal_weights)))

    return optimal_weights


def model_mix_predict(y_preds, optimal_weights):
    """
    Take a list of sklearn models, weights and a dataset and return the weighted prediction
            over the samples
    :param X: ndarray, (n_samples, n_features), dataset to predict
    :param models: list of tuple (name, model, fit_params), with model a sklearn model already trained
    :param optimal_weights: list of float, weight for each model (sum(weight)==1)

    :return y_pred_p: ndarray, (n_samples, n_classes), probability for each class for each sample
    """
    names, y_pred_list = zip(*y_preds)

    final_prediction = np.zeros(y_pred_list[0].shape)
    for weight, prediction in zip(optimal_weights, y_pred_list):
        final_prediction += weight * prediction

    return final_prediction

from scipy.optimize import minimize
from scipy import sparse
import numpy as np

from tools import *
from embeddings import *
from models import *

from sklearn.model_selection import train_test_split



def model_mix(data_train, y_train_all, data_test, models):

    y_proba_pred = []
    models_output = []

    for (name, params) in models:

        print('----PREDICTING MODEL: {}'.format(name))

        ## CLEANING
        params_clean = params['params_clean']

        clean_data_train = transform_dataset(data_train, transformer=CommentCleaner, kwargs=params_clean)
        clean_data_test = transform_dataset(data_test, transformer=CommentCleaner, kwargs=params_clean)

        ## VECTORIZER
        # Object vectorizer should already be initialized
        vectorizer = params['vectorizer']
        X_train_all, _ = encode(clean_data_train, clean_data_test, vectorizer=vectorizer)

        SPLIT_VALID_RATIO = params['split_valid_ratio']

        USE_AUX_FEATURES = False
        ## AUX FEATURES
        if params['aux_features_params'] != {}:

            USE_AUX_FEATURES = True

            # TODO: encapsulate this part
            print("Computing comments length")
            comments_lengths_train = np.array(transform_dataset(data_train, transformer=CommentLength, n_prints=5))
            comments_lengths_test = np.array(transform_dataset(data_test, transformer=CommentLength, n_prints=5))

            print("Computing number of punctuation marks in comments")
            params_punct = {'divide_by_len': True, 'chars_set': {'!'}}
            comments_punctuation_train = np.array(transform_dataset(data_train, transformer=CharCounter, kwargs=params_punct))
            comments_punctuation_test = np.array(transform_dataset(data_test, transformer=CharCounter, kwargs=params_punct))

            print("Computing number of upper cased words in comments")
            params_upper = {'divide_by_len': True}
            comments_upperwords_train = np.array(transform_dataset(data_train, transformer=UppercaseWordsCounter, kwargs=params_upper))
            comments_upperwords_test = np.array(transform_dataset(data_test, transformer=UppercaseWordsCounter, kwargs=params_upper))

            # concatenation of auxiliary features
            X_aux_train_all = np.vstack((comments_lengths_train, comments_punctuation_train, comments_upperwords_train)).T
            X_aux_test = np.vstack((comments_lengths_test, comments_punctuation_test, comments_upperwords_test)).T

            # auxiliary input
            X_aux_train, X_aux_valid, _, _ = train_test_split(X_aux_train_all, y_train_all,
            test_size=SPLIT_VALID_RATIO)


        # numerical comments
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_all, y_train_all,
                                                              test_size=SPLIT_VALID_RATIO)

        # Check which type of model you should load
        if 'NN_batch_size' in params.keys():
            model = load_nnet(name)
            y_train_pred = model.predict([X_train, X_aux_train] if USE_AUX_FEATURES else X_train,
                                         batch_size=params['NN_batch_size'])
            y_valid_pred = model.predict([X_valid, X_aux_valid] if USE_AUX_FEATURES else X_valid,
                                         batch_size=params['NN_batch_size'])

        else:
            #model = load_sklearn(name)
            y_train_pred = model.predict_proba(hstack((X_train, X_aux_train)).astype(int).tocsr() if USE_AUX_FEATURES else X_train)
            y_valid_pred = model.predict_proba(hstack((X_valid, X_aux_valid)).astype(int).tocsr() if USE_AUX_FEATURES else X_valid)

        train_score = evaluate(y_train, y_train_pred)
        valid_score = evaluate(y_valid, y_valid_pred)
        print("ROC-AUC score for model {} on train set : {:.4f}".format(name, train_score))
        print("ROC-AUC score for model {} on validation set : {:.4f}".format(name, valid_score))

        ## PREDICT

        y_proba_pred.append(y_valid_pred)

        models_output.append((name, params, model))

    # --------------------------------
    #  Find ensemble learning weights
    # --------------------------------

    # We want to minimize the logloss of the global prediction
    def score_func(weights, func=evaluate):
        final_prediction = 0
        for weight, prediction in zip(weights, y_proba_pred):
            final_prediction += weight * prediction
        return func(y_valid, final_prediction)

    # Uniform initialisation
    init_weights = np.ones((len(y_proba_pred),)) / len(y_proba_pred)
    # Weights are in range [0; 1] and must sum to 1
    constraint = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    bounds = [(0, 1)] * len(y_proba_pred)
    # Compute best weights (method chosen with the advice of Kaggle kernel)
    res = minimize(score_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraint)
    optimal_weights = res['x']

    return models_output, optimal_weights


def model_mix_predict(X, models, optimal_weights, n_classes):
    """
    @brief: take a list of sklearn models, weights and a dataset and return the weighted prediction
            over the samples
    @param:
            X: ndarray, (n_samples, n_features), dataset to predict
            models: list of tuple (name, model, fit_params), with model a sklearn model already trained
            optimal_weights: list of float, weight for each model (sum(weight)==1)
    @return:
            y_pred_p: ndarray, (n_samples, n_classes), probability for each class for each sample
    """
    y_pred_p = np.zeros((X.shape[0], n_classes))

    for i_model, model in enumerate(models):
        if 'NN_batch_size' in model[1].keys():
            y_pred_p += optimal_weights[i_model] * model[1].predict(X,
                                                                    batch_size=model[1]['NN_batch_size'],
                                                                    verbose=1)
        else:
            y_pred_p += optimal_weights[i_model] * model[1].predict_proba(X)

    return y_pred_p



if __name__ == '__main__':

    # load raw string data
    data_train, y_train_all, data_test, id_test = load_data()

    print('Nb comments: {} (y_shape: {})'.format(len(data_train), y_train_all.shape))
    #print(y_train_all.sum(0)/y_train_all.shape[0])


    DATA_AUGMENT = False

    if DATA_AUGMENT:
        y_train_all_toxic_idx = np.where(np.sum(y_train_all, axis=1)!=0)[0]

        for language_extension in ['_fr', '_es', '_de']:
            print(language_extension)
            data_train_lg, _, _ ,_ = load_data(language=language_extension)
            data_train_lg_toxic = [data_train_lg[idx] for idx in y_train_all_toxic_idx]
            data_train += data_train_lg_toxic
            y_train_all = np.vstack((y_train_all, y_train_all[y_train_all_toxic_idx]))

        print('Nb comments after data augment: {} (y_shape: {})'.format(len(data_train), y_train_all.shape))
        #print(y_train_all.sum(0)/y_train_all.shape[0])

    models = []
    params_1 = {'params_clean': {'clean': False, 'lower': False, 'lemma': False, 'stop_words': False},
                'vectorizer': TokenVectorizer(max_len=200, max_features=30000),
                'split_valid_ratio': 0.1,
                'aux_features_params': {},
                'NN_batch_size': 32
              }
    models.append(('draft_embed_bidirlstm_2fc_EMB_PRETRAINED_GLOVE100t_TWITTER_TRAINABLE_TRUE', params_1))

    params_2 = {'params_clean': {'clean': False, 'lower': False, 'lemma': False, 'stop_words': False},
                'vectorizer': TokenVectorizer(max_len=100, max_features=30000),
                'split_valid_ratio': 0.05,
                'aux_features_params': {},
                'NN_batch_size': 32
               }
    models.append(('pooled-gru-fasttext_kernel_reformat', params_2))


    models, optimal_weights = model_mix(data_train, y_train_all, data_test, models)

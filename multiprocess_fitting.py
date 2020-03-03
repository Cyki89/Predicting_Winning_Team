''' need separate script to run multiprocesses on Windows'''

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from preprocessing_pipelines import ImportantFeaturesSelector


def fit_tested_classifier(estimator, params, X_train, y_train, X_val, y_val, verbose):
    ''' fit tesed model in parallel '''
    
    # set parameters for model
    model = estimator(**params)
    model_name = f'{model.__class__.__name__}{params}'
    
    # create pipeline 
    pipe = Pipeline([
    ('feature_seletion', ImportantFeaturesSelector( model, 'basic' ) ),
    ('classification', model)
    ])
    
    pipe.fit(X_train, y_train)

    score_train = accuracy_score(pipe.predict(X_train), y_train)
    score_valid = accuracy_score(pipe.predict(X_val), y_val)

    if verbose == 1:
        print(model_name)
        print(f'Accuracy score on training set: {score_train.round(4)} | Accuracy score on validation set: {score_valid.round(4)}')
        print('-'*127)
    
    return ( (model_name, pipe), score_valid )


def fit_base_classifier(estimator, X_train, y_train):
    ''' fit base model in parallel '''
    
    # fit single model or pipeline
    estimator.fit(X_train, y_train)

    return estimator


# return only one model fitted on all data
def fit_base_learner(model, X, y, folds, idx):
    ''' fit base learner in parallel '''
    
    print("Model: {} : {}".format(idx, model.steps[2][1].__class__.__name__) )
    X_blend_train = np.zeros(len(y))
    total_acc = 0
    for i, (train, val) in enumerate(folds):
            X_train = X.iloc[train]
            y_train = y[train]
            X_val = X.iloc[val]
            y_val = y[val]
            model.fit(X_train, y_train)
            pred_val = np.array(model.predict_proba(X_val))
            X_blend_train[val] = pred_val[:, 1]
            acc = accuracy_score(y_val, pred_val[:, 1].round())
            total_acc += acc
            print("{}: Fold #{}: accuracy={}".format(model.steps[2][1].__class__.__name__, i, acc) )
    print("{}: Mean accuracy={}".format(model.steps[2][1].__class__.__name__, total_acc/len(folds) ) )
    
    # return only one model fitted on all data
    model.fit(X,y)
    return model, X_blend_train


def fast_fit_base_learner(model, X, y, X_test, folds, idx):
    ''' fit base learner in parallel (fast version StackClassifier) '''
    
    print("Model: {} : {}".format(idx, model.steps[2][1].__class__.__name__) )
    X_blend_train = np.zeros(len(y))
    fold_sums = np.zeros((X_test.shape[0], len(folds)))
    total_acc = 0
    for i, (train, val) in enumerate(folds):
            X_train = X.iloc[train]
            y_train = y[train]
            X_val = X.iloc[val]
            y_val = y[val]
            model.fit(X_train, y_train)
            pred_val = np.array(model.predict_proba(X_val))
            X_blend_train[val] = pred_val[:, 1]
            pred_test = np.array(model.predict_proba(X_test))
            fold_sums[:, i] = pred_test[:, 1]
            acc = accuracy_score(y_val, pred_val[:, 1].round())
            total_acc += acc
            print("{}: Fold #{}: accuracy={}".format(model.steps[2][1].__class__.__name__,i,acc))
    print("{}: Mean accuracy={}".format(model.steps[2][1].__class__.__name__, total_acc/len(folds)))
    X_blend_test = fold_sums.mean(axis=1)
    
    return X_blend_train, X_blend_test
''' need separate script to run multiprocesses on Windows'''

import numpy as np
from sklearn.metrics import accuracy_score


# return models from each folds
def fit_base_learner(model, X, y, folds, idx):
    ''' fit base learner in parallel '''
    
    print("Model: {} : {}".format(idx, model.steps[2][1].__class__.__name__) )
    X_blend_train = np.zeros(len(y))
    total_acc = 0
    trained_models = []
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
            trained_models.append(model)
            print("{}: Fold #{}: accuracy={}".format(model.steps[2][1].__class__.__name__, i, acc) )
    print("{}: Mean accuracy={}".format(model.steps[2][1].__class__.__name__, total_acc/len(folds) ) )

    # return models from each folds
    return trained_models, X_blend_train
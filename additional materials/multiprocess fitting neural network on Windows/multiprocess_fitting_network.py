''' nessesary libraries '''
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from preprocessing_pipelines import ImportantFeaturesSelector
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import os # to disable GPU processor
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

''' some nessesary class '''

# this class change input shape of neural network after feature selection
class AnnBuilder(BaseEstimator):
    ''' build ann network in pipeline '''
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        self.model.set_params(input_shape=X.shape[1:])
        self.model.fit(X, y)
        return self.model
  
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def predict(self, X):
        return self.model.predict(X)

    
# this class change input shape of neural network after feature selection and reshape featureset to 3d array        
class RnnBuilder(BaseEstimator):
    ''' build rnn network in pipeline '''
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        # RNN need 3d array
        X_reshaped = X.reshape(*X.shape,1)
        self.model.set_params(input_shape=X_reshaped.shape[1:])
        self.model.fit(X_reshaped, y)
        return self.model
    
    def predict_proba(self, X):
        X_reshaped = X.reshape(*X.shape,1)
        return self.model.predict_proba(X_reshaped)
    
    def predict(self, X):
        X_reshaped = X.reshape(*X.shape,1)
        return self.model.predict(X_reshaped)

''' core function '''

def fit_tested_network(build_func, params, X_train, y_train, X_val, y_val, early_stopping, kind='ann', 
                       epochs=20, shuffle=False, verbose=1):
    ''' fit tesed network in parallel '''
    
    # set parameters for model
    model = KerasClassifier(build_fn=build_func, 
                            validation_split=0.1, 
                            input_shape=X_train.shape[1:],
                            epochs=epochs, 
                            shuffle=shuffle, 
                            callbacks=[early_stopping],
                            verbose=0,
                            **params)
    
    # give model name
    model_name = f'{model.__class__.__name__}{params}'
    
    if kind =='ann':
        Builder = AnnBuilder
    else:
        Builder = RnnBuilder
        
    # create pipeline
    pipe = Pipeline([
        ('feature_seletion', ImportantFeaturesSelector(model, kind) ),
        ('classifier', Builder(model) )
    ])
    
    pipe.fit(X_train, y_train)

    score_train = accuracy_score(pipe.predict(X_train), y_train)
    score_valid = accuracy_score(pipe.predict(X_val), y_val)

    if verbose == 1:
        print(model_name)
        print(f'Accuracy score on training set: {score_train.round(4)} | Accuracy score on validation set: {score_valid.round(4)}')
        print('-'*127)
     
    return ( (model_name, pipe), score_valid, params)

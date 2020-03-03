import numpy as np
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import ParameterSampler
from tensorflow.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import CuDNNLSTM, Dense, Flatten, Input
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from preprocessing_pipelines import ImportantFeaturesSelector
import multiprocess_fitting
import concurrent.futures

''' machine learning - classes and functions '''

class Metrics:
#     ''' placeholder of predition metrics '''
    def __init__(self):
        # create separate lists for each metric and name
        self.precision = []
        self.recall =[]
        self.f1 = []
        self.roc_auc = []
        self.accuracy = []
        self.names = []
        
    def add_metrics(self, model, model_name, X_test, y_test):
        # add prediction metrics and model name to the appropriate lists
        self.accuracy.append(metrics.accuracy_score(y_test, model.predict(X_test)))
        self.precision.append(metrics.precision_score(y_test, model.predict(X_test)))
        self.recall.append(metrics.recall_score(y_test , model.predict(X_test)))
        self.f1.append(metrics.f1_score(y_test , model.predict(X_test)))
        try:
            self.roc_auc.append(metrics.roc_auc_score(y_test , model.predict_proba(X_test)[:,1]))
        except IndexError: # for AveragingClassifier
            self.roc_auc.append(metrics.roc_auc_score(y_test , model.predict_proba(X_test)))
        self.names.append(model_name)
        
    def get_metrics(self):
        # return all metric lists
        return self.precision, self.recall, self.f1, self.roc_auc, self.accuracy
    
    def get_names(self):
        # return list of model names
        return self.names


def show_best_models(best_models, best_scoring):
    ''' function shows the best selected models and their scores ''' 
    for i, (model, score) in enumerate(zip(best_models, best_scoring),1):
        print(f'Place: {i}')
        print(f'{model[0]}')
        print(f'Accuracy score on validation set: {score.round(4)}')
        print('-'*127)

        
def select_best_classifiers(estimator, params_grid, n_iter, random_state, X_train, y_train, X_val, y_val, verbose=1, 
                           n_best_models=5, max_workers=3):
    ''' select n best models of one type using multiprocessing'''

    # list of dicts random generated paramters 
    params_list = list(ParameterSampler(params_grid, n_iter, random_state) )
    
    # fitting models using multiproccessing
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        
        executor_results_holder = [executor.submit(multiprocess_fitting.fit_tested_classifier, estimator, params, 
                                                   X_train, y_train, X_val, y_val, verbose) for params in params_list]
        
        results = [executor_result.result() for executor_result in 
                   concurrent.futures.as_completed(executor_results_holder)]
    
    # results is a list of tuple: ( (model_name, model), val_accuracy_score )
    models = [result[0] for result in results]
    scoring_list = [result[1] for result in results]
    
    # select best models using scoring list
    best_models_index = np.argsort(scoring_list)[::-1][:n_best_models]
    best_models = np.array(models)[best_models_index]
    best_scoring = np.array(scoring_list)[best_models_index]             
    
    return best_models, best_scoring


''' deep learning - classes and functions '''

def build_ann(n_hiden_layers, hidden_layer_size, optimizer='adam', input_shape=None):
    ''' function to build the ANN architecture '''
    
    # intialize a classifier
    classifier = Sequential()
    
    # input layer
    classifier.add( Input(shape=input_shape) )
    
    # hidden layers
    for n in range(n_hiden_layers):
        classifier.add(Dense(units=hidden_layer_size, kernel_initializer='uniform', activation = 'relu'))
        # each next hidden layer of the network will be 50% smaller
        hidden_layer_size = int(hidden_layer_size/2)
    
    # output layers
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    
    #compile model
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return classifier


def build_rnn(n_lstm_layers, lstm_layer_size, n_hiden_layers, hidden_layer_size, optimizer='adam', input_shape=None):
    ''' function to build the RNN architecture '''

    # intialize a classifier
    classifier = Sequential()
    
    # input layer
    classifier.add(Input(shape=input_shape))

    # lstm layers
    for n in range(n_lstm_layers):
        classifier.add(CuDNNLSTM(units=lstm_layer_size, return_sequences=True) )
                  
    # flatten array to 1d vector             
    classifier.add(Flatten())

    # hidden layers
    for n in range(n_hiden_layers):
        classifier.add( Dense(units=hidden_layer_size, kernel_initializer='uniform', activation = 'relu') )
    
    # output layer
    classifier.add( Dense(units=1, kernel_initializer='uniform', activation='sigmoid') )
    
    # compile model
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return classifier


class AnnBuilder(BaseEstimator):
    ''' ANN wrapper to dynamically change first layers input shape in pipeline '''
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

    
class RnnBuilder(BaseEstimator):
    ''' RNN wrapper to dynamically change first layers input shape in pipeline '''
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        # RNN need 3d array
        X_reshaped = X.reshape(X.shape[0], 1, X.shape[1]) 
        self.model.set_params(input_shape=X_reshaped.shape[1:])
        self.model.fit(X_reshaped, y)
        return self.model

    
    def predict_proba(self, X):
        X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
        return self.model.predict_proba(X_reshaped)
    
    def predict(self, X):
        X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
        return self.model.predict(X_reshaped)

    
def fit_network(build_func, params, X_train, y_train, X_val, y_val, early_stopping, kind='ann', 
                       epochs=50, shuffle=True, verbose=1):
    ''' make pipeline with feature selecion, fit it on train set and evaluate on validation set'''
      
    # shuffle training data in each iteration
    idx = np.arange(len(y_train))
    np.random.shuffle(idx)    
    
    # set parameters for model
    model = KerasClassifier(build_fn=build_func, 
                            # validation_split=0.1, 
                            validation_split=0.03,
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
    
    pipe.fit(X_train[idx], y_train[idx])
    
    score_train = accuracy_score(pipe.predict(X_train[idx]), y_train[idx])
    score_valid = accuracy_score(pipe.predict(X_val), y_val)

    if verbose == 1:
        print(model_name)
        print(f'Accuracy score on training set: {score_train.round(4)} | Accuracy score on validation set: {score_valid.round(4)}')
        print('-'*127)
     
    return ( (model_name, pipe), score_valid )

    
def select_best_networks(build_func, params_grid, n_iter, random_state, X_train, y_train, X_val, y_val, early_stopping,
                         kind='ann', epochs=50, shuffle=True, verbose=1, n_best_models=5):
    ''' select best neural networks using KerasClassifier and ParameterSampler to generate models with diffrent parameters'''
   
    # list of dicts random generated paramters 
    params_list = list(ParameterSampler(params_grid, n_iter, random_state) )
 
    results = [fit_network(build_func,  params, X_train, y_train, X_val, y_val, early_stopping, 
                           kind, epochs, shuffle) for params in params_list]

    # results is a list of tuple: ( (model_name, model), val_accuracy_score )
    models = [result[0] for result in results]
    scoring_list = [result[1] for result in results]
    
    # select best models using scoring list
    best_models_index = np.argsort(scoring_list)[::-1][:n_best_models]
    best_models = np.array(models)[best_models_index]
    best_scoring = np.array(scoring_list)[best_models_index]             
    
    return best_models, best_scoring
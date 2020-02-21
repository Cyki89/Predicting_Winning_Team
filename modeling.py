import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import ParameterSampler
from sklearn.ensemble import VotingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import CuDNNLSTM, Dense, Input
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


def make_voting_classifier(estimator, params_grid, n_iter, random_state, X_train, y_train, X_val, y_val, verbose=1, 
                           n_best_models=5, voting='soft'):
    ''' make voting classifier with n best models '''

    models = []
    scoring_list = []
    
    # list of dicts random generated paramters 
    params_list = list(ParameterSampler(params_grid, n_iter, random_state) )
    
    for params in params_list:
        
        # set parameters for model
        model = estimator(**params)
        model_name = f'{model.__class__.__name__}{params}'
        model.fit(X_train, y_train)
        
        score_train = accuracy_score(model.predict(X_train), y_train)
        score_valid = accuracy_score(model.predict(X_val), y_val)
        
        scoring_list.append(score_valid)
        models.append( (model_name, model) )
        
        if verbose == 1:
            print(model_name)
            print(f'Accuracy score on training set: {score_train.round(4)} | Accuracy score on validation set: {score_valid.round(4)}')
            print('-'*80)
    
    # select best models using scoring list
    best_models_index = np.argsort(scoring_list)[::-1][:n_best_models]
    best_models = np.array(models)[best_models_index]
    
    # make voting classifier using best models
    voting_clf = VotingClassifier(estimators=best_models, voting=voting)
    
    return voting_clf


def build_ann_classifier(n_hiden_layers, hidden_layer_size, optimizer='adam', input_shape=None):
    ''' function to build the ANN architecture '''
    
    # intialize a classifier
    classifier = Sequential()
    
    # input layer
    classifier.add( Input(shape=input_shape) )
    
    # hidden layers
    for n in range(n_hiden_layers):
        classifier.add(Dense(units=hidden_layer_size, kernel_initializer='uniform', activation = 'relu'))
        # each next hidden layer of the network will be 50% smaller
        hidden_layer_size /= hidden_layer_size
    
    # output layers
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    
    #compile model
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return classifier


def build_rnn_classifier(n_lstm_layers, lstm_layer_size, n_hiden_layers, hidden_layer_size, optimizer='adam', input_shape=None):
    ''' function to build the RNN architecture '''
    
    # intialize a classifier
    classifier = Sequential()
    
#     # input layer
#     classifier.add( Input(shape=input_shape) )
   
    # frist lstm layers
    for n in range(n_lstm_layers-1):
        classifier.add( CuDNNLSTM(units=lstm_layer_size, return_sequences=True) )
    
    # last lstm layer
    classifier.add( CuDNNLSTM(units=lstm_layer_size, return_sequences=False, input_shape=input_shape) )
    
    # hidden layers
    for n in range(n_hiden_layers):
        classifier.add( Dense(units=hidden_layer_size, kernel_initializer='uniform', activation = 'relu') )
        # each next hidden layer of the network will be 50% smaller
        hidden_layer_size /= hidden_layer_size
    
    # output layer
    classifier.add( Dense(units=1, kernel_initializer='uniform', activation='sigmoid') )
    
    # compile model
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return classifier


# this method dosn't work beacouse some bug in KerasClassifier class
def make_nn_voting_classifier(build_func, params_grid, n_iter, random_state, X_train, y_train, X_val, y_val, 
                              early_stopping, epochs=100, shuffle=False, verbose=1, n_best_models=5, voting='soft'):
    ''' make voting classifier with n best neural network models using Keras wrappers for the Scikit-Learn API'''

    models = []
    scoring_list = []
    
    # list of dicts random generated paramters 
    params_list = list(ParameterSampler(params_grid, n_iter, random_state) )
    
    for params in params_list:
        
        # set parameters for model
        model = KerasClassifier(build_fn=build_func, 
                                validation_data=(X_val, y_val), 
                                input_shape=X_train.shape[1:],
                                epochs=epochs, 
                                shuffle=shuffle, 
                                callbacks=[early_stopping],
                                verbose=0,
                                **params)
        
        model_name = f'{model.__class__.__name__}{params}'
        model.fit(X_train, y_train)
        
        score_train = accuracy_score(model.predict(X_train), y_train)
        score_valid = accuracy_score(model.predict(X_val), y_val)
        
        scoring_list.append(score_valid)
        models.append( (model_name, model) )
        
        if verbose == 1:
            print(model_name)
            print(f'Accuracy score on training set: {score_train.round(4)} | Accuracy score on validation set: {score_valid.round(4)}')
            print('-'*100)
    
    # select best models using scoring list
    best_models_index = np.argsort(scoring_list)[::-1][:n_best_models]
    best_models = np.array(models)[best_models_index]
    
    # make voting classifier using best models
    voting_clf = VotingClassifier(estimators=best_models, voting=voting)
    
    return voting_clf


def select_best_nn_classifier(build_func, params_grid, n_iter, random_state, X_train, y_train, X_val, y_val, 
                              early_stopping, epochs=100, shuffle=False, verbose=1):
    ''' select best neural network model using KerasClassifier and ParameterSampler to generate models with diffrent parameters'''

    models = []
    scoring_list = []
    
    # list of dicts random generated paramters 
    params_list = list(ParameterSampler(params_grid, n_iter, random_state) )
    
    for params in params_list:
        
        # set parameters for model
        model = KerasClassifier(build_fn=build_func, 
                                validation_data=(X_val, y_val), 
                                input_shape=X_train.shape[1:],
                                epochs=epochs, 
                                shuffle=shuffle, 
                                callbacks=[early_stopping],
                                verbose=0,
                                **params)
        
        model_name = f'{model.__class__.__name__}{params}'
        model.fit(X_train, y_train)
        
        score_train = accuracy_score(model.predict(X_train), y_train)
        score_valid = accuracy_score(model.predict(X_val), y_val)
        
        scoring_list.append(score_valid)
        models.append( (model_name, model) )
        
        if verbose == 1:
            print(model_name)
            print(f'Accuracy score on training set: {score_train.round(4)} | Accuracy score on validation set: {score_valid.round(4)}')
            print('-'*100)
    
    # select best models using scoring list
    best_model_index = np.argmax(scoring_list)
    best_model = models[best_model_index]
    
    return best_model[0], best_model[1]
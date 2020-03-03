import numpy as np
from sklearn.model_selection import ParameterSampler
import concurrent.futures
import multiprocess_fitting
from sklearn import metrics 

    
class AveragingClassifier():
    ''' class allows to train base models in parallel and makes them an average classifier '''
    def __init__(self, base_estimators, voting='soft'):
        self.base_estimators = base_estimators
        self.voting = voting
                 
    def fit(self, X_train, y_train, max_workers=3):
        # fitting base models in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            executor_models_holder = [executor.submit(multiprocess_fitting.fit_base_classifier, estimator, X_train, y_train) 
                                       for estimator in self.base_estimators]
            self.base_estimators = [executor_model.result() for executor_model in 
                                    concurrent.futures.as_completed(executor_models_holder)]
        return self

    def predict_proba(self, X_test):
        predicions = np.zeros( ( len(X_test), len(self.base_estimators) ) )
        for i, estimator in enumerate(self.base_estimators):
            if self.voting == 'soft':
                predicions[:,i] = estimator.predict_proba(X_test)[:,1]
            else:
                predicions[:,i] = estimator.predict(X_test) 
        return np.mean(predicions, axis=1)
                 
    def predict(self, X_test):
        return self.predict_proba(X_test).round()

    
class AveragingNetworkClassifier():
    ''' class make averaging classifier with neural networks '''
    def __init__(self, base_estimators, voting='soft'):
        self.base_estimators = base_estimators
        self.voting = voting
                 
    def fit(self, X_train, y_train):
        for estimator in self.base_estimators:
            estimator.fit(X_train, y_train)
        return self

    def predict_proba(self, X_test):
        predicions = np.zeros( ( len(X_test), len(self.base_estimators) ) )
        for i, estimator in enumerate(self.base_estimators):
            if self.voting == 'soft':
                predicions[:,i] = estimator.predict_proba(X_test)[:,1]
            else:
                predicions[:,i] = estimator.predict(X_test) 
        return np.mean(predicions, axis=1)
                 
    def predict(self, X_test):
        return self.predict_proba(X_test).round()

    
class LargeAveragingClassifier():
    ''' class allows takes trained AveragingClassifiers and makes them a large average classifier '''
    def __init__(self, base_estimators, voting='soft'):
        self.base_estimators = base_estimators
        self.voting = voting
                 
    def fit(self, X_train, y_train):
        # base models is already fitted
        return self

    def predict_proba(self, X_test):
        predicions = np.zeros( ( len(X_test), len(self.base_estimators) ) )
        for i, estimator in enumerate(self.base_estimators):
            if self.voting == 'soft':
                # dedicated for AveragingClassifier
                predicions[:,i] = estimator.predict_proba(X_test)
            else:
                predicions[:,i] = estimator.predict(X_test) 
        return np.mean(predicions, axis=1)
                 
    def predict(self, X_test):
        return self.predict_proba(X_test).round()


# save only one model of each type
class StackClassifier():
    ''' implemetion of  stack classifier, that allows to train base models in parallel'''
    def __init__(self, base_models, meta_model, kfold):
        self.base_models = base_models
        self.base_trained_models = None
        self.meta_model = meta_model
        self.kfold = kfold
        self.X_blend_train = None

    def fit(self, X_train, y_train, max_workers=3):
        # kfolds for cross-validation
        folds = list(self.kfold.split(X_train, y_train))
        
        # train base model in paralel
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        
            executor_results_holder = [executor.submit(multiprocess_fitting.fit_base_learner, model,
                                                       X_train, y_train, folds, idx) 
                                      for idx, model in enumerate(self.base_models)]
        
            results = [executor_result.result() for executor_result in 
                       concurrent.futures.as_completed(executor_results_holder)]
        
        # save only one model
        self.base_trained_models = [result[0] for result in results] 
        
        # to release memory
        self.base_models = None 
        
        self.X_blend_train = np.concatenate([np.array(result[1]).reshape(-1,1) for result in results], axis=1)
        
        # train meta model
        self.meta_model.fit(self.X_blend_train, y_train)
        
        return self
    
    def predict(self, X_test):
        # feature test set for meta model - single model
        X_blend_test = np.concatenate([model.predict_proba(X_test)[:,1].reshape(-1,1) 
                                       for model in self.base_trained_models], axis=1)
        
        return self.meta_model.predict(X_blend_test)

        
    def predict_proba(self, X_test):
        # feature test set for meta model - single model
        X_blend_test = np.concatenate([model.predict_proba(X_test)[:,1].reshape(-1,1) 
                                       for model in self.base_trained_models], axis=1)
        
        return self.meta_model.predict_proba(X_blend_test)
   
    
class FastStackClassifier:
    ''' fast implementation of StackClassifier only for testing '''
    def __init__(self, base_models, kfold, meta_model=None):
        self.base_models = base_models
        self.kfold = kfold
        self.meta_model = meta_model
        self.X_blend_train = None
        self.X_blend_test = None

    def fit_base_models(self, X_train, y_train, X_test, max_workers=3):
        folds = list(self.kfold.split(X_train, y_train))

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        
            executor_results_holder = [executor.submit(multiprocess_fitting.fast_fit_base_learner, model,
                                                       X_train, y_train, X_test, folds, idx) 
                                      for idx, model in enumerate(self.base_models)]
        
            results = [executor_result.result() for executor_result in 
                       concurrent.futures.as_completed(executor_results_holder)]

        self.X_blend_train = np.concatenate([np.array(result[0]).reshape(-1,1) for result in results], axis=1)
        self.X_blend_test = np.concatenate([np.array(result[1]).reshape(-1,1) for result in results], axis=1)
        
        return self

    def fit_meta_model(self, y_train):
        self.meta_model.fit(self.X_blend_train, y_train)
        return self
    
    def predict(self, X_test):
        return self.meta_model.predict(self.X_blend_test)
        
    def predict_proba(self_test):
        return self.meta_model.predict_proba(self.X_blend_test)
    
    def evaluate_meta_models(self, meta_models, y_train, y_test, metric=metrics.accuracy_score):
        best_score = 0
        best_meta_model = None
        
        for i, meta_model in enumerate(meta_models):
            meta_model.fit(self.X_blend_train, y_train)
            y_pred = meta_model.predict(self.X_blend_test)
            
            curr_score = metric(y_test, y_pred)
            if curr_score > best_score:
                best_meta_model = meta_model
                best_score = curr_score
                
            print("Model {} : {}".format(i, meta_model) )
            print(f'Accuracy score: {round(curr_score,4)}')
            print('-'*125)
            
        print('-'*125)
        print(f'Best meta model: {best_meta_model}')
        print(f'Best score: {round(best_score,4)}')
        
        return best_meta_model
import numpy as np
from sklearn.model_selection import ParameterSampler
from multiprocess_fitting_network import fit_tested_network


def select_best_networks(build_func, params_grid, n_iter, random_state, X_train, y_train, X_val, y_val, early_stopping,
                              kind='ann', max_workers=3, epochs=20, shuffle=False, verbose=1, n_best_models=5):
    ''' select best neural networks model using KerasClassifier and ParameterSampler to generate models with diffrent parameters'''
   
    # list of dicts random generated paramters 
    params_list = list(ParameterSampler(params_grid, n_iter, random_state) )
    
    # fitting models using multiproccessing
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        
        executor_results_holder = [executor.submit(multiprocess_network.fit_tested_network, build_func,  params, X_train, y_train,
                                                   X_val, y_val, early_stopping, kind, epochs, shuffle) for params in params_list]
        
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
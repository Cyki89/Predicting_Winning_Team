import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

# this method don't work on this project but can useful on other data
# fit data on train set and evaluate feature importance on validation data
def perturbation_rank(model, x, y, names, regression):
    errors = []

    for i in range(x.shape[1]):
        hold = np.array(x.iloc[:, i])
        np.random.shuffle(x.iloc[:, i])
        
        if regression:
            pred = model.predict(x)
            error = metrics.mean_squared_error(y, pred)
        else:
            pred = model.predict(x)
            error = metrics.log_loss(y, pred)
            
        errors.append(error)
        x.iloc[:, i] = hold
        
    max_error = np.max(errors)
    importance = [e/max_error for e in errors]

    data = {'name':names,'error':errors,'importance':importance}
    result = pd.DataFrame(data, columns = ['name','error','importance'])
    result.sort_values(by=['importance'], ascending=[0], inplace=True)
    result.reset_index(inplace=True, drop=True)
    return result
''' script contains preprocessing pipelines for linear and tree based models, all transformers and helper function using in pipelines ''' 

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, log_loss

''' helper functions '''

def feature_reduction_pipeline(model, X, y):
    ''' reduce number of features using perturbation techinque'''

    model.fit(X,y)
    
    base_acc = accuracy_score(y, model.predict(X))
    base_log_loss= log_loss( y, model.predict_proba(X)[:,1] )
    
    best_features_idx = []
    
    for i in range(X.shape[1]):

        hold = X.copy()
        np.random.shuffle(X[:, i])

        curr_acc = accuracy_score( y, model.predict(X) )
        diff_acc = curr_acc - base_acc

        curr_log_loss = log_loss( y, model.predict_proba(X)[:,1] )
        diff_log_loss = curr_log_loss - base_log_loss
        
        if diff_log_loss > 0: # if diff_acc < 0 and diff_log_loss > 0:
            best_features_idx.append(i)

        X = hold
        
    if not best_features_idx:
        best_features_idx = list(range(X.shape[1]))
        
    return np.array(best_features_idx)


def feature_reduction_ann_pipeline(model, X, y):
    ''' reduce number of features for ann using perturbation techinque'''
    
    model.set_params(input_shape=X.shape[1:])
    model.fit(X,y)
    
    base_acc = accuracy_score(y, model.predict(X))
    base_log_loss = log_loss( y, model.predict_proba(X)[:,1] )
    
    best_features_idx = []
    
    for i in range(X.shape[1]):

        hold = X.copy()
        np.random.shuffle(X[:, i])

        curr_acc = accuracy_score( y, model.predict(X) )
        diff_acc = curr_acc - base_acc

        curr_log_loss = log_loss( y, model.predict_proba(X)[:,1] )
        diff_log_loss = curr_log_loss - base_log_loss

        if diff_log_loss > 0: # if diff_acc < 0 and diff_log_loss > 0:
            best_features_idx.append(i)

        X = hold
    
    if not best_features_idx:
        best_features_idx = list(range(X.shape[1]))
        
    return np.array(best_features_idx)


def feature_reduction_rnn_pipeline(model, X, y):
    ''' reduce number of features for rnn using perturbation techinque'''

    X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
    
    model.set_params(input_shape=X_reshaped.shape[1:])
    model.fit(X_reshaped, y)
    
    base_acc = accuracy_score(y, model.predict(X_reshaped))
    base_log_loss = log_loss( y, model.predict_proba(X_reshaped)[:,1] )
    best_features_idx = []
    
    for i in range(X.shape[1]):

        hold = X_reshaped.copy()
        np.random.shuffle(X_reshaped[:, :, i])

        curr_acc = accuracy_score( y, model.predict(X_reshaped) )
        diff_acc = curr_acc - base_acc
        curr_log_loss = log_loss( y, model.predict_proba(X_reshaped)[:,1] ) 
        diff_log_loss = curr_log_loss - base_log_loss

        if diff_log_loss > 0: # if diff_acc < 0 and diff_log_loss > 0:
            best_features_idx.append(i)

        X_reshaped = hold
    
    if not best_features_idx:
        best_features_idx = list(range(X.shape[1]))
    
    return np.array(best_features_idx)

    
def target_mean_encoding(df, cat_name, target, weight=10):
    ''' function return smoothing target mean encoding '''

    # Compute the global mean
    mean = df[target].mean()

    # Compute the number of values and the mean of each group
    agg = df.groupby(cat_name)[target].agg(['count', 'mean'])

    counts = agg['count']
    means = agg['mean']

    # Compute the "smoothed" means
    smooth = (counts * means + weight * mean) / (counts + weight)

    return smooth, mean


''' pipeline transformers '''

class DataFrameSelector(BaseEstimator, TransformerMixin):
    ''' select columns from dataframe and return numpy array '''
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return np.array(X[self.attribute_names])


class TwoColumnScaler(BaseEstimator, TransformerMixin):
    ''' take two columns and scaling it's keeping original ratio between them '''
    def __init__(self, scaler):
        self.scaler = scaler
        
    def fit(self, X, y=None):
        columns_merged = np.concatenate((X[:,0], X[:,1]), axis=0)
        self.scaler.fit(columns_merged.reshape(-1,1))
        return self
    
    def transform(self, X, y=None):
        X1 = self.scaler.transform(X[:, 0].reshape(-1,1))
        X2 = self.scaler.transform(X[:, 1].reshape(-1,1))
        X_new = np.concatenate((X1, X2), axis=1)
        return X_new

    
class DictionaryEncoder(BaseEstimator, TransformerMixin):
    ''' encoding labels using dictionary '''
    def __init__(self, dictionary):
        self.dictionary = dictionary
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.replace(self.dictionary).values
    

class ToDataFrame(BaseEstimator, TransformerMixin):
    ''' transform numpy array to dataframe '''
    def __init__(self, columns):
        self.columns = columns
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return pd.DataFrame(X, columns=self.columns)

    
class Array3dTransformer(BaseEstimator, TransformerMixin):
    ''' transform 2d numpy array to 3d numpy array '''
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.reshape(*X.shape,1)

    
class ImportantFeaturesSelector(BaseEstimator, TransformerMixin):
    ''' select most important features from numpy array'''
    def __init__(self, model, model_type):
        self.model = model
        self.model_type = model_type
        
    def fit(self, X, y=None):
        if self.model_type == 'basic':
            self.important_features = feature_reduction_pipeline(self.model, X, y)
        elif self.model_type == 'ann':
            self.important_features = feature_reduction_ann_pipeline(self.model, X, y)
        elif self.model_type == 'rnn':
            self.important_features = feature_reduction_rnn_pipeline(self.model, X, y)
        else:
            raise TypeError('model_type have to be basic, ann or rnn')
        return self
    
    def transform(self, X, y=None):
        return X[:, self.important_features]
    

class TargetMeanEncodingTransformer(BaseEstimator, TransformerMixin):
    ''' transform feature using target mean encoding'''
    def __init__(self, cat_name, target):
        self.cat_name = cat_name
        self.target = target
        
    def fit(self, X, y=None):
        self.target_dict, self.global_mean = target_mean_encoding(X, self.cat_name, self.target)
        return self

    def transform(self, X, y=None):
        X_arr = np.zeros(len(X)).reshape(-1,1)
        for i in range(len(X_arr)):
            try:
                X_arr[i] = self.target_dict.loc[ X[self.cat_name].iloc[i] ]
            except KeyError: # category doesnt occur in training set
                X_arr[i] = self.global_mean
        return X_arr
    
    
''' basic pipelines '''     

# read raw data
X_train_set = pd.read_csv('./preprocessed_data/train_set_stage2.csv', index_col=0)

# create list of team names for ordinal encoder
home_team_names = np.unique(X_train_set['HomeTeam'])
away_team_names = np.unique(X_train_set['AwayTeam'])
team_names=[home_team_names, away_team_names]

# assign manually features to the groups
target_col = ['FTR']

teams_cols =['HomeTeam','AwayTeam']

teams_ratio_cols = ['HomeTeamWinRatio', 'AwayTeamWinRatio']

teams_ratio_cat_cols = ['HomeTeamWinRatio_Cat', 'AwayTeamWinRatio_Cat']

last_year_postion_cols = ['HomeTeamLastYearPosition', 'AwayTeamLastYearPosition']

total_cols = ['HomeTeamGoalsScored','AwayTeamGoalsScored','HomeTeamGoalsLost','AwayTeamGoalsLost','HomeTeamShootsMade', 
              'AwayTeamShootsMade','HomeTeamTargetShootsMade','AwayTeamTargetShootsMade','HomeTeamCorners','AwayTeamCorners',
              'HomeTeamTotalPoints','AwayTeamTotalPoints']

total_cat_cols = ['HomeTeamTargetShootsMade_Cat', 'AwayTeamTargetShootsMade_Cat', 'HomeTeamGoalsScored_Cat',
                  'AwayTeamGoalsScored_Cat', 'HomeTeamGoalsLost_Cat','AwayTeamGoalsLost_Cat', 'HomeTeamShootsMade_Cat',
                  'AwayTeamShootsMade_Cat','HomeTeamCorners_Cat', 'AwayTeamCorners_Cat', 'HomeTeamTotalPoints_Cat',
                  'AwayTeamTotalPoints_Cat',]

last_matches_results_cols = ['HomeTeamLast1Match','AwayTeamLast1Match', 'HomeTeamLast2Match', 'AwayTeamLast2Match',
                             'HomeTeamLast3Match', 'AwayTeamLast3Match', 'HomeTeamLast4Match','AwayTeamLast4Match', 
                             'HomeTeamLast5Match', 'AwayTeamLast5Match',]

last_matches_points_cols = ['HomeTeamPointsFromLast3Matches','AwayTeamPointsFromLast3Matches', 
                            'HomeTeamPointsFromLast5Matches','AwayTeamPointsFromLast5Matches', 
                            'HomeTeamPointsFromLast10Matches','AwayTeamPointsFromLast10Matches']

binary_cols = ['HomeTeamWinStreak3', 'HomeTeamWinStreak5', 'HomeTeamLossStreak3','HomeTeamLossStreak5', 
               'AwayTeamWinStreak3', 'AwayTeamWinStreak5','AwayTeamLossStreak3', 'AwayTeamLossStreak5',
               'IsHomeTeamRegulars', 'IsAwayTeamRegulars', 'IsHomeTeamRookie', 'IsAwayTeamRookie']

diff_cols = ['HomeTeamGoalsDifference', 'AwayTeamGoalsDifference','TotalGoalsDifference','DifferenceTotalPoints',
             'Difference1MatchPoints', 'Difference3MatchesPoints','Difference5MatchesPoints','Difference10MatchesPoints',
             'DifferenceInShoots', 'DifferenceInTargetShoots', 'DifferenceInCorners','DifferenceInLastYearPosition'] 

diff_cat_cols = ['HomeTeamGoalsDifference_Cat','AwayTeamGoalsDifference_Cat', 'TotalGoalsDifference_Cat',
                 'DifferenceTotalPoints_Cat', 'Difference10MatchesPoints_Cat','DifferenceInShoots_Cat',
                 'DifferenceInTargetShoots_Cat','DifferenceInCorners_Cat']


''' Base pipeline for tree-based models '''

standard_scaling_base_pipeline = Pipeline([
    ('select_cols', DataFrameSelector([*binary_cols, *teams_ratio_cols, *last_matches_points_cols, 
                                       *last_matches_results_cols, *last_year_postion_cols, *diff_cols]) ),
    ('standard_scaler', StandardScaler() )
])

# label enocoding team names
ordinal_encoder_pipeline = Pipeline([
    ('select_cols', DataFrameSelector([*teams_cols]) ),
    ('ordinal_encoder', OrdinalEncoder(categories=team_names) ),
    ('standard_scaler', StandardScaler() )
])

# process two features to the same scale(leaving dependencies between them)
goals_scored_pipeline = Pipeline([
    ('select_cols', DataFrameSelector([total_cols[0], total_cols[1]]) ),
    ('two_column_scaler', TwoColumnScaler(scaler=StandardScaler() ))
])

goals_lost_pipeline = Pipeline([
    ('select_cols', DataFrameSelector([total_cols[2], total_cols[3]]) ),
    ('two_column_scaler', TwoColumnScaler(scaler=StandardScaler() ))
])

shoot_made_pipeline = Pipeline([
    ('select_cols', DataFrameSelector([total_cols[4], total_cols[5]]) ),
    ('two_column_scaler', TwoColumnScaler(scaler=StandardScaler() ))
])

total_shoot_made_pipeline = Pipeline([
    ('select_cols', DataFrameSelector([total_cols[6], total_cols[7]]) ),
    ('two_column_scaler', TwoColumnScaler(scaler=StandardScaler() ))
])

corners_pipeline = Pipeline([
    ('select_cols', DataFrameSelector([total_cols[8], total_cols[9]]) ),
    ('two_column_scaler', TwoColumnScaler(scaler=StandardScaler() ))
])

total_points_pipeline = Pipeline([
    ('select_cols', DataFrameSelector([total_cols[10], total_cols[11]]) ),
    ('two_column_scaler', TwoColumnScaler(scaler=StandardScaler() ))
])

basic_preprocess_pipeline = FeatureUnion(transformer_list=[
                                        ('standard_scaling_pipeline', standard_scaling_base_pipeline),
                                        ('ordinal_encoder_pipeline', ordinal_encoder_pipeline),
                                        ('goals_scored_pipeline', goals_scored_pipeline),
                                        ('goals_lost_pipeline', goals_lost_pipeline),
                                        ('shoot_made_pipeline', shoot_made_pipeline),
                                        ('total_shoot_made_pipeline', total_shoot_made_pipeline),
                                        ('corners_pipeline', corners_pipeline),
                                        ('total_points_pipeline', total_points_pipeline),
])


''' Pipeline for linear models '''

base_cat_pipeline = Pipeline([
    ('select_cols', DataFrameSelector([*binary_cols]) ),
    ('standard_scaler', StandardScaler() )
])

home_team_encoding_pipeline = Pipeline([
    ('encoding', TargetMeanEncodingTransformer(teams_cols[0], *target_col) ),
    ('standard_scaler', StandardScaler() )
])

away_team_encoding_pipeline = Pipeline([
    ('encoding', TargetMeanEncodingTransformer(teams_cols[1], *target_col) ),
    ('standard_scaler', StandardScaler() )
])

standard_scaling_cat_pipeline = Pipeline([
    ('select_cols', DataFrameSelector([*teams_ratio_cat_cols, *last_matches_points_cols, *last_matches_results_cols,
                                       *last_year_postion_cols, *diff_cat_cols, *total_cat_cols]) ),
    ('standard_scaler', StandardScaler() )
])

categorical_preprocess_pipeline = FeatureUnion(transformer_list=[
                                              ('home_teams_encoding', home_team_encoding_pipeline),
                                              ('away_teams_encoding', away_team_encoding_pipeline),
                                              ('base_pipeline ', base_cat_pipeline),
                                              ('standard_scaling_pipeline', standard_scaling_cat_pipeline),
])
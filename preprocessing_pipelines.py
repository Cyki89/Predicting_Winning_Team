''' script contains preprocessing pipelines for linear and tree based models ''' 

from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from data_preprocessing import *

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

X_train = pd.read_csv('./preprocessed_data/train_set_stage2.csv', index_col=0)
X_test = pd.read_csv('./preprocessed_data/test_set_stage2.csv', index_col=0) 

home_team_names = np.unique(X_train['HomeTeam'])
away_team_names = np.unique(X_train['AwayTeam'])
team_names=[home_team_names, away_team_names]


''' PipeLine for tree-based models '''

# all transformers from data_preprocessing.py
base_pipeline = Pipeline([
    ('select_cols', DataFrameSelector([* binary_cols, *teams_ratio_cols]) ),
])

minmax_scaling_pipeline = Pipeline([
    ('select_cols', DataFrameSelector([*last_matches_points_cols, *last_matches_results_cols, *last_year_postion_cols]) ),
    ('minmax_scaler', MinMaxScaler() )
])

standard_scaling_pipeline = Pipeline([
    ('select_cols', DataFrameSelector([*diff_cols]) ),
    ('standard_scaler', StandardScaler() )
])

# label enocoding team names
ordinal_encoder_pipeline = Pipeline([
    ('select_cols', DataFrameSelector([*teams_cols]) ),
    ('ordinal_encoder', OrdinalEncoder(categories=team_names) ),
    ('minmax_scaler', MinMaxScaler() )
])

# process two features to the same scale(leaving dependencies between them)
goals_scored_pipeline = Pipeline([
    ('select_cols', DataFrameSelector([total_cols[0], total_cols[1]]) ),
    ('two_column_scaler', TwoColumnScaler(scaler=MinMaxScaler() ))
])

goals_lost_pipeline = Pipeline([
    ('select_cols', DataFrameSelector([total_cols[2], total_cols[3]]) ),
    ('two_column_scaler', TwoColumnScaler(scaler=MinMaxScaler() ))
])

shoot_made_pipeline = Pipeline([
    ('select_cols', DataFrameSelector([total_cols[4], total_cols[5]]) ),
    ('two_column_scaler', TwoColumnScaler(scaler=MinMaxScaler() ))
])

total_shoot_made_pipeline = Pipeline([
    ('select_cols', DataFrameSelector([total_cols[6], total_cols[7]]) ),
    ('two_column_scaler', TwoColumnScaler(scaler=MinMaxScaler() ))
])

corners_pipeline = Pipeline([
    ('select_cols', DataFrameSelector([total_cols[8], total_cols[9]]) ),
    ('two_column_scaler', TwoColumnScaler(scaler=MinMaxScaler() ))
])

total_points_pipeline = Pipeline([
    ('select_cols', DataFrameSelector([total_cols[10], total_cols[11]]) ),
    ('two_column_scaler', TwoColumnScaler(scaler=MinMaxScaler() ))
])

tree_preprocess_pipeline = FeatureUnion(transformer_list=[
                                    ('base_pipeline ', base_pipeline),
                                    ('minmax_scaling_pipeline', minmax_scaling_pipeline),
                                    ('standard_scaling_pipeline', standard_scaling_pipeline),
                                    ('ordinal_encoder_pipeline', ordinal_encoder_pipeline),
                                    ('goals_scored_pipeline', goals_scored_pipeline),
                                    ('goals_lost_pipeline', goals_lost_pipeline),
                                    ('shoot_made_pipeline', shoot_made_pipeline),
                                    ('total_shoot_made_pipeline', total_shoot_made_pipeline),
                                    ('corners_pipeline', corners_pipeline),
                                    ('total_points_pipeline', total_points_pipeline),
])


''' PipeLine for linear models '''

# all transformers from data_preprocessing.py
base_cat_pipeline = Pipeline([
    ('select_cols', DataFrameSelector([*binary_cols]) ),
])

home_team_encoding_pipeline = Pipeline([
    ('encoding', TargetMeanEncodingTransformer(X_train, teams_cols[0], *target_col) ),
])

away_team_encoding_pipeline = Pipeline([
    ('encoding', TargetMeanEncodingTransformer(X_train, teams_cols[1], *target_col) ),
])

minmax_scaling_pipeline = Pipeline([
    ('select_cols', DataFrameSelector([*teams_ratio_cat_cols, *last_matches_points_cols, *last_matches_results_cols,
                                       *last_year_postion_cols, *diff_cat_cols, *total_cat_cols]) ),
    ('minmax_scaler', MinMaxScaler() )
])

linear_preprocess_pipeline = FeatureUnion(transformer_list=[
                                         ('home_teams_encoding', home_team_encoding_pipeline),
                                         ('away_teams_encoding', away_team_encoding_pipeline),
                                         ('base_pipeline ', base_cat_pipeline),
                                         ('minmax_scaling_pipeline', minmax_scaling_pipeline),
])
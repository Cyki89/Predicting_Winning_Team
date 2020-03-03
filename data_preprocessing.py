import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import ParameterSampler

                    
''' data preprocessing - stage 1 '''

# some preprocessing functions in stage 1 have been converted or directly useds from 
# https://github.com/RudrakshTuwani/Football-Data-Analysis-and-Prediction
def read_datasets(dir_path, usecols, datacols):
    ''' read each datasets from dir and put everything on the list '''
    datasets = []
    for filename in os.listdir(dir_path):
        datasets.append(pd.read_csv(os.path.join(dir_path,filename), 
        				parse_dates=datacols, usecols=usecols))
    return datasets


def encoding_labels(string):
    ''' encodning string labels '''
    if string == 'W':
        return 3
    if string == 'D':
        return 1
    else:
        return 0
    

def encoding_binary_labels(string):
    ''' encodning string labels '''
    if string == 'H':
        return 1
    else:
        return 0


def binary_labels(dataset, target):
    ''' transform target labels to binary labels '''
    dataset[target] = dataset[target].apply(encoding_binary_labels)
    return dataset
 

def get_win_ratio(dataset):
    ''' calculate win ratio for home and away teams and put them in new columns '''
    ht = np.zeros(len(dataset))
    at = np.zeros(len(dataset))
    for i in range(20, len(dataset)): # skip first two round
        home_win_ratio = dataset.iloc[:i].groupby('HomeTeam').FTR.mean()
        away_win_ratio = dataset.iloc[:i].groupby('AwayTeam').FTR.mean()  
        ht[i] = dataset.iloc[i:i+1]['HomeTeam'].map(home_win_ratio).values[0]
        at[i] = dataset.iloc[i:i+1]['AwayTeam'].map(away_win_ratio).values[0]
    dataset['HomeTeamWinRatio'] = ht
    dataset['AwayTeamWinRatio'] = at
    return dataset


def get_statistic(dataset, kind='scored', h_col=None, a_col=None, rounds=38):
    ''' get aggregate statistic arranged by teams and matchweek '''
    if kind not in ['scored', 'lost']:
        raise ValueError('kind should be scored or lost') 
    # create a dictionary with team names as keys
    teams = {}
    for i in np.unique(dataset['HomeTeam']):
        teams[i] = []
    # assign the appropriate column names to the variables to calculate the statistic scored / lost
    if kind == 'scored':
        home = h_col
        away = a_col
    else:
        home = a_col
        away = h_col
    # the value corresponding to keys is a list containing the match location.
    for i in range(len(dataset)):
        HTG = dataset.iloc[i][home]
        ATG = dataset.iloc[i][away]
        teams[dataset.iloc[i].HomeTeam].append(HTG)
        teams[dataset.iloc[i].AwayTeam].append(ATG)
    # create a dataframe for statistics scored/lost where rows are teams and cols are matchweek.
    statistics = pd.DataFrame(data=teams, index=[i for i in range(1, rounds+1)]).T
    statistics[0] = 0
    # agregate statistics to get uptil that point
    for i in range(2, rounds+1):
        statistics[i] = statistics[i] + statistics[i-1]
    return statistics


def get_multiple_statistics(dataset, rounds=38, num_matches=10):
    ''' caclulate multiple statistic and pass them to DataFrame '''

	# goals scored
    GS = get_statistic(dataset, kind='scored', h_col='FTHG', a_col='FTAG', rounds=rounds)
    # goals lost
    GL = get_statistic(dataset, kind='lost', h_col='FTHG', a_col='FTAG', rounds=rounds)
    # all shoots made
    SM = get_statistic(dataset, kind='scored', h_col='HS', a_col='AS', rounds=rounds) 
    # target shoots made
    TSM = get_statistic(dataset, kind='scored', h_col='HST', a_col='AST', rounds=rounds)
    # numbers of corners
    NC = get_statistic(dataset, kind='scored', h_col='HC', a_col='AC', rounds=rounds) 
    
    # set initial column
    j = 0 
    
    # create empty list for each statistics
    HTGS, ATGS, HTGL, ATGL, HSM, ASM, HTSM, ATSM, HNC, ANC   = [], [], [], [], [], [], [], [], [], []
    
    # add each statistics to lists
    for i in range(num_matches*rounds): 
        ht = dataset.iloc[i].HomeTeam
        at = dataset.iloc[i].AwayTeam
        # goals scored lists
        HTGS.append(GS.loc[ht][j])
        ATGS.append(GS.loc[at][j])
        # goals lost lists
        HTGL.append(GL.loc[ht][j])
        ATGL.append(GL.loc[at][j])
        # shot made lists
        HSM.append(SM.loc[ht][j])
        ASM.append(SM.loc[at][j])
        # target shot made lists
        HTSM.append(TSM.loc[ht][j])
        ATSM.append(TSM.loc[at][j])
        # corners list
        HNC.append(NC.loc[ht][j])
        ANC.append(NC.loc[at][j])
        # change col to another round
        if ((i + 1)% num_matches) == 0: 
            j = j + 1

    # create new columns for each statistic
    dataset['HomeTeamGoalsScored'] = HTGS 
    dataset['AwayTeamGoalsScored'] = ATGS
    
    dataset['HomeTeamGoalsLost'] = HTGL
    dataset['AwayTeamGoalsLost'] = ATGL
    
    dataset['HomeTeamShootsMade'] = HSM
    dataset['AwayTeamShootsMade'] = ASM
    
    dataset['HomeTeamTargetShootsMade'] = HTSM
    dataset['AwayTeamTargetShootsMade'] = ATSM
    
    dataset['HomeTeamCorners'] = HNC
    dataset['AwayTeamCorners'] = ANC

    return dataset


def get_cumulative_points(matches, rounds=38, num_matches=10):
    ''' calculate cumulative points '''
    matches_points = matches.applymap(encoding_labels)
    for i in range(2,rounds+1):
        matches_points[i] = matches_points[i] + matches_points[i-1]
    matches_points.insert(column =0, loc = 0, value = [0*i for i in range(num_matches*2)])
    return matches_points


def get_matches(dataset, rounds=38):
    ''' create dataframe with W(win) and L(lose) labels for futhure use'''
    teams = {} # create a dictionary with team names as keys
    for i in np.unique(dataset['HomeTeam']):
        teams[i] = []
    # the value corresponding to keys is a list containing the match result
    for i in range(len(dataset)):
        if dataset.iloc[i].FTR == 'H':
            teams[dataset.iloc[i].HomeTeam].append('W')
            teams[dataset.iloc[i].AwayTeam].append('L')
        elif dataset.iloc[i].FTR == 'A':
            teams[dataset.iloc[i].AwayTeam].append('W')
            teams[dataset.iloc[i].HomeTeam].append('L')
        else:
            teams[dataset.iloc[i].AwayTeam].append('D')
            teams[dataset.iloc[i].HomeTeam].append('D')
    return pd.DataFrame(data=teams, index = [i for i in range(1, rounds+1)]).T


def get_aggregate_points(dataset, rounds=38, num_matches=10):
    ''' caclulate aggregate points and pass them to DataFrame '''
    matches = get_matches(dataset, rounds)
    cum_pts = get_cumulative_points(matches, rounds, num_matches)
    HTP, ATP = [], []
    # set initial column
    j = 0
    for i in range(rounds*num_matches):
        ht = dataset.iloc[i].HomeTeam
        at = dataset.iloc[i].AwayTeam
        HTP.append(cum_pts.loc[ht][j])
        ATP.append(cum_pts.loc[at][j])
        # change col to another round
        if ((i + 1)% num_matches) == 0:
            j = j + 1    
    dataset['HomeTeamTotalPoints'] = HTP 
    dataset['AwayTeamTotalPoints'] = ATP 
    return dataset


def calculate_last_points(matches_points, num, rounds=38, num_matches=10):
    # calculate last points from last n matches
    last_points = matches_points.copy()
    last_points.insert(column=0, loc = 0, value = [0*i for i in range(num_matches*2)])
    for i in range(1,rounds+1):
        # caclulate starting index
        idx = i-num if i-num >= 0 else 0
        last_points[i] = np.sum(matches_points.iloc[:, idx:i], axis=1)
    return last_points
    

def add_points_from_n_last_matches(dataset, n, rounds=38, num_matches=10):
    ''' add points from last n matches '''
    matches_points = get_matches(dataset, rounds).applymap(encoding_labels)
    last_points = calculate_last_points(matches_points, n, rounds, num_matches) 
    # skip fisrst round
    h = [0 for i in range(num_matches)]  
    a = [0 for i in range(num_matches)]
    # set initial col
    j = 1
    for i in range(num_matches, rounds*num_matches):
        ht = dataset.iloc[i].HomeTeam
        at = dataset.iloc[i].AwayTeam
        # get last n results for home team
        last_home = last_points.loc[ht][j]  
        h.append(last_home)   
        # get last n results for away team
        last_away = last_points.loc[at][j]              
        a.append(last_away) 
        # change to another col
        if ((i + 1)% num_matches) == 0:
            j = j + 1
    # add new columns to dataframe
    dataset[f'HomeTeamPointsFromLast{str(n)}Matches'] = h                 
    dataset[f'AwayTeamPointsFromLast{str(n)}Matches'] = a
    return dataset


def add_points_from_last_matches(dataset, rounds=38, num_matches=10):
    ''' add last matches points from several intervals '''
    # points scored in last game
    new_dataset = add_points_from_n_last_matches(dataset, 1, rounds, num_matches) 
    # points scored in last three game
    new_dataset = add_points_from_n_last_matches(dataset, 3, rounds, num_matches)
    # points scored in last five game
    new_dataset = add_points_from_n_last_matches(dataset, 5, rounds, num_matches)
    # points scored in last ten game
    new_dataset = add_points_from_n_last_matches(dataset, 10, rounds, num_matches)
    return new_dataset 


def get_form(dataset, num, rounds=38):
    ''' create helper dataframe with last matches results '''
    form = get_matches(dataset, rounds)
    form_final = form.copy()
    for i in range(num, rounds+1):
        form_final[i] = ''
        j = 0
        while j < num:
            form_final[i] += form[i-j]
            j += 1           
    return form_final


def add_last_n_matches_results(dataset, num, num_matches=10, rounds=38):
    ''' add last n matches results '''
    form = get_form(dataset, num, rounds)
    # mark first n unknown for each team with 'M'
    h = ['M' for i in range(num * num_matches)]  
    a = ['M' for i in range(num * num_matches)]
    # set initial columns
    j = num
    for i in range((num*num_matches), num_matches*rounds):
        ht = dataset.iloc[i].HomeTeam
        at = dataset.iloc[i].AwayTeam
       	# get past n results for home team
        past = form.loc[ht][j]	
        h.append(past[num-1])
        # get past n results for away team
        past = form.loc[at][j]  # get past n results.
        a.append(past[num-1])
        # change column
        if ((i + 1)% num_matches) == 0:
            j = j + 1
    # add new columns to dataframe
    dataset[f'HomeTeamLast{str(num)}Match'] = h[:dataset.shape[0]]                 
    dataset[f'AwayTeamLast{str(num)}Match'] = a[:dataset.shape[0]]
    return dataset


def add_last_matches_results(dataset, num_matches=10, rounds=38):
    ''' add last 5 matches results'''
    for n in range(1,6):   
        dataset = add_last_n_matches_results(dataset, num=n, num_matches=num_matches, rounds=rounds)
    return dataset


def get_form_points(string):
    ''' encode string to points '''
    total = 0
    for letter in string:
        total += encoding_labels(letter)
    return total


def get_5game_form(dataset):
    ''' add points from last 5 games to dataframe '''
    str_cols = ['HomeTeamFormPtsStr', 'AwayTeamFormPtsStr']
    num_cols = ['HomeTeamFormPts', 'AwayTeamFormPts']
    prefixes = ['HomeTeamLast', 'AwayTeamLast']
    n_num_matches = 5
    for str_col, num_col, pref in zip(str_cols, num_cols, prefixes):
        dataset[str_col] = ''
        for n in range(1,n_num_matches+1):
            dataset[str_col] += dataset[f'{pref}{n}Match']
        dataset[num_col] = dataset[str_col].apply(get_form_points)
    return dataset


# identify win/lose streaks if any.
def get_3game_ws(string):
    if string[-3:] == 'WWW':
        return 1
    else:
        return 0
    
def get_5game_ws(string):
    if string == 'WWWWW':
        return 1
    else:
        return 0
    
def get_3game_ls(string):
    if string[-3:] == 'LLL':
        return 1
    else:
        return 0
  
def get_5game_ls(string):
    if string == 'LLLLL':
        return 1
    else:
        return 0

   
def get_games_streaks(dataset):
    ''' add win/lose streaks ''' 
    dataset = get_5game_form(dataset)
    # add streaks for home teams
    dataset['HomeTeamWinStreak3'] = dataset['HomeTeamFormPtsStr'].apply(get_3game_ws)
    dataset['HomeTeamWinStreak5'] = dataset['HomeTeamFormPtsStr'].apply(get_5game_ws)
    dataset['HomeTeamLossStreak3'] = dataset['HomeTeamFormPtsStr'].apply(get_3game_ls)
    dataset['HomeTeamLossStreak5'] = dataset['HomeTeamFormPtsStr'].apply(get_5game_ls)
    # add streaks for away teams
    dataset['AwayTeamWinStreak3'] = dataset['AwayTeamFormPtsStr'].apply(get_3game_ws)
    dataset['AwayTeamWinStreak5'] = dataset['AwayTeamFormPtsStr'].apply(get_5game_ws)
    dataset['AwayTeamLossStreak3'] = dataset['AwayTeamFormPtsStr'].apply(get_3game_ls)
    dataset['AwayTeamLossStreak5'] = dataset['AwayTeamFormPtsStr'].apply(get_5game_ls)
    return dataset


def encode_last_results(dataset, n=5):
    ''' encode labels from all last result columns '''
    for i in range (1, n+1):
        dataset[f'HomeTeamLast{str(i)}Match'] = dataset[f'HomeTeamLast{str(i)}Match'].apply(encoding_labels)               
        dataset[f'AwayTeamLast{str(i)}Match'] = dataset[f'AwayTeamLast{str(i)}Match'].apply(encoding_labels)
    return dataset


def get_last_year_position(dataset, Standings, year, num_matches=10, rounds=38):
    ''' add last year position for each team'''
    HomeTeamLP = []
    AwayTeamLP = []
    for i in range(num_matches*rounds):
        ht = dataset.iloc[i].HomeTeam
        at = dataset.iloc[i].AwayTeam
        HomeTeamLP.append(Standings.loc[ht][year])
        AwayTeamLP.append(Standings.loc[at][year])
    dataset['HomeTeamLastYearPosition'] = HomeTeamLP
    dataset['AwayTeamLastYearPosition'] = AwayTeamLP
    return dataset


def is_rookie(dataset):
    ''' add information whether the team is rookie in current season'''
    dataset['IsHomeTeamRookie'] = (dataset['HomeTeamLastYearPosition'] == 18).astype('int')
    dataset['IsAwayTeamRookie'] = (dataset['AwayTeamLastYearPosition'] == 18).astype('int')
    return dataset

def is_regular(dataset, regulars):
    ''' add information whether the team played in the Premier League every season '''
    dataset['IsHomeTeamRegulars'] = (dataset['HomeTeam'].isin(regulars)).astype('int')
    dataset['IsAwayTeamRegulars'] = (dataset['AwayTeam'].isin(regulars)).astype('int')
    return dataset


def get_difference(dataset):
    ''' add some statistic difference '''
    
    # difference in goals
    dataset['HomeTeamGoalsDifference'] = dataset['HomeTeamGoalsScored'] - dataset['HomeTeamGoalsLost']
    dataset['AwayTeamGoalsDifference'] = dataset['AwayTeamGoalsScored'] - dataset['AwayTeamGoalsLost']
    dataset['TotalGoalsDifference'] = dataset['HomeTeamGoalsDifference'] - dataset['AwayTeamGoalsDifference']

    # difference in points
    dataset['DifferenceTotalPoints'] = dataset['HomeTeamTotalPoints'] - dataset['AwayTeamTotalPoints']
    dataset['Difference1MatchPoints'] = dataset['HomeTeamPointsFromLast1Matches'] - dataset['AwayTeamPointsFromLast1Matches']
    dataset['Difference3MatchesPoints'] = dataset['HomeTeamPointsFromLast3Matches'] - dataset['AwayTeamPointsFromLast3Matches']
    dataset['Difference5MatchesPoints'] = dataset['HomeTeamPointsFromLast5Matches'] - dataset['AwayTeamPointsFromLast5Matches']
    dataset['Difference10MatchesPoints'] = dataset['HomeTeamPointsFromLast10Matches'] - dataset['AwayTeamPointsFromLast10Matches']
    
    # difference in shoots
    dataset['DifferenceInShoots'] = dataset['HomeTeamShootsMade'] - dataset['AwayTeamShootsMade']
    dataset['DifferenceInTargetShoots'] = dataset['HomeTeamTargetShootsMade'] - dataset['AwayTeamTargetShootsMade']
    
    # difference in corners
    dataset['DifferenceInCorners'] = dataset['HomeTeamCorners'] - dataset['AwayTeamCorners']
    
    # difference in last year positions
    dataset['DifferenceInLastYearPosition'] = dataset['HomeTeamLastYearPosition'] - dataset['AwayTeamLastYearPosition']

    return dataset


def drop_first_rounds(dataset, rounds=5, num_matches=10):
    ''' drop 5 first round from each dataset '''
    return dataset.iloc[rounds*num_matches:]


def get_match_week(dataset, matches=10, rounds=38):
    ''' add number of match week to dataset '''
    j = 1
    MatchWeek = []
    for i in range(matches*rounds):
        MatchWeek.append(j)
        if ((i + 1)% matches) == 0:
            j = j + 1
    dataset['MatchWeek'] = MatchWeek
    return dataset


def select_columns(dataset, cols):
    ''' select some columns of dataset '''
    return dataset[cols]


def concat_datasets(datasets):
    ''' concat several datasets from list in on dataframe '''
    new_dataset = pd.DataFrame()
    for dataset in datasets:
        new_dataset = pd.concat((new_dataset, dataset), ignore_index=True)
    return new_dataset



''' data preprocessing - stage 2 '''

def numeric_to_categorical(dataset, col, bins):
    ''' replace numeric to categorical feature and put its in new column'''
    #set up bins
    bins = bins
    #use pd.cut function can attribute the values into its specific bins
    category = pd.cut(dataset[col], bins=bins, labels=False)
    category = category.to_frame()
    category.columns = [f'{col}_Cat']
    #concatenate old dataset and new category column
    dataset = pd.concat([dataset,category], axis = 1)
    return dataset


def calculate_correlation(dataset, target, feature_name):
    ''' function calculate correlation between feature and target value '''
    
    # create dictionary contains feature values as dictionary keys and target mean values as dictionary values 
    target_mean_dict = dataset.groupby(feature_name)[target].mean().to_dict()
    
    # calculate and print Pearson's corelation
    corr = pearsonr(list(target_mean_dict.keys()), list(target_mean_dict.values() ) )[0]
    
    return target_mean_dict, corr


def show_correlation(target_mean_dict, corr, feature_name):
    ''' fuction plot correlation between feature and target value '''
    
    # plot correlation
    plt.figure(figsize= (7.5, 7.5))
    plt.scatter(x=target_mean_dict.keys(), y=target_mean_dict.values())
    plt.title(f'Corealation between feature "{feature_name}" and target value')
    plt.ylabel('target value')
    plt.xlabel('feature value')
    plt.show()
    
    # print correlation value
    print('Pearson corelation:', corr)
    print('-'*75, '\n')


def calculate_and_show_correlation(dataset, target, feature_name):
    ''' function show and return correlation between feature and target value '''

    # create target mean dictionary and calculate Pearson's correlation
    target_mean_dict, corr = calculate_correlation(dataset, target, feature_name)

    # plot correlation
    show_correlation(target_mean_dict, corr, feature_name)
    
    return corr


def create_correlation_plot(target_mean_dict, corr, feature_name, bins):
    ''' function create single corelation plot'''
    
    # create plot for number of bins
    plt.scatter(x=target_mean_dict.keys(), y=target_mean_dict.values(), 
                label=f'Number of bins: {bins} Pearson coefficient: {corr.round(2)}')
    plt.ylabel('mean target value')
    plt.xlabel('feature value')
    plt.legend(loc='best')

    
def show_correlation_bins(results_set, feature_name):
    ''' function create corelations plot for diffrent bins '''
    
    # set plot size and title
    plt.figure(figsize= (10, 10))
    plt.title(f'Corealation between feature "{feature_name}" and mean target value')
    
    # create correlation plot for each numbers of bins
    for target_mean_dict, corr, bins in results_set:
        create_correlation_plot(target_mean_dict, corr, feature_name, bins)
    
    # show all results in one plot
    plt.show()
    print('-'*85)

    
def calculate_and_show_correlation_bins(dataset, target, bins_set, feature_name):
    ''' function calculate Peason's coefficient and create corelations plot for diffrent bins ''' 
    results_set = []
    
    for bins in bins_set:
        # create temp dataframes to store new column
        temp_df = numeric_to_categorical(dataset, feature_name, bins=bins)
        
        # create target mean dictionary and calculate Pearson's correlation
        target_mean_dict, corr = calculate_correlation(temp_df, target, f'{feature_name}_Cat')
        
        results_set.append( (target_mean_dict, corr, bins) )
        
    show_correlation_bins(results_set, feature_name)
    
    return np.array(results_set)



''' data preprocessing - stage 3 '''

# Source: https://maxhalford.github.io/blog/target-encoding-done-the-right-way/
def calc_smooth_mean(df1, df2, df3, cat_name, target, weight=10):
    ''' function return smoothing target mean encoding '''
    
    # Compute the global mean
    mean = df1[target].mean()

    # Compute the number of values and the mean of each group
    agg = df1.groupby(cat_name)[target].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the "smoothed" means
    smooth = (counts * means + weight * mean) / (counts + weight)

    # Replace each value by the according smoothed mean
    if df2 is None:
        return df1.loc[cat_name].map(smooth)
    else:
        return df1[cat_name].map(smooth), df2[cat_name].map(smooth.to_dict()), df3[cat_name].map(smooth.to_dict())



''' data preprocessing - stage 4 '''    
 
def calculate_base_metrics(clf, X_train, y_train, X_valid, y_valid):
    '''function calculate accuracy and log loss for training and validation sets'''
    base_acc_train = accuracy_score(y_train, clf.predict(X_train))
    base_acc_valid = accuracy_score(y_valid, clf.predict(X_valid))
    base_log_loss_train = log_loss(y_train, clf.predict_proba(X_train)[:,1])
    base_log_loss_valid = log_loss(y_valid, clf.predict_proba(X_valid)[:,1])
    return base_acc_train, base_acc_valid, base_log_loss_train, base_log_loss_valid


def show_base_metrics(clf, base_acc_train, base_acc_valid, base_log_loss_train, base_log_loss_valid):
    ''' function printing accuracy and log loss score for training and validation sets '''
    print(f'Base test for {clf.__class__.__name__}')
    print(f'Accuracy score on train set: {base_acc_train.round(2)}')
    print(f'Accuracy score on valid set: {base_acc_valid.round(2)}')
    print(f'Base log loss result on train set: {base_log_loss_train.round(2)}')
    print(f'Base log loss result on valid set: {base_log_loss_valid.round(2)}')    

    
def show_diff_after_shuffling(diff_acc_results, diff_log_loss_results, columns):
    ''' function show difference between acc and log loss after shuffling each feature'''

    # calculate scaling ratio to transform diff_acc_results and diff_log_loss_results to the same scale
    ratio = np.max( np.abs(diff_acc_results) ) / np.max(np.abs( diff_log_loss_results) )

    # plot differences
    plt.figure(figsize=(12.5,12.5))
    plt.barh(columns, diff_acc_results, label='accuracy')
    plt.barh(columns, diff_log_loss_results*ratio, label='log loss')
    plt.legend()
    plt.show()    


def show_acc_log_loss_difference(base_acc_train, base_acc_valid, base_log_loss_train, base_log_loss_valid,
                                reduced_acc_train, reduced_acc_valid, reduced_log_loss_train, reduced_log_loss_valid):
    ''' function show difference in accuracy and log loss between training and validation sets'''

    # create lists of result and labels as argumets to charts
    labels_acc = ['training accuracy', 'validation accuracy']
    labels_log_loss = ['training log loss', 'validation log loss']
    base_dataset_acc = [base_acc_train, base_acc_valid]
    base_dataset_log_loss = [base_log_loss_train, base_log_loss_valid]
    reduced_dataset_acc = [reduced_acc_train, reduced_acc_valid]
    reduced_dataset_log_loss = [reduced_log_loss_train, reduced_log_loss_valid]

    # create subplot and show results
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

    ax1.bar(labels_acc, base_dataset_acc, label='base dataset', width=0.2, align='edge')
    ax1.bar(labels_acc, reduced_dataset_acc, label='reduced dataset', width=-0.2, align='edge')
    ax1.legend()

    ax2.bar(labels_log_loss, base_dataset_log_loss, label='base dataset', width=0.2, align='edge')
    ax2.bar(labels_log_loss, reduced_dataset_log_loss, label='reduced dataset', width=-0.2, align='edge')
    ax2.legend()
    plt.show()


def plot_feature_importances(model, labels):
    ''' plot feature importance for passing model '''
    n_features = len(labels)
    plt.figure(figsize=(10, 10))
    if hasattr(model, 'feature_importances_'):
        plt.barh(range(n_features), model.feature_importances_, align='center')
    else:
        plt.barh(range(n_features), np.abs(model.coef_[0]), align='center')
    plt.yticks(np.arange(n_features), labels)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)
    plt.show()

    
def feature_reduction(model, X, y, base_acc, base_log_loss):
    ''' reduce number of features using perturbation techinque'''
    
    best_features_idx = []
    diff_acc_results = []
    diff_log_loss_results = []

    for i in range(X.shape[1]):

        hold = np.array(X.iloc[:, i])
        np.random.shuffle(X.iloc[:, i])

        curr_acc = accuracy_score( y, model.predict(X) )
        diff_acc = curr_acc - base_acc
        diff_acc_results.append(diff_acc)

        curr_log_loss = log_loss( y, model.predict_proba(X)[:,1] )
        diff_log_loss = curr_log_loss - base_log_loss
        diff_log_loss_results.append(diff_log_loss)
        
        # if diff_acc < 0 and diff_log_loss > 0:
        if diff_log_loss > 0:
            best_features_idx.append(i)

        X.iloc[:, i] = hold
    
    return np.array(best_features_idx), np.array(diff_acc_results), np.array(diff_log_loss_results)


def feature_reduction_deep_learning(model, X, y, base_acc, base_log_loss, kind='rnn'):
    ''' reduce number of features using perturbation techinque for deep learning models'''
    
    best_features_idx = []
    diff_acc_results = []
    diff_log_loss_results = []

    size = X.shape[1] if kind =='ann' else X.shape[2]
    
    for i in range(size):
        
        hold = X.copy()
        
        if kind == 'ann':
            np.random.shuffle(X[:, i])
        else:
            np.random.shuffle(X[:, :, i])
        
        curr_acc = accuracy_score( y, model.predict(X) )
        diff_acc = curr_acc - base_acc
        diff_acc_results.append(diff_acc)

        curr_log_loss = log_loss( y, model.predict_proba(X)[:,1] ) 
        diff_log_loss = curr_log_loss - base_log_loss
        diff_log_loss_results.append(diff_log_loss)

        # if diff_acc < 0 and diff_log_loss > 0:
        if diff_log_loss > 0:
            best_features_idx.append(i)

        X = hold
    
    return np.array(best_features_idx), np.array(diff_acc_results), np.array(diff_log_loss_results)
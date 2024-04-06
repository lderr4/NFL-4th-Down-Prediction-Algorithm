from hyperopt import hp

keep_cols = [
    'home_team', 
    'season_type',
    'week',
    'posteam', 
    'posteam_type', 
    'defteam',
    'yardline_100',
    'quarter_seconds_remaining',
    'half_seconds_remaining',
    'game_seconds_remaining', 
    'drive' ,
    'down' ,
    'goal_to_go',
    'ydstogo' , 
    'ydsnet', 
    'play_type', 
    'no_huddle', 
    'posteam_timeouts_remaining',
    'defteam_timeouts_remaining',
    'posteam_score',
    'defteam_score',
    'score_differential',
    'no_score_prob', 
    'opp_fg_prob', 
    'opp_td_prob', 
    'fg_prob', 
    'td_prob',
    'ep',    
    'wp',
    'vegas_wp',
    'wpa',
    'vegas_wpa',     
    'fixed_drive',
    'drive_play_count',    
    'spread_line',
    'total_home_epa',
    'total_away_epa', 
    'total_home_rush_epa',
    'total_away_rush_epa',
    'total_home_pass_epa',
    'total_away_pass_epa',
]

drop_cols=[
                 'home_team', 
                 'season_type',  
                 'total_home_epa',
                 'total_away_epa', 
                 'total_home_rush_epa',
                 'total_away_rush_epa',
                 'total_home_pass_epa',
                 'total_away_pass_epa',
                 'wp',
                 'vegas_wp',
                 'home_team',
                 'defteam',
                 'down',
                 'posteam_score',
                 'defteam_score',
                 'posteam',
                 'posteam_type',
                 'wpa',
                 'vegas_wpa',
                 'wpa_difference',        
                 'fixed_drive',
                 'quarter_seconds_remaining', 
                 'goal_to_go', 
                 'opp_td_prob', 
                 'opp_fg_prob',
                 'posteam_total_epa',
                 'posteam_total_rush_epa',
                 'posteam_total_pass_epa',
                 'posteam_epa_per_sec',
 
                  ]

param_space_regressors = {
        'objective': 'reg:squarederror',
        'eta': hp.uniform('eta',0.001, 0.1),
        'max_depth': hp.quniform("max_depth", 3, 13, 1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': hp.quniform('n_estimators', 100, 1000, 25)
    }

param_grid_decision_tree = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'auto', 'sqrt', 'log2'],
    'max_leaf_nodes': [None, 10, 20, 30, 40, 50]
}


dataset_csv_path = "data/fourth_down_dataset.csv"

run_path = 'models/run_model.model'
pass_path = 'models/pass_model.model'
fg_path = 'models/fg_model.model'
punt_path = 'models/punt_model.model'
classifier_path = 'models/play_type_classifier.pkl'
robo_coach_path = 'models.robo_coach.pkl'
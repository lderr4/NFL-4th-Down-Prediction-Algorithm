from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from hyperopt import fmin, tpe, STATUS_OK, Trials
import xgboost as xgb
from preprocessing import get_dataset, partition_dataset_by_play_type, prep_for_classifier
from constants import param_space_regressors, param_grid_decision_tree, classifier_path, run_path, pass_path, fg_path, punt_path
import numpy as np
import pickle


def feature_selection(base_model , X_train_full, y_train,is_reg):
    if is_reg:
        cv = KFold(n_splits=5, shuffle=True, random_state=44449999)
        scoring = 'r2'
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=44449999)
        scoring = 'accuracy'
    
    rfecv = RFECV(estimator=base_model, step=1, cv=cv, scoring=scoring)
    rfecv.fit(X_train_full, y_train)
    gr = rfecv.cv_results_
    
    num_features = list(range(1, len(gr['mean_test_score']) + 1))
    mean_minus_std = np.array(gr['mean_test_score']) - np.array(gr['std_test_score'])
    
    idx = np.argmax(mean_minus_std)
    
    num_features_selected = num_features[idx]
    
    rfe = RFE(estimator=base_model, n_features_to_select=num_features_selected, step=1)
    rfe.fit(X_train_full, y_train)
    selected_cols = X_train_full.columns[(rfe.support_)]
    return selected_cols

def objective(params):
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])
    
    
    model = xgb.XGBRegressor(
         objective=params['objective'],
         eta=params['eta'],
         max_depth=params['max_depth'],
         colsample_bytree=params['colsample_bytree'],
         min_child_weight=params['min_child_weight'],
         n_estimators=params['n_estimators'],
         seed=4455542111
    )
    
    scores = cross_val_score(model, params['X_train'], params['y_train'], cv=5, scoring='neg_root_mean_squared_error')
    rmse = np.abs(scores.mean())
    
    return {'loss': rmse, 'status': STATUS_OK}

def train_regression_models(years):
    
    dataset = get_dataset(years)
    dataset = partition_dataset_by_play_type(dataset)
    model_features = {}
    model_paths = {"run": run_path, 
                   "pass" : pass_path, 
                   "fg": fg_path, 
                   "punt": punt_path}

    for name, play_data in dataset: 
        
                
        print(f"Producing {name} model...")
        model_path = model_paths[name]
        y = play_data['wpa_avg'].to_numpy()
        X = play_data.drop('wpa_avg', axis=1)
        cols = X.columns
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        xg = xgb.XGBRegressor(objective='reg:squarederror',random_state=44449999)
        
        cols = feature_selection(xg, X_train, y_train, is_reg=True)
        model_features[name] = cols
        print(f"Number of Features: {len(cols)}")
        
        X_train_sel = X_train[cols]
        param_space_regressors['X_train'] = X_train_sel
        param_space_regressors['y_train'] = y_train
        
        trials = Trials()
        final_params = fmin(fn=objective,
                            space=param_space_regressors,
                            algo=tpe.suggest,
                            max_evals=100,
                            trials=trials)
        
        
        final_params['max_depth'] = int(final_params['max_depth'])
        final_params['n_estimators'] = int(final_params['n_estimators'])

        best_model = xgb.XGBRegressor(**final_params)
        best_model.fit(X_train_sel, y_train)
        
        print(f"Saving {name} Model to {model_path}")
        best_model.save_model(model_path)

    return model_features


def train_classifier_model(years):
    
    dataset = get_dataset(years)
    dataset = prep_for_classifier(dataset)
    y = dataset['play_type']
    X = dataset.drop('play_type', axis=1)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Calculating optimal features for Classifier...")
    dt = DecisionTreeClassifier(random_state=42)
    cols = feature_selection(dt, X_train, y_train, is_reg=False)
    X_train = X_train[cols]
    print("Features Calculated.")
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid_decision_tree, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')

    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    
    final_model = grid_search.best_estimator_

    
    with open(classifier_path, 'wb') as f:
        pickle.dump(final_model, f)
    return cols
    

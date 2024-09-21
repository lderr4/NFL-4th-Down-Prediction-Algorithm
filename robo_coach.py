import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from model_training import train_classifier_model, train_regression_models
from preprocessing import get_dataset, prep_for_classifier
from constants import classifier_path, run_path, pass_path, fg_path, punt_path, robo_coach_path
import pickle
import os
from pandas import to_numeric


class robo_coach():
    
    def __init__(self, 
                 classifier_min_threshold = 0,
                 classifier_max_threshold=0.99,
                 years=[2023, 2022, 2021, 2020, 2019,2018]):
        

        # minimum probability predicted for a play to be used in one of the regressors
        self.classifier_min_threshold = classifier_min_threshold
        
        # if one of the plays exceeds the max threshold, that will be the prediction, else 
        # any that are greater thatn the min and less than the max threshold will be run through 
        # their respective regression modelsnto determine which play has the highest predicted increase in win probability added
        self.classifier_max_threshold = classifier_max_threshold
        
        # years to use in the dataset 
        self.years = years
        
        

        if not os.path.exists('data'):
            os.makedirs('data')
        if not os.path.exists('models'):
            os.makedirs('models')

    def update_attributes(self, updates: dict):
    
        for attr, value in updates.items():
            if hasattr(self, attr):
                setattr(self, attr, value)
            else:
                print(f"Warning: {self} has no attribute '{attr}'")


    def load_models(self):
        
        run_model = xgb.Booster()
        run_model.load_model(run_path)

        pass_model = xgb.Booster()
        pass_model.load_model(pass_path)

        fg_model = xgb.Booster()
        fg_model.load_model(fg_path)

        punt_model = xgb.Booster()
        punt_model.load_model(punt_path)
        

        with open(classifier_path, 'rb') as f:
            classifier = pickle.load(f)
            
        self.run_model = run_model
        self.pass_model = pass_model
        self.fg_model = fg_model
        self.punt_model = punt_model
        self.classifier = classifier
        self.classifier_columns = classifier.feature_names_in_.tolist()
        
        self.models = {
            "field_goal": (self.fg_model, self.fg_columns),
            "pass": (self.pass_model, self.pass_columns),
            "punt": (self.punt_model, self.punt_columns),
            "run": (self.run_model, self.run_columns),
        }
    
    def train_models(self):
        self.classifier_columns = train_classifier_model(self.years)
        regressor_cols = train_regression_models(self.years)
     
        self.fg_columns = regressor_cols['fg']
        self.pass_columns = regressor_cols['pass']
        self.punt_columns = regressor_cols['punt']
        self.run_columns = regressor_cols['run']
        
        self.load_models()
        self.save_to_file(robo_coach_path)
        print(f"Robo Coach object saved to {robo_coach_path}")

    
    def get_classifier_predict_proba(self, X):
        
        X = X[self.classifier_columns]
        return self.classifier.predict_proba(X)
    
    def predict_wpa(self, df):
        # takes in a df slice with plays that are within the threshold
        suffix = '_wpa_pred'
        for play_type in self.models: # for each model, make a prediction on the entire df, and store it in the {play_type}_wpa_pred column
            model, cols = self.models[play_type]
            dmat = xgb.DMatrix(data=df[cols])
            preds = model.predict(dmat)
            df[f'{play_type}{suffix}'] = preds
        
        return df 
    
    def predict(self, X):
        def get_wpa_prediction(row, idx_to_possible_plays):
            suffix = '_wpa_pred'
            idx = row.name
            possible_plays = idx_to_possible_plays[idx]
            wpa_pred_cols = [f'{play_type}{suffix}' for play_type in possible_plays]
            
            preds = to_numeric(row[wpa_pred_cols])
            play = preds.idxmax().replace(suffix, "")
            wpa_pred = preds.max()
            
            return {idx : (play, wpa_pred)}
        
        
        X = X.reset_index()
        
        # X rows, 4 cols (probs for each play)
        predict_proba = self.get_classifier_predict_proba(X) 

        classes = self.classifier.classes_
        
        
        # true / false array with true values representing plays where one or more play probs are above the maximum threshold
        threshold_exceeded = np.any((predict_proba > self.classifier_max_threshold), axis=1)


        # inverse of threshold exceeded array
        threshold_not_exceeded = ~threshold_exceeded
        

        # indices where threshold is not exceeded; [0] because np.where is weird 
        classifier_preds = np.where(threshold_exceeded)[0] 
        
        # dict mapping the index to the predicted class for plays that did not exceed threshold
        idx_to_pred = {i: classes[np.argmax(predict_proba[i])] for i in classifier_preds}
        
        # indices where the regressor is going to make predictions (threshold is exceeded)
        regressor_preds = np.where(threshold_not_exceeded)[0]
        
        # maps idx to [class1, class2] for each prediction that will be made by the regressors
        idx_to_possible_plays = {i: [c for c in classes[np.where(predict_proba[i] > self.classifier_min_threshold)[0]]] for i in regressor_preds}

        


        # get the dataframe with the respective idx values for regression prediction
        X_uncertain = X.iloc[regressor_preds,]
        
        X_uncertain = self.predict_wpa(X_uncertain)
        
        # get the highest wpa predicted of the valid plays
        final_preds = X_uncertain.apply(get_wpa_prediction,idx_to_possible_plays=idx_to_possible_plays, axis=1)
        
        final_preds_master_dict = {}
        
        for item in final_preds:
            final_preds_master_dict.update(item)
        
        final_preds_master_dict.update(idx_to_pred)
        
        final_preds_list = [0] * len(final_preds_master_dict)
        
        for idx, val in final_preds_master_dict.items():
            final_preds_list[idx] = val
        
        return final_preds_list
            


    def save_to_file(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
        return obj
    
       
def main():
    
    with open(robo_coach_path, 'rb') as f:
        rb = pickle.load(f)
    dataset = get_dataset(rb.years)
    dataset = prep_for_classifier(dataset)
    
    
    y = dataset[['play_type', 'wpa_avg']]
    X = dataset.drop('play_type', axis=1)
    X = dataset.drop('wpa_avg', axis=1)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_test.to_csv("results/X_test.csv")
    y_test.to_csv("results/y_test.csv")
    
    

    # cmints = [-1, 0, 0.00001, 0.0001, 0.001, 0.01, 0.02, 0.05, 0.1, 0.5]
    # cmaxts = [2, 1, 0.99999, 0.9999, 0.999, 0.99, 0.9, 0.85, 0.6]
    # for cmint in cmints:
    #     for cmaxt in cmaxts:
    #         print(cmint, cmaxt)

    #         updates = {"classifier_min_threshold" : cmint,
    #                     "classifier_max_threshold": cmaxt}
    #         rb.update_attributes(updates)
    #         preds = rb.predict(X_test)

    #         filename = f"results/min{cmint}max{cmaxt}.pickle"

    #         with open(filename, 'wb') as f:
    #             pickle.dump(preds, f)
            
    




if __name__ == "__main__":
    main()   
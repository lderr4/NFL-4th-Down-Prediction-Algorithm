import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from model_training import train_classifier_model, train_regression_models
from preprocessing import get_dataset, prep_for_classifier
from constants import classifier_path, run_path, pass_path, fg_path, punt_path, robo_coach_path
import pickle
import os


class robo_coach():
    
    def __init__(self, 
                 classifier_min_threshold = 0,
                 classifier_max_threshold=0.99,
                 years=[2023, 2022, 2021, 2020, 2019,2018]):
        
        self.classifier_min_threshold = classifier_min_threshold
        self.classifier_max_threshold = classifier_max_threshold
        self.years = years
        
        

        if not os.path.exists('data'):
            os.makedirs('data')
        if not os.path.exists('models'):
            os.makedirs('models')


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
        
        suffix = '_wpa_pred'
        for play_type in self.models:
            model, cols = self.models[play_type]
            dmat = xgb.DMatrix(data=df[cols])
            df[f'{play_type}{suffix}'] = model.predict(dmat)
        
        return df
    
    def predict(self, X):
        def get_wpa_prediction(row, idx_to_possible_plays):
            suffix = '_wpa_pred'
            idx = row.name
            possible_plays = idx_to_possible_plays[idx]
            wpa_pred_cols = [f'{play_type}{suffix}' for play_type in possible_plays]
            preds = row[wpa_pred_cols]
            
            play = preds.idxmax().replace(suffix, "")
            wpa_pred = preds.max()
            
            return {idx : (play, wpa_pred)}
            
        
        X = X.reset_index()
        
        predict_proba = self.get_classifier_predict_proba(X)
        classes = self.classifier.classes_
    
        
        threshold_exceeded = np.any((predict_proba > self.classifier_max_threshold), axis=1)
        
        theshold_not_exceeded = ~threshold_exceeded
        
        classifier_preds = np.where(threshold_exceeded)[0]
        
        idx_to_pred = {i: classes[np.argmax(predict_proba[i])] for i in classifier_preds}
        
        regressor_preds = np.where(theshold_not_exceeded)[0]
        
        idx_to_possible_plays = {i: [c for c in classes[np.where(predict_proba[i] > self.classifier_min_threshold)[0]]] for i in regressor_preds}
    
        X_uncertain = X.iloc[regressor_preds,]
        
        X_uncertain = self.predict_wpa(X_uncertain)
        
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
    outfile = "robo_coach.pkl"
    rb = robo_coach()
    rb.train_models()
    rb.save_to_file(outfile)
    dataset = get_dataset(rb.years)
    dataset = prep_for_classifier(dataset)
    print(dataset.columns)
    y = dataset['play_type']
    X = dataset.drop('play_type', axis=1)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preds = rb.predict(X_test)
    print(preds)
    




if __name__ == "__main__":
    main()   
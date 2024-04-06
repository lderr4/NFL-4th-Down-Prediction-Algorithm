from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from xgboost import DMatrix


def regression_accuracy(y_pred, y_true):
    c = 0
    for yhat, y in zip(y_pred, y_true):
        if (yhat > 0 and y > 0) or (yhat < 0 and y < 0):
            c += 1
    return c / len(y_true)

def mse_positive(y_pred, y_true):
    indices = np.where(y_true > 0)[0]
    mse = metrics.mean_squared_error(y_pred[indices], y_true[indices])
    return mse
    
        

def mse_negative(y_pred, y_true):
    indices = np.where(y_true < 0)[0]
    mse = metrics.mean_squared_error(y_pred[indices], y_true[indices])
    return mse

class regression_evaluator():
    
    
    def __init__(self, col_names):
        self.col_names = col_names
        funcs = [
            metrics.max_error,
            metrics.mean_absolute_error,
            mse_positive,
            mse_negative,
            regression_accuracy,
            metrics.mean_squared_error,
            metrics.root_mean_squared_error,
            metrics.r2_score,
            
        ]
        func_names = [
            'Max Error',
            'Mean Absolute Error',
            'Mean Squared Error for +wpa',
            'Mean Squared Error for -wpa',
            'Regression Accuracy',
            'Mean Squared Error',
            'Root Mean Squared Error',
            'R-Squared',
            
        ]
        
        self.model_metrics = [] # (model, {metric dictionary})
        self.model_feature_importances = []
        
        self.metric_list = list(zip(func_names, funcs))
        
    def evaluate(self, model, X_train, X_test, y_train, y_test, clear=True, pr = False, dmat=False):
        if clear:
            self.clear_saved_metrics()
        train_mets = self.metrics(model, X_train, y_train, message="Train", pr=pr, dmat=dmat)
        test_mets = self.metrics(model, X_test,y_test, message="Test", pr=pr, dmat=dmat)
        if hasattr(model, 'feature_importances_'):
            self.feature_importances(model, pr)
        if pr:
            self.reg_plot(model, X_train, y_train, "Train", dmat=dmat)
            self.reg_plot(model, X_test, y_test, "Test", dmat=dmat)
        
        return train_mets, test_mets
    
    def metrics(self,model, X, y, message="Test", pr=False, dmat=False):
        
        '''
        y_true, y_pred
        model must be fitted
        '''
        
        if dmat:
            X_dmat = DMatrix(data=X)
            y_pred = model.predict(X_dmat)
        else:
            y_pred = model.predict(X)
        
        model_metrics = {}
        
        if pr:
            print(f"Evaluation for ({message})")
        
        for metric_name, metric_function in self.metric_list:
            
            score = metric_function(y_pred, y)
            model_metrics[metric_name] = score
            if pr:
                print(f"{metric_name}: {score}")
          
        self.model_metrics.append((model , model_metrics))
        if pr:
            print("----------------------------")
        
        return model_metrics
    
    
    def feature_importances(self,model, pr=False):
        feature_importance_list = []
        if pr:
            print(f"Feature Importances")
        for name, imp in sorted(zip(self.col_names, model.feature_importances_), key=lambda x:x[1], reverse=True):
            if pr:
                print(f"{name}: {imp}")
            feature_importance_list.append((name, imp))
        if pr:
            print("----------------------------")
        self.model_feature_importances.append((model, feature_importance_list))
    
    
    def reg_plot(self, model, X  , y_true, message, dmat=False):
        if dmat:
            X_dmat = DMatrix(data=X)
            y_predict = model.predict(X_dmat)
        else:
            y_predict = model.predict(X)
        residuals = y_true - y_predict
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(y_predict, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.title(f'Residuals vs Fitted ({message})')
        plt.xlabel('Fitted values')
        plt.ylabel('Residuals')

        # Plot y_true vs y_predict
        plt.subplot(1, 2, 2)
        plt.scatter(y_true, y_predict, alpha=0.5)
        line_x = np.linspace(start=min(min(y_true), min(y_predict)), stop=max(max(y_true), max(y_predict)), num=1000)
        line_y = line_x  # y = x
        plt.axvline(x=0, color='gray', linestyle='--')
        plt.axhline(y=0, color='gray', linestyle='--')
        plt.plot(line_x, line_y, color='red', label='y = x')
        plt.title(f'True values vs Predicted values ({message})')
        plt.xlabel('True values')
        plt.ylabel('Predicted values')

        plt.tight_layout()
        plt.show()
        
    def clear_saved_metrics(self):
        self.model_feature_importances = []
        self.model_metrics = []
        
        
        
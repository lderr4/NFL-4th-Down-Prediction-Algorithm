# Robo-NFL-Coach
  
## Description   
This project combines classification and regression tasks to determine whether a team should run, pass, punt, or kick a field goal on 4th down, emulating the duties of an NFL coach. 

[Python Libraries Used](https://github.com/lderr4/Robo-NFL-Coach/blob/main/requirements.txt)

## How to run   
```bash
# clone project   
git clone https://github.com/lderr4/Robo-NFL-Coach.git

# Switch to project directory 
cd Robo-NFL-Coach

# Setup Python3.8 Virtual Env
python3.8 -m venv env_name
source env_name/bin/activate
pip install -r requirements.txt

# Run the Training Script (which saves the Models, Dataset, and Class):
python3 robo_coach.py
```
## Things I Learned
- Nailing down the correct approach to a problem (before writing any code) is the key to avoiding time waste
- Importance of feature engineering and preprocessing
- Recursive Feature Elimination and Bayesian Optimization
- Cleaning up messy Jupyter Notebook code into usable Python Modules
- Visual evaluation metrics are often more informative than scores

## Challenges/Shortcomings
- Coming up with the approach to the question: What is the optimal 4th Down Play?
- Going from ~300 features in the raw dataset to ~30
- Poor Performance of Regression Models

## Flowcharts
### Model Training Flowchart
![Model Training Flowchart Image](https://github.com/lderr4/Robo-NFL-Coach/blob/main/Model-Training-Data-Flow.png)

This diagram depicts the flow of data from the initial API request to the final loading of each model. Data is pulled from the [nfl_data_py](https://pypi.org/project/nfl-data-py/) library. In total, five models are trained:
- **Classifier:** a multi-class Decision Tree Classifier which classifies fourth down plays as Pass, Run, Punt, or Field Goal.
- **Pass Regressor:** Predict the change in win probability (Win Probability Added) for Passing plays on 4th down
- **Run Regressor:** Predict the Win Probability Added for Run plays on 4th down
- **Punt Regressor:** Predict the Win Probability Added for Punt plays on 4th down
- **Field Goal Regressor:** Predict the Win Probability Added for Field Goal plays on 4th down

### Predict Function Flowchart
![Predict Function Flowchart Image](https://github.com/lderr4/Robo-NFL-Coach/blob/main/Robot-Coach-Predict-Function.png)

This diagram illustrates the decision flow chart for the predict function of the robo coach class. The classifier's _predict_proba_ function is used to determine the degree of certainty of the prediction. If the probability of the highest play is above the max probability threshold parameter, the classifier will be used for the prediction. Otherwise, for each play probability that is greater than the minimum probability threshold parameter, the corresponding regression model is used to predict Win Probability Added. The play with the highest Win Probability Added is chosen. 

## Metrics and Plots
### 1. Features
#### 1.1 Feature Correlation Heatmap
![Correlation Heatmap](https://github.com/lderr4/Robo-NFL-Coach/blob/main/plots/corr_heatmap.png)
Feature correlation was one of the feature selection techniques used in this project.

#### 1.2 Feature vs Win Probability Added (wpa) Scatterplots
![Correlation Heatmap](https://github.com/lderr4/Robo-NFL-Coach/blob/main/plots/features_vs_wpa.png)

### 2. Classifier
#### 2.1 Final Classifier Confusion Matrix On Test Set
![Confusion Matrix](https://github.com/lderr4/Robo-NFL-Coach/blob/main/plots/confusion_mat.png)

#### 2.2 Probability Distribution of _predict_proba_ Function on Testset
![Probability Distribution](https://github.com/lderr4/Robo-NFL-Coach/blob/main/plots/probability%20distributions%20of%204th%20down%20plays.png)

#### 2.3 Proportion of 4th Down Plays with a Play Exceeding Probability Threshold
![Proportion of 4th Down Plays with a Play Exceeding Probability Threshold](https://github.com/lderr4/Robo-NFL-Coach/blob/main/plots/percent_exceeding_threshold.png)
Notice the proportion of passing plays shooting up at about the 1% mark. This indicates lots of 4th down plays have ~1% probability of being a passing play. This is because of fake punts and trick plays which happen in typical punt or field goal situations.

### 3. Regressors
#### 3.1 Run Regressor Metrics (Test)
| Metric | Score |
|------------|---------|
| $R^2$     |  0.525    | 
| _MSE_     |  0.00143  |
| _MAE_     |  0.0257  |
| _Correct Sign (+/-) %_        |      0.882     |

#### 3.2 Run Regressor Plots
![Run Regressor Plots](https://github.com/lderr4/Robo-NFL-Coach/blob/main/plots/run_plots.png)

#### 3.3 Pass Regressor Metrics (Test)
| Metric | Score |
|------------|---------|
| $R^2$     |  0.392    | 
| _MSE_     |  0.00275  |
| _MAE_     |  0.0266  |
| _Correct Sign (+/-) %_        |      0.865     |

#### 3.4 Pass Regressor Plots
![Pass Regressor Plots](https://github.com/lderr4/Robo-NFL-Coach/blob/main/plots/pass_plots.png)


#### 3.5 Field Goal Regressor Metrics (Test)
| Metric | Score |
|------------|---------|
| $R^2$     |  -0.0573    | 
| MSE     |  0.00291  |
| MAE     |  0.0247  |
| Correct Sign (+/-) %        |      0.766     |

**Note**: Despite my best efforts, the field goal model still has a negative $R^2$, meaning simply predicting the mean would yield a lower $MSE$. This is because the outcome of a field goal is incredibly hard to predict, and is essentially random. 

#### 3.6 Field Regressor Plots
![Run Regressor Plots](https://github.com/lderr4/Robo-NFL-Coach/blob/main/plots/fg_plots.png)


#### 3.7 Punt Regressor Metrics (Test)
| Metric | Score |
|------------|---------|
| $R^2$     |  0.339    | 
| _MSE_     |  0.000908  |
| _MAE_     |  0.0151  |
| _Correct Sign (+/-) %_        |      0.742     |

#### 3.8 Punt Regressor Plots
![Run Regressor Plots](https://github.com/lderr4/Robo-NFL-Coach/blob/main/plots/punt_plots.png)

## Results

![https://github.com/lderr4/Robo-NFL-Coach/blob/main/plots/classifier_usage.png]

![https://github.com/lderr4/Robo-NFL-Coach/blob/main/plots/model_acc_heat.png]

![https://github.com/lderr4/Robo-NFL-Coach/blob/main/plots/reg_acc_heat.png]

![https://github.com/lderr4/Robo-NFL-Coach/blob/main/plots/wrong_preds_heat.png]









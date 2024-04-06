# Robo-NFL-Coach
A machine learning algorithm to automate the duties of an NFL Coach.
  
## Description   
This project combines classification and regression tasks to determine whether a team should run, pass, punt, or kick a field goal on 4th down.

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
This diagram show the decision flow chart for the predict function of the robo coach class. The classifier's _predict_proba_ function is used to determine the degree of certainty of the prediction. If the probability of the highest play is above the max probability threshold parameter, the classifier will be used for the prediction. Otherwise, for each play probability that is greater than the minimum probability threshold parameter, the corresponding regression model is used to predict Win Probability Added. The play with the highest Win Probability Added is chosen. 

## Metrics and Plots
### 1. Features
#### 1.1 Feature Correlation Heatmap
![Correlation Heatmap](https://github.com/lderr4/Robo-NFL-Coach/blob/main/plots/corr_heatmap.png)
Feature correlation was one of the feature selection techniques used in this project.

#### 1.2 Feature vs Win Probability Added Scatterplots
![Correlation Heatmap](https://github.com/lderr4/Robo-NFL-Coach/blob/main/plots/features_vs_wpa.png)

### 2. Classifier
#### 2.1 Final Classifier Confusion Matrix On Test Set
![Confusion Matrix](https://github.com/lderr4/Robo-NFL-Coach/blob/main/plots/confusion_mat.png)

#### 2.2 Probability Distribution of _predict_proba_ Function on Testset
![Probability Distribution](https://github.com/lderr4/Robo-NFL-Coach/blob/main/plots/probability%20distributions%20of%204th%20down%20plays.png)

#### 2.3 Proportion of 4th Down Plays with a Play Exceeding Probability Threshold
![Proportion of 4th Down Plays with a Play Exceeding Probability Threshold](https://github.com/lderr4/Robo-NFL-Coach/blob/main/plots/percent_exceeding_threshold.png)
Notice the proportion of passing plays shooting up at about the 1% mark. This indicates lots of 4th down plays have ~1% probability of being a passing play. This is because of fake punts and trick plays which happen in typical punt or field goal situations.

### Regressors
#### Run Regressor Metrics (Test)








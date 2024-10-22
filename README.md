# T20 Cricket Score Prediction Model

### Overview

This project builds an IPL score prediction model using various machine learning techniques. The model predicts the total score of a team based on match conditions like overs, wickets, and runs. It utilizes popular regression algorithms such as Random Forest, XGBoost, Decision Tree, and more. The dataset used for training contains IPL match data, and Exploratory Data Analysis (EDA) is performed to better understand the features.

### Project Structure

#### 1) Data Loading and Exploration

   The IPL dataset (ipl_data.csv) is loaded and explored using basic Pandas functions to check for missing values, datatype, and unique values.

#### 2) Exploratory Data Analysis (EDA)

 - Visualizations for wickets and runs distribution using Seaborn.
 - Heatmap displaying correlation between numerical features.

#### 3) Data Cleaning

- Unnecessary columns such as mid, date, venue, and others are removed.
- Filtering to only include consistent teams.
- Removal of the first 5 overs of every match, as they do not provide significant insight for prediction.

#### 4) Data Preprocessing and Encoding

- Label encoding is performed on categorical columns (bat_team and bowl_team).
- One-hot encoding is applied to prepare the dataset for model training.

#### 5) Model Building

#### Five machine learning models are trained:
- Decision Tree Regressor
- Linear Regression
- Random Forest Regressor
- Support Vector Regressor
- XGBoost Regressor

#### 6) Model Evaluation

  Each model is evaluated using Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) to measure prediction performance.

#### 7) Prediction Function

  A function score_predict is created to predict the final score of a match based on inputs such as batting team, bowling team, runs, wickets, overs, and runs/wickets in the last 5 overs.

#### 8) Model Exporting

  The best-performing model, Random Forest, is saved using Pickle for future use.


# IPL Score Prediction Model

## Project Overview
This repository contains an IPL score prediction model built using machine learning techniques. The objective is to predict the final score of an IPL team based on match conditions such as overs played, wickets fallen, and runs scored. Multiple regression algorithms like **Random Forest**, **XGBoost**, **Decision Tree**, and more are used to build and evaluate the model.

---

## Table of Contents
1. [Installation](#installation)
2. [Project Structure](#project-structure)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Training](#model-training)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Usage](#usage)
7. [Results](#results)
8. [Future Scope](#future-enhancements)

---

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
       git clone https://github.com/username/ipl-score-prediction.git

2. Install the required dependencies

       pip install numpy pandas seaborn matplotlib scikit-learn xgboost

3. Run the Python scripts provided in the repository.

## Project Structure

- **data/**: Contains the IPL dataset (`ipl_data.csv`).
- **notebooks/**: Jupyter notebooks for data exploration and model training.
- **models/**: Serialized machine learning models saved using Pickle.
- **scripts/**: Python scripts for model training, evaluation, and prediction.
- **README.md**: Project documentation.

---

## Data Preprocessing

- **Label Encoding**: Converts categorical variables like `bat_team` and `bowl_team` into numerical values using `LabelEncoder` from `sklearn`.

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ipl_df['bat_team'] = le.fit_transform(ipl_df['bat_team'])
ipl_df['bowl_team'] = le.fit_transform(ipl_df['bowl_team'])
```

## Model Training

The project includes training five different models:

- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Support Vector Regressor**
- **XGBoost Regressor**

Each model is trained using `train_test_split` to divide the dataset into training and testing sets. The best model is selected based on evaluation metrics like **MSE**, **MAE**, and **RMSE**.

---

## Evaluation Metrics

The following metrics are used to evaluate the model's performance:

- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**

Example:

```python
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(test_labels, predictions)
```
## Usage

### Train the Model:

```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(train_features, train_labels)
```

### Predict Scores:
Use the `score_predict` function with match details to predict the final score:

```python
predicted_score = score_predict('Mumbai Indians', 'Chennai Super Kings', 90, 3, 12, 30, 1)
```

### Export Model:
To save the trained model for future use, you can export it using Pickle:

```python
import pickle
filename = 'ipl_score_prediction.pkl'
pickle.dump(rf, open(filename, 'wb'))
```

## Conclusion
The IPL Score Prediction Model effectively leverages machine learning algorithms to predict cricket match scores based on historical data. By employing various models such as Linear Regression, Decision Tree, Random Forest, Support Vector Regressor, and XGBoost, we are able to achieve reliable predictions that can assist analysts, teams, and fans in understanding potential match outcomes.

## Future Scope
- **Feature Enhancement**: Integrate additional features such as player form, weather conditions, and venue statistics to improve prediction accuracy.
- **Deep Learning Models**: Explore the implementation of deep learning algorithms to capture more complex patterns in the data for enhanced predictions.
- **Real-Time Predictions**: Develop a web-based interface or API to facilitate real-time predictions during live matches.
- **Comprehensive Analytics**: Expand the model to provide detailed analytics and insights, such as player performance and team strategies.


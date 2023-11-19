import pandas as pd
import os
import numpy as np

# Read in the data
train_data = pd.read_csv('covid_train.csv')
test_data = pd.read_csv('covid_test.csv')

# Use XGBoost to train the model
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectPercentile 
from sklearn.feature_selection import r_regression
from sklearn.feature_selection import f_regression


# Split the data into train and test
X, y = train_data.iloc[:, :-1], train_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# Select the features which correlation is greater than 0.8

selectot = SelectPercentile(score_func=f_regression, percentile=30)
selectot.fit(X, y)
X = selectot.transform(X)
test_data = selectot.transform(test_data)


# Grid search for best estimator for MLPRegressor

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# Define the model
reg = MLPRegressor(random_state=42)

# Define the parameters
parameters = {
       'hidden_layer_sizes': [(8, 8, 8), (8, 8), (8,), (16, 16, 16), (16, 16), (16,), (32, 32, 32), (32, 32), (32,)],
       'activation': ['relu'],
       'solver': ['adam'],
       'alpha': [0.0001, 0.05, 0.1, 0.5, 1],
       'learning_rate': ['constant', 'adaptive'],
       'max_iter': [1000, 2000, 3000, 4000, 5000],
       'learning_rate_init': [0.001, 0.01, 0.1],
       'early_stopping': [True, False]
}

# Define the scorer
scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Grid search
grid_obj = GridSearchCV(reg, parameters, scoring=scorer, n_jobs=-1, cv=5, verbose=1)
grid_fit = grid_obj.fit(X, y)

# Print average CV score of the best estimator
print(grid_fit.best_score_)

# Print the best parameters
print(grid_fit.best_params_)

# Get the best estimator
reg = grid_fit.best_estimator_

# Train the model
reg.fit(X, y)





# Save the results with only id and prediction(tested_positive)
output_pd = pd.DataFrame()
output_pd['tested_positive'] = reg.predict(test_data)
output_pd["id"] = output_pd.index
output_pd = output_pd[['id', 'tested_positive']]
output_pd.to_csv('submission.csv', index=False)










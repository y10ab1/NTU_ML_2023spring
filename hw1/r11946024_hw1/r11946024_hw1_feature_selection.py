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


# Split the raw data into data and label
X, y = train_data.iloc[:, :-1], train_data.iloc[:, -1]

# Scale the data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
test_data = scaler.transform(test_data)



# Select the features using RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge

# Create the RFE object and rank each pixel
rfe = RFE(estimator=Ridge(), n_features_to_select=8)
rfe.fit(X, y)
X = rfe.transform(X)
test_data = rfe.transform(test_data)



# # Select the features which correlation is greater than 0.8
# from sklearn.feature_selection import SelectPercentile
# from sklearn.feature_selection import f_regression

# selectot = SelectPercentile(score_func=f_regression, percentile=50)
# selectot.fit(X, y)
# X = selectot.transform(X)
# test_data = selectot.transform(test_data)


# Stacking different models
from sklearn.ensemble import StackingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge


from sklearn.metrics import mean_squared_error

# Grid search for each model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer



estimators = [
       ('rf', RandomForestRegressor(random_state=42)),
       # ('gb', GradientBoostingRegressor(random_state=42)),
       # ('ada', AdaBoostRegressor(random_state=42)),
       # ('dt', DecisionTreeRegressor(random_state=42)),
       # ('knn', KNeighborsRegressor()),
       ('svr', SVR()),
       
       ('xgb', xgb.XGBRegressor(objective ='reg:squarederror', random_state=42)),
       ('mlp', MLPRegressor(random_state=42, 
                            activation='relu', 
                            solver='adam', 
                            max_iter=10000, 
                            alpha=1, 
                            learning_rate='adaptive', 
                            learning_rate_init=0.01,
                            batch_size=256,
                            early_stopping=True,
                            hidden_layer_sizes=(128,64,32,16))),
       ('mlp2', MLPRegressor(random_state=42, 
                            activation='relu', 
                            solver='adam', 
                            max_iter=10000, 
                            alpha=1, 
                            learning_rate='adaptive', 
                            learning_rate_init=0.01,
                            batch_size=256,
                            early_stopping=True,
                            hidden_layer_sizes=(256,128,64,32))),
       ('mlp3', MLPRegressor(random_state=42, 
                            activation='relu', 
                            solver='adam', 
                            max_iter=10000, 
                            alpha=1, 
                            learning_rate='adaptive', 
                            learning_rate_init=0.01,
                            batch_size=256,
                            early_stopping=True,
                            hidden_layer_sizes=(512,128,32,16)))
]


# # Grid search for the final estimator
# param_grid = {
#     'rf__n_estimators': [100, 200, 300],
#        'rf__max_depth': [10, 20, 30],
#        'rf__min_samples_split': [2, 4, 6],
#        'gb__n_estimators': [100, 200, 300],
#        'gb__max_depth': [10, 20, 30],
#        'gb__min_samples_split': [2, 4, 6],
#        'ada__n_estimators': [100, 200, 300],
#        'ada__learning_rate': [0.1, 0.2, 0.3],
#        'dt__max_depth': [10, 20, 30],
#        'dt__min_samples_split': [2, 4, 6],
#        'knn__n_neighbors': [3, 5, 7],
#        'knn__weights': ['uniform', 'distance'],
#        'svr__C': [0.1, 1, 10],
#        'svr__gamma': [0.1, 1, 10],
#        'svr__kernel': ['rbf', 'poly', 'sigmoid'],
#        'mlp__hidden_layer_sizes': [(10,),(50, ), (100, ), (200, ), (300, )],
#        'mlp__alpha': [0.1, 0.01, 0.001],
#        'xgb__n_estimators': [100, 200, 300],
#        'xgb__max_depth': [10, 20, 30],
#        'xgb__min_samples_split': [2, 4, 6],
#        'final_estimator__fit_intercept': [True, False],
#        'final_estimator__normalize': [True, False]
# }

# # Define the scorer
# scorer = make_scorer(mean_squared_error, greater_is_better=False)

# grid_search = GridSearchCV(reg, param_grid=param_grid, cv=5, scoring=scorer , n_jobs=-1, verbose=1)
# grid_search.fit(X, y)

# reg = grid_search.best_estimator_



# Clear training weights and refit the model by KFold cross validation to avoid overfitting
reg = StackingRegressor(
       estimators=estimators, passthrough=True
)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold = KFold(n_splits=5, random_state=42, shuffle=True)
scores = cross_val_score(reg, X, y, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
print("StackingRegressor score: %.3f" % scores.mean())


# refit the model
reg = StackingRegressor(
       estimators=estimators, passthrough=True
)
reg.fit(X, y)






# Save the results with only id and prediction(tested_positive)
output_pd = pd.DataFrame()
output_pd['tested_positive'] = reg.predict(test_data)
output_pd["id"] = output_pd.index
output_pd = output_pd[['id', 'tested_positive']]
output_pd.to_csv('submission.csv', index=False)










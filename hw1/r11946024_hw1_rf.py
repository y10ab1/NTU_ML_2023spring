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

# Select the features included 'ili', 'cli', tested_positive'
# train_data = train_data[['ili', 'cli', 'tested_positive','tested_positive.2']]
# test_data = test_data[['ili', 'cli', 'tested_positive']]
# print(train_data.head())
# print(test_data.head())

# Split the data into train and test
X, y = train_data.iloc[:, :-1], train_data.iloc[:, -1]

###################### Manually select the features #####################
# X = X[['ili', 'nohh_cmnty_cli', 'hh_cmnty_cli', 'tested_positive',
#        'tested_positive.1']]

# test_data = test_data[['ili', 'nohh_cmnty_cli', 'hh_cmnty_cli', 'tested_positive',
#        'tested_positive.1']]

X = X[['nohh_cmnty_cli.1', 'cli', 'hh_cmnty_cli.1', 'nohh_cmnty_cli',
       'wbelief_masking_effective',  'ili', 'hh_cmnty_cli',
       'tested_positive', 'tested_positive.1']]
test_data = test_data[['nohh_cmnty_cli.1', 'cli', 'hh_cmnty_cli.1', 'nohh_cmnty_cli',
       'wbelief_masking_effective',  'ili', 'hh_cmnty_cli',
       'tested_positive', 'tested_positive.1']]
#########################################################################

print(X.head())
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Train the model with grid search
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# Define the random forest model
rf_model = RandomForestRegressor(random_state=42)

# Define the parameters
parameters = {'n_estimators': [100, 200, 300, 500], 'max_depth': [3, 5, 7, 9, 11], 'min_samples_split': [2, 5, 10, 20], 'min_samples_leaf': [1, 2, 4, 8]}

# Define the scorer
scorer = make_scorer(mean_squared_error, greater_is_better=False)

# Perform grid search with some logging
grid_obj = GridSearchCV(rf_model, parameters, scoring=scorer, verbose=1, n_jobs=-1, cv=5)
#grid_fit = grid_obj.fit(X, y)
grid_fit = grid_obj.fit(X_train, y_train)

# Get the best estimator
best_xgb = grid_fit.best_estimator_




# Make predictions using the unoptimized and model
predictions = (rf_model.fit(X_train, y_train)).predict(X_test)
best_predictions = best_xgb.predict(X_test)

# Report the before-and-afterscores

print("Unoptimized model: ", mean_squared_error(y_test, predictions))
print("Optimized Model: ", mean_squared_error(y_test, best_predictions))

# Refit the model on the training data
best_xgb.fit(X, y)

# Save the results with only id and prediction(tested_positive)
output_pd = pd.DataFrame()
output_pd['tested_positive'] = best_xgb.predict(test_data)
output_pd["id"] = output_pd.index
output_pd = output_pd[['id', 'tested_positive']]
output_pd.to_csv('submission.csv', index=False)

# Use SHAP to explain the xgb_model
from shap import TreeExplainer, summary_plot
explainer = TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_train)



# Select features with top 10 SHAP values and train the model again
shap_vals = explainer.shap_values(X_train)
shap_vals = np.abs(shap_vals).mean(axis=0)
print(X_train.columns[np.argsort(shap_vals)[-2:]])
print(X_train.columns[np.argsort(shap_vals)[-2:]].values)



# Plot the SHAP values and save the plot to shap_plot.png

summary_plot(shap_values, X_train, plot_type="bar", show=False)
import matplotlib.pyplot as plt
plt.savefig('shap_plot.png')









import pandas as pd

# Read in the data
submission = pd.read_csv('submission.csv')

# Switch the order of the columns
submission = submission[['id', 'tested_positive']]

# Save the submission file
submission.to_csv('submission.csv', index=False)
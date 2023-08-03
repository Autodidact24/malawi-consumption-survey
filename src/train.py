# Import necessary libraries
import pandas as pd
import yaml

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load parameters
with open('params.yaml') as file:
    params = yaml.safe_load(file)

## Read the data
data = pd.read_csv(params['data']['filepath'])

# Drop columns that are not predictive features
data = data.drop(columns=['unique_id', 'latitude', 'longitude'])

# Handle missing values by dropping rows with missing values
data = data.dropna()

# Convert categorical variables into numerical using label encoding
for col in data.columns[data.dtypes == 'object']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Assuming we have identified the top 10 features and stored them in 'top_10_features' list
top_10_features = ['expenditure_2', 'hhsize', 'drinking_water_cost', 'days_milk', 'days_sugar', 'used_computer', 'age', 'breakfast_dependant', 'rooms_occupied', 'age_school_start']

# Prepare the data
X = data[top_10_features]
y = data['consumption']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params['train']['train_test_split'], random_state=params['train']['random_state'])

# Create a model
model = RandomForestRegressor(n_estimators=params['model']['n_estimators'], max_depth=params['model']['max_depth'], random_state=params['model']['random_state'])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print and write the evaluation metrics to text files
print('Mean Squared Error:', mse)
with open('reports/MSE.txt', 'w') as f:
    f.write(str(mse))

print('Mean Absolute Error:', mae)
with open('reports/MAE.txt', 'w') as f:
    f.write(str(mae))

print('R^2 Score:', r2)
with open('reports/R2.txt', 'w') as f:
    f.write(str(r2))

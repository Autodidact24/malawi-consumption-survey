# Predictive Model for Consumption in Malawi

This project aims to understand household consumption in Malawi using survey data. The main goal is to identify the top 10 most predictive questions which will be used in a shorter survey to administer to households in Malawi.

## Project Structure

The project has the following structure:

- `src/`: This folder contains the main Python scripts for the project.
- `data/raw/`: This folder contains the raw survey data used for the analysis.
- `reports/`: This folder contains the generated reports, including the Mean Squared Error of the model.
- `dvc.yaml`: This file contains the DVC pipeline definition.
- `params.yaml`: This file contains the parameters for the model and data paths.
- `requirements.txt`: This file lists all the Python dependencies required to run the project.

## Data

The data for this project comes from a survey of more than 11,000 households in Malawi. It contains responses related to education, health, food consumption, and other indicators that might be predictive of overall consumption.

## Feature Importance

The feature importance is determined using a RandomForestRegressor, which provides an estimate of the contribution of each feature in predicting the target variable. The top 10 most predictive questions are selected based on their feature importance score.

## Setup and Installation

To set up the project, follow these steps:

1. Get the project folder locally.
2. Run the following commands to setup the virtual environment:
```
python3 -m venv .venv
echo "export PYTHONPATH=$PWD" >> .venv/bin/activate
source .venv/bin/activate
```
3. Install the required dependencies using the command `pip install -r requirements.txt`.
4. Run the DVC pipeline using the command `dvc repro`.

## Model

The model used in this project is a Random Forest Regressor. It is trained using a subset of the data, specifically the top 10 most predictive questions identified through feature importance analysis.

## Question 3: Evaluation

The model is evaluated using Mean Squared Error, Mean Absolute Error, and R^2 score. The results can be found in the `reports/` directory.

1. The R^2 score of 0.779 suggests that the model is able to explain approximately 78% of the variability in the target variable (consumption). This is a good indicator that our model performs well. However, the Mean Squared Error (MSE) is quite high (22994039403.56002), which could be due to outliers in the data. While this doesn't necessarily mean that the model is bad, it does suggest that there might be room for improvement.

2. I would have reservations using this model as this was trained on entire sample as opposed to the criteria (consumption below 1.90 USD/day). The indicators that be most predictive for this data would be different as the baseline markers for someone living in ultra-poverty vs poverty would be different.

## LASSO Regression
LASSO regression is a type of linear regression that adds a penalty term to the loss function that is proportional to the sum of the absolute values of the coefficients. This penalty term encourages the coefficients to be small, and can even force some of them to zero. This makes LASSO regression a good choice for feature selection, as it can automatically identify the most important variables in the model.

There are a few complications that can make LASSO regression less suitable for cases where there are many categorical variables. One complication is that the penalty term can become very large when there are many variables, which can make it difficult to find a good solution. We do have a few columns with many possible values.

The possible solutions to avoid this problem, would be to use models like RandomForest. RandomForest is a type of ensemble model that combines the predictions of many decision trees. Random Forest doesn't overfit and is able to handle multiple categorical variables well.

## One-Hot Encoding
One-hot encoding is a way of representing categorical variables as numerical variables. This is done by creating a new binary variable for each possible value of the categorical variable. For example, we encoded variables like language, religion, marital_status, highest_education, can_read_language_fam. If a categorical variable has 3 possible values, then one-hot encoding would create 3 new binary variables.

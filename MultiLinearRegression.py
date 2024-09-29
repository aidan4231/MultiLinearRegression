#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 16:33:16 2024

@author: rausch
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
dataset = pd.read_csv('/Users/rausch/Desktop/Python/ai/Datasets/housingprices.csv')

parking = dataset.pop('parking')
dataset.insert(5, 'parking', parking)

y = dataset.iloc[:,0:1].values
x = dataset.iloc[:,1:].values
label_encoder = LabelEncoder()

i = 5
for i in range(12):
    x[:,i] = label_encoder.fit_transform(x[:,i])  
    
# Plot a graph for each attribute against the housing price
for column in dataset.columns[1:]:  # Exclude the target variable (housing price)
    plt.scatter(dataset[column], dataset['price'])
    plt.title(f'{column} vs. Price')
    plt.xlabel(column)
    plt.ylabel('Price')
    plt.show()
#X = dataset.drop(['bedrooms'])



# Multiple Linear Regression Model
X = dataset.drop(['price'], axis=1) 
X = X.drop(['bedrooms'], axis = 1) # Exclude the 'bedroom' column
y = dataset['price']


# Apply label encoding to specified columns
label_columns = [ 'basement', 'hotwaterheating', 'mainroad','guestroom','airconditioning','prefarea', 'furnishingstatus']
label_encoder = LabelEncoder()

for column in label_columns:
    X[column] = label_encoder.fit_transform(X[column])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Build a multiple linear regression model using all available attributes
model = LinearRegression()
model.fit(X_train, y_train)

# Print the summary statistics of the model
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Backward Elimination
#X_train = sm.add_constant(X_train)  # Add a constant term for the intercept
model_sm = sm.OLS(y_train, X_train).fit()

print(model_sm.summary())

p_values = model_sm.pvalues
print("P-values for each attribute:")
print(p_values)


p_values = model_sm.pvalues
print("P-values for each attribute:")
print(p_values.apply(lambda x: round(x, 4)))  # Round p-values to 4 decimal places
# Handle categorical variables using one-hot encoding
#X = pd.get_dummies(X, columns=['prefarea', 'furnishingstatus'], drop_first=True)



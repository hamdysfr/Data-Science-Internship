# Making the imports
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
plt.rcParams['figure.figsize'] = (12.0, 9.0)

# Preprocessing Input data
data = pd.read_csv('Tsk01Data.csv')
X = data['math']
Y = data['cs']
plt.scatter(X, Y)
plt.show()
# Building the model
m = 0
c = 0

L = 0.0001  # The learning Rate


n = float(len(X))  # Number of elements in X
Y_pred = m * X + c  # The current predicted value of Y
cost=(1 / n) * sum(pow((Y - Y_pred),2))
D_m = (-2 / n) * sum(X * (Y - Y_pred))  # Derivative wrt m
D_c = (-2 / n) * sum(Y - Y_pred)  # Derivative wrt c
m = m - L * D_m  # Update m
c = c - L * D_c  # Update c
# Performing Gradient Descent
while True :
    cost_previous=cost
    Y_pred = m * X + c  # The current predicted value of Y
    cost=(1 / n) * sum(pow((Y - Y_pred),2))
    D_m = (-2 / n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2 / n) * sum(Y - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
    if (math.isclose(cost, cost_previous, rel_tol=1e-20)):
        break

print("coefficient is:",m,"and intercept is:", c)
# Making predictions
Y_pred = m*X + c

plt.scatter(X, Y)
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red') # predicted
plt.show()
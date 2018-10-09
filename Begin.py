import csv

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn import datasets, linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics.regression import mean_squared_error
from sklearn.model_selection import train_test_split

data = datasets.load_boston()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)
lm = linear_model.LinearRegression()
sgd_reg = GradientBoostingRegressor(n_estimators=100, learning_rate=0.5)
model = sgd_reg.fit(X_train, y_train)
predictions = model.predict(X_test)
plt.scatter(y_test, predictions)
print(predictions)
print(np.array(y_test))
print(mean_squared_error(predictions, y_test))
print("AAAAA")
print(X_test[:, 1])
plt.plot([0, 50], [0, 50], "--k")
plt.xticks(())
plt.yticks(())
plt.xlabel("True values ")
plt.ylabel("PRedictions")
plt.tight_layout()
plt.show()

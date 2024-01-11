from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict
import numpy as np
import matplotlib.pyplot as plt

def LinearReg(X, y):
  from sklearn.linear_model import LinearRegression

  clf = LinearRegression()

  # Do cross validation
  scores = cross_val_score(clf, X, y, cv=5, scoring='neg_mean_squared_error')
  scores = np.sqrt(-scores)
  print(scores)
  mean = scores.mean()
  print(f'The mean score is: {mean}')
  std = np.std(scores)
  print(f'The Standard Deviation is {std}')
  predictions = cross_val_predict(clf, X, y)

  residuals = y - predictions
  plt.scatter(y, residuals)
  plt.xlabel('Predicted Values')
  plt.ylabel('Residuals')
  plt.axhline(y=0, color='r', linestyle='-')
  plt.title('Residual Plot')
  plt.show()
  model = clf.fit(X, y)
  plt.bar(X.columns, model.coef_)
  plt.xticks(X.columns, rotation=90)
  plt.yticks(model.coef_)
  plt.show()

def RForest(X, y):
  from sklearn.ensemble import RandomForestRegressor

  # Create a random forest regressor
  regressor = RandomForestRegressor(n_estimators=100, max_depth=10)
  scores = cross_val_score(regressor, X, y, cv=5, scoring='neg_mean_squared_error')
  scores = np.sqrt(-scores)
  # Print the scores

  print(scores)
  mean = scores.mean()
  print(f'The mean score is: {mean}')
  std = np.std(scores)
  print(f'The Standard Deviation is {std}')
  predictions = cross_val_predict(regressor, X, y)

  residuals = y - predictions
  plt.scatter(y, residuals)
  plt.xlabel('Predicted Values')
  plt.ylabel('Residuals')
  plt.axhline(y=0, color='r', linestyle='-')
  plt.title('Residual Plot')
  plt.show()

def BuildMLP(X, y):
  from sklearn.neural_network import MLPRegressor

  # Create the regressor
  regressor = MLPRegressor(hidden_layer_sizes=(50, 10, 10), activation='relu', solver='adam')
  scores = cross_val_score(regressor, X, y, cv=5, scoring='neg_mean_squared_error')
  scores = np.sqrt(-scores)
  # Print the scores
  print(scores)
  mean = scores.mean()
  print(f'The mean score is: {mean}')
  std = np.std(scores)
  print(f'The Standard Deviation is {std}')
  predictions = cross_val_predict(regressor, X, y)

  residuals = y - predictions
  plt.scatter(y, residuals)
  plt.xlabel('Predicted Values')
  plt.ylabel('Residuals')
  plt.axhline(y=0, color='r', linestyle='-')
  plt.title('Residual Plot')
  plt.show()

def NeighborReg(X, y):
  from sklearn.neighbors import KNeighborsRegressor

  # Create a KNeighborsRegressor object
  knn = KNeighborsRegressor(n_neighbors=8)
  scores = cross_val_score(knn, X, y, cv=5, scoring='neg_mean_squared_error')

  scores = np.sqrt(-scores)
  # Print the scores
  print(scores)
  mean = scores.mean()
  print(f'The mean score is: {mean}')
  std = np.std(scores)
  print(f'The Standard Deviation is {std}')
  predictions = cross_val_predict(knn, X, y)

  residuals = y - predictions
  plt.scatter(y, residuals)
  plt.xlabel('Predicted Values')
  plt.ylabel('Residuals')
  plt.axhline(y=0, color='r', linestyle='-')
  plt.title('Residual Plot')
  plt.show()

def buildLasso(X, y):
  from sklearn.linear_model import Lasso

  lasso = Lasso()
  scores = cross_val_score(lasso, X, y, cv=5, scoring='neg_mean_squared_error')

  scores = np.sqrt(-scores)
  # Print the scores
  print(scores)
  mean = scores.mean()
  print(f'The mean score is: {mean}')
  std = np.std(scores)
  print(f'The Standard Deviation is {std}')
  predictions = cross_val_predict(lasso, X, y)

  residuals = y - predictions
  plt.scatter(y, residuals)
  plt.xlabel('Predicted Values')
  plt.ylabel('Residuals')
  plt.axhline(y=0, color='r', linestyle='-')
  plt.title('Residual Plot')
  plt.show()
  model = lasso.fit(X, y)
  plt.bar(X.columns, model.coef_)
  plt.xticks(X.columns, rotation=90)
  plt.yticks(model.coef_)
  plt.show()

def buildRidge(X, y):
  from sklearn.linear_model import Ridge
  ridge = Ridge()
  scores = cross_val_score(ridge, X, y, cv=5, scoring='neg_mean_squared_error')

  scores = np.sqrt(-scores)
  # Print the scores
  print(scores)
  mean = scores.mean()
  print(f'The mean score is: {mean}')
  std = np.std(scores)
  print(f'The Standard Deviation is {std}')
  predictions = cross_val_predict(ridge, X, y)

  residuals = y - predictions
  plt.scatter(y, residuals)
  plt.xlabel('Predicted Values')
  plt.ylabel('Residuals')
  plt.axhline(y=0, color='r', linestyle='-')
  plt.title('Residual Plot')
  plt.show()
  model = ridge.fit(X, y)
  plt.bar(X.columns, model.coef_)
  plt.xticks(X.columns, rotation=90)
  plt.yticks(model.coef_)
  plt.show()

def trainAllModels(df):
  y = df['fare_amount']
  print(y)
  X = df.copy().drop('fare_amount', axis=1)
  print('All models scored using root mean squared error.')
  print("Scores from Linear Regression Model:")
  LinearReg(X,y)
  print("Lasso Scores:")
  buildLasso(X, y)
  print("Ridge Scores:")
  buildRidge(X, y)
  print('Scores from 5 nearest neighbors:')
  NeighborReg(X, y)
  print("Scores from Random Forest:")
  RForest(X,y)
  print("MLP Scores:")
  BuildMLP(X, y)




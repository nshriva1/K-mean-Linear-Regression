from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as matplt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = load_boston()

matplt.figure(figsize=(4, 3))
matplt.hist(data.target)
matplt.xlabel('Price in $1000s')
matplt.ylabel('Count')
matplt.tight_layout()

for index, feature in enumerate(data.feature_names):
    matplt.figure(figsize=(10, 5))
    matplt.scatter(data.data[:, index], data.target)
    matplt.ylabel('Price', size=20)
    matplt.xlabel(feature, size=20)
    matplt.tight_layout()

# Linear Regression prediction
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

LRclf = LinearRegression()
LRclf.fit(X_train, y_train)
predicted_values = LRclf.predict(X_test)
expected_values = y_test

matplt.figure(figsize=(10, 5))
matplt.scatter(expected_values, predicted_values)
matplt.plot([0, 50], [0, 50], '--k')
matplt.axis('tight')
matplt.xlabel('True price in $1000s')
matplt.ylabel('Predicted price in $1000s')
matplt.tight_layout()

print("RMS error is  %r " % np.sqrt(np.mean((predicted_values - expected_values) ** 2)))

matplt.show()

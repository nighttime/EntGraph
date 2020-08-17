from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import pickle
from typing import *


fname = '2020-06-29_01.37-PREC0.85_thresholds.pkl'

with open('../eval_data/tf_results/{}'.format(fname), 'rb') as f:
	data = pickle.load(f)

# Assuming BU graphs have more levels

X = data['thresholds']['U->U'][:-1]
Y_full = data['thresholds']['B->U']
Y = np.zeros(X.shape)

pX = data['precisions']['U->U']
pY = data['precisions']['B->U']

print('Calculating...')
for i in range(X.size):
	y_idx = np.abs(pY - pX[i]).argmin()
	Y[i] = Y_full[y_idx]

ct = np.count_nonzero(Y == 1)
Y = Y[:-ct]
X = X[:-ct]

# X_feat = X.reshape(-1, 1)
X_feat = np.array([[x**p for p in range(1,4)] for x in X])

model = LinearRegression()
reg = model.fit(X_feat, Y)

rsq = reg.score(X_feat, Y)
coef = reg.coef_
inter = reg.intercept_

print('R-sq: {:.3f}\nCoef: {}\nIntercept: {}'.format(rsq, coef, inter))

Y_hat = model.predict(X_feat)
residuals = Y - Y_hat

# Residual Plot
# plt.hlines(0, 0, 1, colors='red')
# sns.lineplot(X, residuals)

# Model vs Actual
sns.lineplot(X, Y, marker='o')
sns.lineplot(X, Y_hat)

plt.show()
print()
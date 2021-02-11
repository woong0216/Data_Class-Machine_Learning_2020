# -*- coding: utf-8 -*-

# Use only following packages
import numpy as np
from scipy import stats
from sklearn.datasets import load_boston

def ftest(X,y):
    # X: inpute variables
    # y: target
    X=data.data
    n, p= X.shape
    X_data = np.concatenate((np.ones(n).reshape(-1,1), X), axis=1)
    X_T_X = np.matmul(X_data.T, X_data)
    beta = np.matmul(np.matmul(np.linalg.inv(X_T_X), X_data.T), y.reshape(-1, 1))
    y_hat = np.matmul(X_data, beta)

    SSE = sum((y.reshape(-1, 1) - y_hat)**2)
    SSR = sum((y_hat - np.mean(y_hat))**2)
    SST = SSR + SSE
    MSR = SSR / p
    MSE = SSE / (n - p - 1)
    f_value = MSR / MSE
    p_value = 1- stats.f.cdf(f_value, p, n - p - 1)

    print("----------------------------------------------------------------------------------------------------")
    print("Factor        SS            DF          MS               F-value             pr>F")
    print("Model", " ",SSR, "  ", p, "  ", MSR, "  ", f_value, "  ", p_value) 
    print("Error", " ",SSE, "  ", n - p - 1, "  ", MSE)
    print("----------------------------------------------------------------------------------------------------")
    print("Total", " ",SSE + SSR, "  ", p + n - p - 1)
    print("----------------------------------------------------------------------------------------------------")
    return 0

def ttest(X,y,varname=None):
    # X: inpute variables
    # y: target
    X=data.data
    name = np.append('const',data.feature_names)
    n, p= X.shape
    X_data = np.concatenate((np.ones(n).reshape(-1,1), X), axis=1)
    X_T_X = np.matmul(X_data.T, X_data)
    beta = np.matmul(np.matmul(np.linalg.inv(X_T_X), X_data.T), y.reshape(-1, 1))
    y_hat = np.matmul(X_data, beta)

    SSE = sum((y.reshape(-1, 1) - y_hat)**2)
    SSR = sum((y_hat - np.mean(y_hat))**2)
    SST = SSR + SSE
    MSR = SSR / p
    MSE = SSE / (n - p - 1)

    se_matrix = MSE*(np.linalg.inv(np.matmul(X_data.T,X_data)))
    se_matrix = np.diag(se_matrix)
    se = np.sqrt(se_matrix)

    t_value = []
    for i in range(len(se_matrix)):
        t_value.append((beta[i] / np.sqrt(se_matrix[i])))

    p_value = ((1 - stats.t.cdf(np.abs(np.array(t_value)), n - p - 1)) * 2)

    print("----------------------------------------------------------------------------------------------------")
    print("Variable         coef                     se                   t                   Pr>|t|")

    for i in range(0,14):
        print(name[i], "       ", beta[i], "     ", se[i], "     ", t_value[i], "     ", p_value[i])
    print("----------------------------------------------------------------------------------------------------")

# Do not change!
# load data
data=load_boston()
X=data.data
y=data.target

ftest(X,y)
ttest(X,y,varname=data.feature_names)

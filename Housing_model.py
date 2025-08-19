# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 11:32:53 2025

@author: mb5542
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
 
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
  
# Data import      
cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS',
        'RAD','TAX','PTRATIO','B','LSTAT','MEDV']
 
data = pd.read_csv(r"e:\dokumenty\github\Housing_data\housing.data",
                   sep=' +', engine='python', header=None, 
                   names=cols)

# Separation of explained and explanatory variables
X = data.drop('MEDV', axis=1)
y = data['MEDV'].values

# Data scaling
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Splitting data into train & test data
X_train, X_test, y_train, y_test = train_test_split(X, y)
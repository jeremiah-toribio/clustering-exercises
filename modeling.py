# My py files
import env
import os
import wrangle as w
# Ignore Warning
import warnings
warnings.filterwarnings("ignore")
# Array and Dataframes
import numpy as np
import pandas as pd
# Imputer
from sklearn.impute import SimpleImputer
# Evaluation: Visualization
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
# Evaluation: Statistical Analysis
from scipy import stats
# Modeling: Preprocessing
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, StandardScaler, RobustScaler
# Modeling: Feature Selection
from sklearn.feature_selection import SelectKBest, f_regression, RFE
# Modeling
from sklearn.linear_model import LinearRegression as lr
from sklearn.linear_model import LassoLars
from sklearn.linear_model import TweedieRegressor
    # Linear: Polynomial
from sklearn.preprocessing import PolynomialFeatures
# Modeling: Selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# Metrics
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score



def select_kbest(x_train_scaled,x_train, y_train, k=2):

    # parameters: f_regression stats test, give me all features - normally in
    f_selector = SelectKBest(f_regression, k=k)#k='all')
    # find the all X's correlated with y
    f_selector.fit(x_train_scaled, y_train)

    # boolean mask of whether the column was selected or not
    feature_mask = f_selector.get_support()

    # get list of top K features. 
    f_feature = x_train.iloc[:,feature_mask].columns.tolist()

    return f_feature
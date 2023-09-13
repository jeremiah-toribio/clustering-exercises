import env
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



def handle_missing_values(df, column_percent, row_percent):
    '''
    Values in 'column_percent' & 'row_percent' should be a decimal between 0-1
    this will indicate how much of the values or columns will be retained based on percentage of missing values.
    '''
    

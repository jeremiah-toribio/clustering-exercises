import env
import os
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


def zillow(database='zillow',user=env.user, password=env.password, host=env.host):
    '''
    Pulls data from mySql server 
    '''
    if os.path.isfile('zillow.csv'):
        # If csv file exists read in data from csv file.
        print('File exists, pulling from system.')
        zillow = pd.read_csv('zillow.csv', index_col=0)
    else:
        print('No file exists, extracting from MySQL.')
        query = ''' '''

        connection = f'mysql+pymysql://{user}:{password}@{host}/{database}'
        zillow = pd.read_sql(query, connection)

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
from sklearn.model_selection import train_test_split


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
        query = '''select ta.*, tb.logerror, tb.transactiondate, act.airconditioningdesc, ast.architecturalstyledesc, bct.buildingclassdesc, hst.heatingorsystemdesc,
                    plt.propertylandusedesc, st.storydesc, tct.typeconstructiondesc
                    from properties_2017 ta
                    inner join (select parcelid from properties_2017
                                group by parcelid having count(parcelid) = 1) as tt
                                on ta.parcelid = tt.parcelid
                    left join predictions_2017 as tb on ta.parcelid = tb.parcelid
                    left join airconditioningtype as act on ta.airconditioningtypeid = act.airconditioningtypeid
                    left join architecturalstyletype as ast on ta.architecturalstyletypeid = ast.architecturalstyletypeid
                    left join buildingclasstype as bct on ta.buildingclasstypeid = bct.buildingclasstypeid
                    left join heatingorsystemtype as hst on ta.heatingorsystemtypeid = hst.heatingorsystemtypeid
                    left join propertylandusetype as plt on ta.heatingorsystemtypeid = plt.propertylandusetypeid
                    left join storytype as st on ta.storytypeid = st.storytypeid
                    left join typeconstructiontype as tct on ta.typeconstructiontypeid = tct.typeconstructiontypeid
                    WHERE ta.bedroomcnt | ta.bathroomcnt != 0
                    AND ta.latitude & ta.longitude IS NOT NULL
                    AND ta.propertylandusetypeid = 261;'''

        connection = f'mysql+pymysql://{user}:{password}@{host}/{database}'
        zillow = pd.read_sql(query, connection)

        # dropping extra columns
        zillow.drop(columns=['id'],inplace=True)
        # cache data
        zillow.to_csv('zillow.csv')
    return zillow

def splitter(df, stratify=None):
    '''
    Returns
    Train, Validate, Test from SKLearn
    Sizes are 60% Train, 20% Validate, 20% Test
    '''
    train, test = train_test_split(df, test_size=.2, random_state=4343, stratify=stratify)

    train, validate = train_test_split(train, test_size=.2, random_state=4343, stratify=stratify)
    print(f'Dataframe: {df.shape}', '100%')
    print(f'Train: {train.shape}', '| ~60%')
    print(f'Validate: {validate.shape}', '| ~20%')
    print(f'Test: {test.shape}','| ~20%')

    return train, validate, test


def mall(database='mall_customers',user=env.user, password=env.password, host=env.host):
    '''
    Pulls data from mySql server 
    '''
    if os.path.isfile('mall.csv'):
        # If csv file exists read in data from csv file.
        print('File exists, pulling from system.')
        mall = pd.read_csv('mall.csv', index_col=0)
    else:
        print('No file exists, extracting from MySQL.')
        query = '''select * from customers;'''

        connection = f'mysql+pymysql://{user}:{password}@{host}/{database}'
        mall = pd.read_sql(query, connection)

        # cache data
        mall.to_csv('mall.csv')
    return mall
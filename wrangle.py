import env
import os
# Ignore Warning
import warnings
warnings.filterwarnings("ignore")
# Imputer
from sklearn.impute import SimpleImputer
# Evaluation: Visualization
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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
        query = '''select ta.*, ta.fips, tb.logerror, tb.transactiondate, act.airconditioningdesc, ast.architecturalstyledesc, bct.buildingclassdesc, hst.heatingorsystemdesc,
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
                    AND ta.propertylandusetypeid = 261
                    AND tb.logerror IS NOT NULL;'''

        connection = f'mysql+pymysql://{user}:{password}@{host}/{database}'
        zillow = pd.read_sql(query, connection)

        # Renaming fips and other columns
        zillow = zillow.rename(columns={'fips':'county', 'taxvaluedollarcnt':'tax_value','calculatedfinishedsquarefeet':'sq_feet'})
        # drop duplicate columns
        zillow = zillow.loc[:,~zillow.columns.duplicated()].copy()
        # assigning values to associated codes for fips/county
        zillow['county'] = zillow['county'].map({6037:'LA',6059:'Orange',6111:'Ventura'})

        # dropping duplicate houses -- on the latest transaction date to determine which to keep
        zillow = zillow.sort_values('transactiondate').drop_duplicates('parcelid',keep='last')
        # dropping extra columns
        zillow.drop(columns=['id','parcelid'],inplace=True)
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

def missing_by_col(df): 
    '''
    returns a single series of null values by column name
    '''
    return df.isnull().sum(axis=0)

def missing_by_row(df) -> pd.DataFrame:
    '''
    prints out a report of how many rows have a certain
    number of columns/fields missing both by count and proportion
    
    '''
    # get the number of missing elements by row (axis 1)
    count_missing = df.isnull().sum(axis=1)
    # get the ratio/percent of missing elements by row:
    percent_missing = round((df.isnull().sum(axis=1) / df.shape[1]) * 100)
    # make a df with those two series (same len as the original df)
    # reset the index because we want to count both things
    # under aggregation (because they will always be sononomous)
    # use a count function to grab the similar rows
    # print that dataframe as a report
    rows_df = pd.DataFrame({
    'num_cols_missing': count_missing,
    'percent_cols_missing': percent_missing
    }).reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).\
    count().reset_index().rename(columns={'index':'num_rows'})
    return rows_df

def report_outliers(df, k=1.5) -> None:
    '''
    report_outliers will print a subset of each continuous
    series in a dataframe (based on numeric quality and n>20)
    and will print out results of this analysis with the fences
    in places
    '''
    num_df = df.select_dtypes('number')
    for col in num_df:
        if len(num_df[col].value_counts()) > 20:
            lower_bound, upper_bound = get_fences(df,col, k=k)
            print(f'Outliers for Col {col}:')
            print('lower: ', lower_bound, 'upper: ', upper_bound)
            print(df[col][(
                df[col] > upper_bound) | (df[col] < lower_bound)])
            print('----------')

def get_fences(df, col, k=1.5) -> tuple:
    '''
    get fences will calculate the upper and lower fence
    based on the inner quartile range of a single Series
    
    return: lower_bound and upper_bound, two floats
    '''
    q3 = df[col].quantile(0.75)
    q1 = df[col].quantile(0.25)
    iqr = q3 - q1
    upper_bound = q3 + (k * iqr)
    lower_bound = q1 - (k * iqr)
    return lower_bound, upper_bound

def summarize(df, k=1.5) -> None:
    '''
    Summarize will take in a pandas DataFrame
    and print summary statistics:
    
    info
    shape
    outliers
    description
    missing data stats
    
    return: None (prints to console)
    '''
    # print info on the df
    print('=======================\n=====   SHAPE   =====\n=======================')
    print(df.shape)
    print('========================\n=====   INFO   =====\n========================')
    print(df.info())
    print('========================\n=====   DESCRIBE   =====\n========================')
    # print the description of the df, transpose, output markdown
    print(df.describe().T.to_markdown())
    print('==========================\n=====   DATA TYPES   =====\n==========================')
    # lets do that for categorical info as well
    # we will use select_dtypes to look at just Objects
    print(df.select_dtypes('O').describe().T.to_markdown())
    print('==============================\n=====   MISSING VALUES   =====\n==============================')
    print('==========================\n=====   BY COLUMNS   =====\n==========================')
    print(missing_by_col(df).to_markdown())
    print('=======================\n=====   BY ROWS   =====\n=======================')
    print(missing_by_row(df).to_markdown())
    print('========================\n=====   OUTLIERS   =====\n========================')
    print(report_outliers(df, k=k))
    print('================================\n=====   THAT IS ALL, BYE   =====\n================================')


def handle_missing_values(df, column_percent=.6, row_percent=.6):
    '''
    Values in 'column_percent' & 'row_percent' should be a decimal between 0-1
    this will indicate how much of the values or columns will be retained based on percentage of missing values.

    The higher the decimal the lower the threshold, vice versa. It will be the inverse of what is put -- (ex. 0.8 means values that have at least 80%
    of none nulls will not be dropped.)
    '''

    col_limit = int(len(df.columns) * column_percent)
    row_limit = int(len(df.columns) * row_percent)
    df.dropna(thresh=col_limit,axis=1,inplace=True)
    df.dropna(thresh=row_limit,axis=0,inplace=True)

    return df


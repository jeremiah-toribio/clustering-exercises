import env
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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
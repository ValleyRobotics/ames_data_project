# Import Statements
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (train_test_split, 
                                     RandomizedSearchCV, 
                                     cross_val_score,
                                     RepeatedStratifiedKFold
                                    )
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from scipy import stats
from sklearn.ensemble import (RandomForestRegressor, 
                              RandomForestClassifier
                             )
from sklearn.metrics import (accuracy_score, 
                             confusion_matrix, 
                             precision_score, 
                             recall_score, 
                             #roc_auc_score, 
                             #roc_curve, 
                             f1_score
                            )
from sklearn.preprocessing import (OneHotEncoder, 
                                   OrdinalEncoder
                                  )
from sklearn.compose import make_column_transformer
from pprint import pprint
from sklearn.pipeline import make_pipeline
from numpy import (mean, 
                   std
                  )
#from numpy import std
from sklearn.datasets import make_classification
from sklearn.tree import export_graphviz



# set display options
pd.set_option('display.max_columns', None)
%matplotlib inline


# Test Function
def hello_world():
    return 'Hello World'

# variables used to setup the data gathering and prep functions 
replace_na_col = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage',
                  'GarageQual', 'GarageCond', 'GarageType', 'GarageFinish', 'GarageYrBlt',
                  'GarageQual', 'BsmtExposure', 'BsmtQual', 'BsmtCond', 'BsmtFinType1',
                  'BsmtFinType2'
                 ]
fields_to_drop = ['LotFrontage', 'GarageYrBlt', 'Utilities', 'Street', 'GarageArea',
                      'YearRemodAdd', 'BsmtFinSF1', 'ScreenPorch', 'EnclosedPorch', 'Alley',
                      'Utilities', 'SaleType', '3SsnPorch', 'Exterior1st', 'Exterior2nd',
                      'Condition1', 'Condition2', 'PoolArea', 'Functional', 'RoofMatl',
                      'RoofStyle', 'Electrical', 'BsmtFinSF2', 'BsmtFinType1', 'BsmtFinType2',
                      'PoolQC', 'HeatingQC', 'BsmtExposure', 'PoolQC', 'BsmtUnfSF', 
                      'SaleCondition', 'LotConfig', 'TotalBsmtSF', 'MSSubClass', 'LowQualFinSF',
                      'LowQualFinSF', 'BsmtFullBath', 'BsmtHalfBath', 'WoodDeckSF', 'LandSlope',
                      'WoodDeckSF'
                     ]
fields_to_drop_sm = ['LotFrontage', 'GarageYrBlt']

# functions used for data collection and reporting
def unique_counts (df):
    ret = []
    ret = pd.DataFrame.from_records([(col, df[col].nunique(), df[col].dtype) for col in df.columns],
                          columns=['Column', 'Num_Unique', 'type']).sort_values(by=['Num_Unique'])
    return ret

def convert_to_cat(df, type_):
    unique_ = unique_counts(df)
    for col in unique_['Column'].loc[unique_['Num_Unique']<20]:
        if df[col].dtype == type_:
            #if unique_[col].loc[unique_['Num_Unique']<20]:
        #train_x[col] = pd.cut(train_x[col], bins=4, labels=[col+'_low', col+'_mid', col+'_high', col+'_very_high'])
            df[col] = df[col].astype('str').astype('category')
    for col in unique_['Column'].loc[unique_['Num_Unique']>=20]:
        if df[col].dtype == type_:
            df[col] = pd.qcut(df[col], 4, duplicates='drop')#bins=4, labels=['_low', '_mid', '_high', '_very_high'])
            df[col] = df[col].astype('str').astype('category')
    return df


def change_missing(df, col_=replace_na_col, val='NA'):
    df[col_] = df[col_].replace({np.nan: val})
    return df

def cond_condense_rooms(cond_):
        cond_ = int(cond_)
        if cond_ < 3:
            return 1
        if cond_ < 5:
            return 2
        if cond_ < 10:
            return 3
        else:
            return 0
        
def cond_tot_rooms(cond_):
    cond_ = int(cond_)
    if cond_ < 5:
        return 1
    if cond_ < 7:
        return 2
    if cond_ < 10:
        return 3
    else:
        return 4    

def qu_condense(cond_):
    if cond_ in ['Ex', 'Gd']:
        return 3
    if cond_ == 'TA':
        return 2
    if cond_ in ['Fa', 'Po']:
        return 1
    else:
        return 0

def cond_condenser(cond_):
    cond_ = int(cond_)
    if cond_ < 3:
        return 1
    elif cond_ < 6:
        return 2
    elif cond_ < 7:
        return 3
    elif cond_ <9:
        return 4
    elif cond_ < 11:
        return 5
    else:
        return 0

# create a setup function that gets the data in the structure I need
def data_load():
    ''' Basic data load and setup. Just loading one of the files - train.csv, splitting
    into a train_x and train_y. Bin some of the data and delete a bunch of the columns
    that after many iterations, didn't need to be included. All data becomes category data
    '''
    import pandas as pd
    train = pd.read_csv('~data/train.csv')
    train_x = train.iloc[:,:-1].drop('Id', axis=1)
    fields_ = ['MasVnrArea', 'Electrical', 'MSZoning', 'Functional', 'Utilities']
    
    val_ = []
    for f in (fields_):
        train_x[f] = train_x[f].fillna(train_x[f].mode()[0])
        
    train_y = train.iloc[:,-1]
    train_x = change_missing(train_x)
     
    # bin y
    bin_ = np.concatenate((np.linspace(25000, 500000, 7),np.array([600000, 800000]) ) )
    #bin_ = np.concatenate((np.linspace(25000, 450000, 9),np.array([500000, 550000, 600000, 800000]) ) )
    train_y = pd.cut(train_y, bins=bin_).astype('str').astype('category')
       
    for col_ in ['FireplaceQu', 'KitchenQual', 'BsmtCond', 'ExterCond', 'ExterCond',
                 'BsmtQual', 'GarageCond', 'GarageQual']:
        train_x[col_] = train_x[col_].apply(lambda x: qu_condense(x))  
    train_x['OverallCond'] = train_x['OverallCond'].apply(lambda x: cond_condenser(x))
    train_x['OverallQual'] = train_x['OverallQual'].apply(lambda x: cond_condenser(x))
    train_x['BedroomAbvGr'] = train_x['BedroomAbvGr'].apply(lambda x: cond_condense_rooms(x))
    train_x['TotRmsAbvGrd'] = train_x['TotRmsAbvGrd'].apply(lambda x: cond_tot_rooms(x))   
    for col in train_x:
        if train_x[col].dtype == 'object':
            train_x[col] = train_x[col].astype('str').astype('category')
#     for col in test_x:
#         if test_x[col].dtype == 'object':
#             test_x[col] = test_x[col].astype('str').astype('category')
    # bin the data
    train_x['LotShape'] = train_x['LotShape'].apply(lambda x: '1' if x == 'Reg' else '0')
    train_x = convert_to_cat(train_x, 'int64')
    train_x = convert_to_cat(train_x, 'float64')
    train_x = train_x.drop(fields_to_drop, axis=1)
    return train_x, train_y

def encode_and_bind(original_dataframe, features_to_encode):
    dummies = pd.get_dummies(original_dataframe[features_to_encode])
    res = pd.concat([dummies, original_dataframe], axis=1)
    res = res.drop(features_to_encode, axis=1)
    return(res)

def by_cat_dist_plot(data_column, name_):
    ax=data_column.value_counts().sort_index().plot(kind = 'bar',  
                                                  figsize=(9,6))
    ax.set_title(name_, fontname='Comic Sans MS', fontsize=20)
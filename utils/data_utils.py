import numpy as np
import pandas as pd
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from utils.data_preprocessor import *

import warnings
warnings.filterwarnings('ignore')

#################################################################################################################################

def load_CRSP():
    '''
        Loads CRSP data from GKX (2020) database.
    '''

    # Load GKX (2020) database
    characs = pd.read_csv('data/raw/characs.csv')
    characs['date'] = characs['date'].apply(str).str.slice(stop=6)
    characs['date'] = pd.to_datetime(characs['date'], format = '%Y%m')

    # Get prc, mve, mom1m
    CRSP_characs = characs[['permno', 'date', 'prc', 'mve', 'mom1m']]
    CRSP_characs = CRSP_characs.rename(columns={'prc': 'Price', 'mve': 'Size', 'mom1m' : 'STreversal'})

    del characs

    return CRSP_characs

def load_welch_and_goyal():
    '''
        Loads Goyal database and construct variables defined in Welch and Goyal (2008).
        Here, I consider only 8 variables used in GKX (2020).
    '''

    welch_goyal = pd.read_csv('data/raw/welch_goyal_raw.csv')
    welch_goyal['yyyymm'] = pd.to_datetime(welch_goyal['yyyymm'], format = '%Y%m')
    welch_goyal = welch_goyal.rename(columns={'yyyymm':'date', 'b/m' : 'bm'})
    welch_goyal = welch_goyal.drop('csp', axis=1)
    welch_goyal = welch_goyal.dropna()

    # Only create variables used in GKX (2020)
    welch_goyal['dp'] = np.log(welch_goyal['D12'])-np.log(welch_goyal['Index'])
    welch_goyal['ep'] = np.log(welch_goyal['E12'])-np.log(welch_goyal['Index'])
    welch_goyal['tms'] = welch_goyal['lty'] - welch_goyal['Rfree']
    welch_goyal['dfy'] = welch_goyal['BAA'] - welch_goyal['AAA']

    welch_goyal_var = welch_goyal[['date', 'dp', 'ep', 'bm', 'ntis', 'tbl', 'tms', 'dfy' ,'svar']]

    # Get risk free rate to calculate excess return
    risk_free = welch_goyal[['date', 'Rfree']]

    del welch_goyal

    return welch_goyal_var, risk_free


#################################################################################################################################

def load_info():
    '''
        Load info table of firm-level and macroeconomic predictors
    '''

    firm_info = pd.read_csv('data/info/SignalDoc.csv')
    firm_info = firm_info[firm_info['Cat.Signal']=='Predictor'].drop('Cat.Signal', axis=1) # Only leave predictor infos 
    macro_info = pd.read_csv('data/info/FRED_MD.csv')

    return firm_info, macro_info

def load_data():
    '''
        Loads dataset by reading csv files.
        Below are the csv files loaded in this function.

         File Name                          | Paper
        ------------------------------------|------------------------------
        1. signed_predictors_dl.wide.csv    | Chen and Zimmermann (2021)
        2. characs.csv                      | Gu, Kelly, Xiu (2020)
        3. welch_goyal_raw.csv              | Welch and Goyal (2008)
        4. FRED_MD.csv                      | McCracken and Ng (2016)
        ------------------------------------|------------------------------

        File 1 and 2 are aggregated to create firm_predictors table, while
        file 3 and 4 are aggregated to create macro_predictors table.

        Return: 
        firm_predictors  (pd.DataFrame): Firm-level characteristics
        macro_predictors (pd.DataFrame): Monthly macroeconomic variables

    '''

    # Load firm-level characteristics from (1) Chen and Zimmermann (2021) and (2) Gu, Kelly, Xiu (2020)
    signals = pd.read_csv('data/raw/signed_predictors_dl_wide.csv')
    signals['yyyymm'] = pd.to_datetime(signals['yyyymm'], format = '%Y%m')
    signals = signals.rename(columns={'yyyymm':'date'})

    # Load information of firm-level characteristics
    firm_info, _ = load_info()

    # Shift quarterly and annual factors to adjust provision delay
    index = ['permno', 'date']
    quarter = list(firm_info[firm_info['Frequency']=='Quarterly']['Acronym'].values)
    annual  = list(firm_info[firm_info['Frequency']=='Annual']['Acronym'].values)

    signals = pd.concat([signals[index], 
                     signals.drop(index+quarter+annual, axis=1), 
                     signals.groupby('permno')[quarter].shift(3), 
                     signals.groupby('permno')[annual].shift(6)], 
                     axis = 1)

    # Get [Price, Size, Short term reversal (mom1m)] from GKX (2020)
    CRSP_characs = load_CRSP()

    # Merge Chen and Zimmermann (2021) and GKX (2020)
    firm_predictors = pd.DataFrame.merge(signals, CRSP_characs, how = 'inner', on = ['date', 'permno'])
    # Drop rows before 1988.01 where some analysts consensus data are missing
    firm_predictors = firm_predictors[firm_predictors['date']>=pd.to_datetime('1988-01-01')]

    del signals
    del CRSP_characs

    # Load macroeconomic predictors from (1) McCracken and Ng (2016) and (2) Welch and Goyal (2008)
    FRED_MD = pd.read_csv('data/raw/FRED_MD.csv')
    FRED_MD['date'] = pd.to_datetime(FRED_MD['date'])

    # Get Welch and Goyal (2008) variables used in GKX (2020)
    welch_goyal, risk_free = load_welch_and_goyal()

    # Merge monthly macro variables
    macro_predictors = pd.DataFrame.merge(FRED_MD, welch_goyal, how='inner', on='date')

    # Forward fill missing values and drop missing columns
    macro_predictors = macro_predictors.ffill().dropna(axis=1)


    del FRED_MD
    del welch_goyal

    return firm_predictors, macro_predictors, risk_free

def get_data(horizon_r = 1, horizon_c = 1):
    '''
        Preprocess raw dataset and return input and target that can be directly employed to learning stage.
        Info files are loaded and used in preprocessing.

        Preprocessing process is summarized as below.
        1. Drop inappropriate firms and columns
        2. Fill missing values with last observations
        3. Drop sparse columns
        4. Drop inappropriate firms
        5. Fill missing values without last observations
        6. Create target variable and rank normalize

    '''

    # Load data and info table
    firm_predictors, macro_predictors, risk_free = load_data()
    firm_info, macro_info = load_info()

    print('Data loaded...')

    ###########################################################################################################
    # 1. Drop inappropriate firms and columns

    # Drop columns with short sample history
    short_col = list(firm_info[firm_info['SampleStartYear']>=1988]['Acronym'].values)
    # Drop columns of consensus variables that will not be used throughout the empirical research
    unused_col = ['CredRatDG', 'DownRecomm', 'UpRecomm']

    # Drop columns
    firm_predictors = firm_predictors.drop(short_col,  axis=1)
    firm_predictors = firm_predictors.drop(unused_col,  axis=1)
    firm_info = firm_info[firm_info['Acronym'].isin(firm_predictors.columns)]

    # Drop firms without enough of analyst data
    before = len(firm_predictors['permno'].unique())
    firm_predictors = drop_firms(firm_predictors, firm_info, method='mean', thr=0.5)
    after = len(firm_predictors['permno'].unique())

    print('Inappropriate firms dropped...')
    print('From ', before, ' firms, total ', after, ' selected')
    print('# total samples: ', len(firm_predictors))

    ###########################################################################################################
    # 2. Fill missing values with last observations

    firm_predictors = fill_firm_na(firm_predictors, firm_info, method='time')

    print('Missing value filled...')

    ###########################################################################################################
    # 3. Drop sparse columns

    # Count missing samples
    firm_na, analyst_na = get_na_summary(firm_predictors, firm_info)

    # Drop analyst columns with too sparse samples
    sparse_col_firm = list(firm_na[firm_na['missing rate']>0.2].index)
    sparse_col_analyst = list(analyst_na[analyst_na['missing rate']>0.2].index)

    # Drop columns
    firm_predictors = firm_predictors.drop(sparse_col_firm, axis=1)
    firm_predictors = firm_predictors.drop(sparse_col_analyst, axis=1)

    before = len(firm_info['Acronym'])-2
    firm_info = firm_info[firm_info['Acronym'].isin(firm_predictors.columns)]
    after = len(firm_info['Acronym'])-2

    print('Sparse column dropped...')
    print('From ', before, ' features, total ', after, ' selected')

    ###########################################################################################################
    # 4. Drop inappropriate firms

    before = len(firm_predictors['permno'].unique())
    firm_predictors = drop_firms(firm_predictors, firm_info, method='max', thr=0.8)
    after = len(firm_predictors['permno'].unique())

    print('Inappropriate firms dropped...')
    print('From ', before, ' firms, total ', after, ' selected')
    print('# total samples: ', len(firm_predictors))

    ###########################################################################################################
    # 5. Fill missing values without last observations

    firm_predictors = fill_firm_na(firm_predictors, firm_info, method='cross')

    print('Missing value filled...')

    ###########################################################################################################
    # 6. Create target variable and normalize input

    # Create series of target variable (asset returns)
    firm_predictors = create_return(firm_predictors, risk_free, horizon_r)
    # Shift consensus variables
    firm_predictors = shift_consensus(firm_predictors, firm_info, horizon_c)
    firm_predictors.dropna(inplace=True)
    
    # Split input and output
    target = firm_predictors[['permno', 'date', 'Return']]
    firm_predictors = firm_predictors.drop('Return', axis=1)

    # Rank normalize
    firm_predictors = rank_norm(firm_predictors)
    # Min-max normalize
    macro_predictors = min_max_norm(macro_predictors)
    # Merge firm and macro factors
    input = pd.DataFrame.merge(firm_predictors, macro_predictors, how='inner', on='date')

    print('\n\n--------------------------------------------------------------------------------------------')
    print('Data preprocessing completed!')
    print('Samples: ', len(input))
    print('Columns: ', len(input.columns)-2)
    print('Firms: ', len(input['permno'].unique()))

    return input, target

#################################################################################################################################

class CB_Dataset(Dataset):
    def __init__(self, input, target, info):

        # Get column names for each dataset
        self.concept_col = list(info[info['Cat.Data'] == 'Analyst']['Acronym'].values)

        # Define dataset in numpy array form
        self.input_data = input.drop(self.concept_col + ['permno', 'date'], axis=1).values
        self.concept_data = input[self.concept_col].values
        self.output_data = target.drop(['permno', 'date'], axis=1).values

    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.input_data[idx]).type(torch.float)
        c = torch.from_numpy(self.concept_data[idx]).type(torch.float)
        y = torch.from_numpy(self.output_data[idx]).type(torch.float)

        return x, c, y


def create_dataloaders(input, target, info, train_date, valid_date, test_date, batch_size):
    '''
        Function that creates dataloaders for train, validation, and test sets
    '''

    # Split data into train, validation, and test data
    train_input  = input[
        input['date'] < pd.to_datetime(train_date)
        ]
    train_target = target[
        target['date'] < pd.to_datetime(train_date)
        ]

    valid_input  = input[
        (input['date'] >= pd.to_datetime(train_date)) & 
        (input['date'] < pd.to_datetime(valid_date))
        ]
    valid_target = target[
        (target['date'] >= pd.to_datetime(train_date)) &
        (target['date'] < pd.to_datetime(valid_date))
        ]

    test_input   = input[
        (input['date'] >= pd.to_datetime(valid_date))  & 
        (input['date'] < pd.to_datetime(test_date))
        ]
    test_target  = target[
        (target['date'] >= pd.to_datetime(valid_date)) &
        (target['date'] < pd.to_datetime(test_date))
        ]

    # Create datasets for train, validation, and test sets
    train_dataset = CB_Dataset(train_input, train_target, info)
    valid_dataset = CB_Dataset(valid_input, valid_target, info)
    test_dataset  = CB_Dataset(test_input , test_target , info)

    g = torch.Generator()
    g.manual_seed(0)

    # Create dataloaders for train, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last = True, shuffle = False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, drop_last = True, shuffle = False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, drop_last = False, shuffle = False)

    # Return date and permno of test dataset for portfolio performance analysis
    test_index = test_target[['date', 'permno']]

    return train_loader, valid_loader, test_loader, test_index
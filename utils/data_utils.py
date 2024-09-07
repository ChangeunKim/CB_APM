import numpy as np
import pandas as pd
import random
import torch
from sklearn.decomposition import PCA

from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from utils.data_preprocessor import *
from models.train import train_autoencoder
from models.networks import Autoencoder

from sklearn.model_selection import KFold
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

def get_data(horizon_r = 1):
    '''
        Preprocess raw dataset and return input and target that can be directly employed to learning stage.
        Info files are loaded and used in preprocessing.

        Preprocessing process is summarized as below.
        1. Drop inappropriate firms and columns
        2. Fill missing values with last observations
        3. Drop sparse columns
        4. Drop inappropriate firms
        5. Fill missing values without last observations
        6. Create target variable
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
    # 6. Create target variable

    # Create series of target variable (asset returns)
    firm_predictors = create_return(firm_predictors, risk_free, horizon_r)
    firm_predictors.dropna(inplace=True)
    
    # Split input and output
    target = firm_predictors[['permno', 'date', 'Return']]
    firm_predictors = firm_predictors.drop('Return', axis=1)

    # Merge firm and macro factors without normalization
    input = pd.DataFrame.merge(firm_predictors, macro_predictors, how='inner', on='date')

    print('\n\n--------------------------------------------------------------------------------------------')
    print('Data preprocessing completed!')
    print('Samples: ', len(input))
    print('Columns: ', len(input.columns)-2)
    print('Firms: ', len(input['permno'].unique()))

    return input, target

#################################################################################################################################

def create_datasets(input_data, target_data, info, train_date, valid_date, test_date):
    """Create train, validation, and test datasets with proper normalization"""
    # Split data by dates
    train_mask = (input_data['date'] >= train_date) & (input_data['date'] < valid_date)
    valid_mask = (input_data['date'] >= valid_date) & (input_data['date'] < test_date)
    test_mask = (input_data['date'] >= test_date)
    
    # Split data into train, validation, and test sets
    train_data = input_data[train_mask].copy()
    valid_data = input_data[valid_mask].copy()
    test_data = input_data[test_mask].copy()
    
    # Get columns
    index_cols = ['date', 'permno']
    firm_cols = list(info[info['Cat.Data'] != 'Analyst']['Acronym'].values)
    concept_cols = list(info[info['Cat.Data'] == 'Analyst']['Acronym'].values)
    
    # Macro columns are those that are not in firm_cols and not index columns
    macro_cols = [col for col in input_data.columns 
                 if col not in firm_cols 
                 and col not in concept_cols
                 and col not in index_cols]
  


    # Extract macro features from each dataset
    train_macro = train_data[['date'] + macro_cols].drop_duplicates('date').copy()
    valid_macro = valid_data[['date'] + macro_cols].drop_duplicates('date').copy()
    test_macro = test_data[['date'] + macro_cols].drop_duplicates('date').copy()
    
    # Apply min-max normalization to macro features using min_max_norm function
    # First, normalize the training data
    train_macro_norm = min_max_norm(train_macro)
    
    # Then normalize validation and test data using training data statistics
    valid_macro_norm = min_max_norm(valid_macro, 
                                   min_dict={col: train_macro[col].min() for col in macro_cols},
                                   max_dict={col: train_macro[col].max() for col in macro_cols})
    test_macro_norm = min_max_norm(test_macro, 
                                  min_dict={col: train_macro[col].min() for col in macro_cols},
                                  max_dict={col: train_macro[col].max() for col in macro_cols})
    
    
    # Choose columns from input data
    train_firm = train_data[index_cols + firm_cols]
    valid_firm = valid_data[index_cols + firm_cols]
    test_firm = test_data[index_cols + firm_cols]
    
    # Apply rank normalization to firm-level features using the rank_norm function
    train_firm = rank_norm(train_firm)
    valid_firm = rank_norm(valid_firm)
    test_firm = rank_norm(test_firm)

    # Choose columns from input data
    train_concept = train_data[index_cols + concept_cols]
    valid_concept = valid_data[index_cols + concept_cols]
    test_concept = test_data[index_cols + concept_cols]
    
    # Apply rank normalization to firm-level features using the rank_norm function
    train_concept_norm = rank_norm(train_concept)
    valid_concept_norm = rank_norm(valid_concept)
    test_concept_norm = rank_norm(test_concept)
    
    # Merge normalized firm data with normalized macro data
    train_data_norm = pd.merge(train_firm, train_macro_norm, on='date')
    valid_data_norm = pd.merge(valid_firm, valid_macro_norm, on='date')
    test_data_norm = pd.merge(test_firm, test_macro_norm, on='date')
    
    # Create input datasets
    X_train = train_data_norm.drop(['date', 'permno'], axis=1)
    X_valid = valid_data_norm.drop(['date', 'permno'], axis=1)
    X_test = test_data_norm.drop(['date', 'permno'], axis=1)
    
    # Create concept datasets
    X_train_concept = train_concept_norm.drop(['date', 'permno'], axis=1)
    X_valid_concept = valid_concept_norm.drop(['date', 'permno'], axis=1)
    X_test_concept = test_concept_norm.drop(['date', 'permno'], axis=1)
    
    # Create target datasets
    y_train = target_data[train_mask]['Return'].values
    y_valid = target_data[valid_mask]['Return'].values
    y_test = target_data[test_mask]['Return'].values
    
    # Save test indices for portfolio formation
    test_index = input_data[test_mask][['date', 'permno']]
    
    return (X_train, X_valid, X_test, 
            X_train_concept, X_valid_concept, X_test_concept,
            y_train, y_valid, y_test, test_index)

class CB_Dataset(Dataset):
    def __init__(self, input_data, target_data, info):

        # Get column names for each dataset
        self.concept_col = list(info[info['Cat.Data'] == 'Analyst']['Acronym'].values)

        # Define dataset in numpy array form
        self.input_data = input_data.drop(self.concept_col + ['permno', 'date'], axis=1).values
        self.concept_data = input_data[self.concept_col].values
        self.output_data = target_data.drop(['permno', 'date'], axis=1).values
    
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.input_data[idx]).type(torch.float)
        c = torch.from_numpy(self.concept_data[idx]).type(torch.float)
        y = torch.from_numpy(self.output_data[idx]).type(torch.float)

        return x, c, y

def embed_macro_autoencoder(train_macro, valid_macro, test_macro, macro_cols, model_path):
    """
    The embed_macro_autoencoder function trains an autoencoder on the provided macro datasets (train_macro, valid_macro, test_macro)
    using the specified macro columns (macro_cols). It encodes the macro features into a lower-dimensional latent space for each dataset.

    Args:
        train_macro (pd.DataFrame): Training macro dataframe (must include 'date' column).
        valid_macro (pd.DataFrame): Validation macro dataframe (must include 'date' column).
        test_macro (pd.DataFrame): Test macro dataframe (must include 'date' column).
        macro_cols (list): List of macro variable column names to be embedded.
        model_path (str or None): If given, load the trained autoencoder from this path.

    Returns:
        train_macro_emb (pd.DataFrame): Embedded training macro dataframe (with 'date' column).
        valid_macro_emb (pd.DataFrame): Embedded validation macro dataframe (with 'date' column).
        test_macro_emb (pd.DataFrame): Embedded test macro dataframe (with 'date' column).
        model (Autoencoder): Trained autoencoder model.
    """

    autoencoder_train_data = torch.from_numpy(
        train_macro.drop(columns=["date"]).values
    ).float()
    autoencoder_valid_data = torch.from_numpy(
        valid_macro.drop(columns=["date"]).values
    ).float()
    autoencoder_test_data  = torch.from_numpy(
        test_macro.drop(columns=["date"]).values
    ).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(TensorDataset(autoencoder_train_data),
                            batch_size=16, shuffle=False,  drop_last=False)
    valid_loader = DataLoader(TensorDataset(autoencoder_valid_data),
                            batch_size=16, shuffle=False, drop_last=False)

    input_dim = len(macro_cols)
    if model_path == None:
        model = train_autoencoder(train_loader, valid_loader, input_dim, device)
    else:
        # Load trained model weights
        model = Autoencoder(input_dim=input_dim, latent_dim=32).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

    #Encode the trained autoencoder to the macro predictors
    with torch.no_grad():
        encoded_train = model.encoder(autoencoder_train_data.to(device)).detach().cpu().numpy()  
        encoded_valid = model.encoder(autoencoder_valid_data.to(device)).detach().cpu().numpy() 
        encoded_test  = model.encoder(autoencoder_test_data .to(device)).detach().cpu().numpy()  
    #Convert the encoded data to dataframe and add date column

    train_macro_emb = pd.DataFrame(encoded_train, columns=[f'macro{i}' for i in range(encoded_train.shape[1])])
    train_macro_emb['date'] = train_macro['date'].values                           # (single assignment is enough)

    valid_macro_emb = pd.DataFrame(encoded_valid, columns=[f'macro{i}' for i in range(encoded_valid.shape[1])])
    valid_macro_emb['date'] = valid_macro['date'].values

    test_macro_emb  = pd.DataFrame(encoded_test , columns=[f'macro{i}' for i in range(encoded_test .shape[1])])
    test_macro_emb['date']  = test_macro['date'].values

    return train_macro_emb, valid_macro_emb, test_macro_emb, model


def embed_macro_pca(train_macro, valid_macro, test_macro, n_components=32):
    '''
        Applies PCA to macroeconomic predictors for dimensionality reduction.

        Args:
            train_macro (pd.DataFrame): Training macroeconomic data with 'date' column.
            valid_macro (pd.DataFrame): Validation macroeconomic data with 'date' column.
            test_macro (pd.DataFrame): Test macroeconomic data with 'date' column.
            macro_cols (list): List of macroeconomic predictor column names to use.
            n_components (int): Number of principal components to retain.

        Returns:
            train_macro_emb (pd.DataFrame): PCA-embedded training macro data with 'date' column.
            valid_macro_emb (pd.DataFrame): PCA-embedded validation macro data with 'date' column.
            test_macro_emb (pd.DataFrame): PCA-embedded test macro data with 'date' column.
            pca (PCA): Fitted PCA object.
    '''

    pca = PCA(n_components=n_components)
    # Fit PCA on training data (drop date)
    train_data = train_macro.drop(['date'], axis=1).values
    train_pca = pca.fit_transform(train_data)

    # Transform validation and test sets
    valid_data = valid_macro.drop(['date'], axis=1).values
    test_data  = test_macro.drop(['date'], axis=1).values
    valid_pca = pca.transform(valid_data)
    test_pca  = pca.transform(test_data)

    # Convert the PCA outputs to DataFrames and add date column
    cols_pca = [f'macro{i}' for i in range(n_components)]

    train_macro_emb = pd.DataFrame(train_pca, columns=cols_pca)
    train_macro_emb['date'] = train_macro['date'].values

    valid_macro_emb = pd.DataFrame(valid_pca, columns=cols_pca)
    valid_macro_emb['date'] = valid_macro['date'].values

    test_macro_emb = pd.DataFrame(test_pca, columns=cols_pca)
    test_macro_emb['date'] = test_macro['date'].values

    return train_macro_emb, valid_macro_emb, test_macro_emb, pca


def create_dataloaders(input_data, target_data, info, train_date, valid_date, test_date, batch_size, 
                       embedding_method='autoencoder', model_path=None):
    '''
        Function that creates dataloaders for train, validation, and test sets
        embedding_method: 'autoencoder', 'pca', 'none'
    '''

    if all(date is None for date in [train_date, valid_date, test_date]):
        raise ValueError("Date input should be given.")

    # Split data into train, validation, and test data
    train_input = input_data[
        input_data['date'] < pd.to_datetime(train_date)
        ].copy()
    train_target = target_data[
        target_data['date'] < pd.to_datetime(train_date)
        ]

    valid_input = input_data[
        (input_data['date'] >= pd.to_datetime(train_date)) & 
        (input_data['date'] < pd.to_datetime(valid_date))
        ].copy()
    valid_target = target_data[
        (target_data['date'] >= pd.to_datetime(train_date)) &
        (target_data['date'] < pd.to_datetime(valid_date))
        ]

    test_input = input_data[
        (input_data['date'] >= pd.to_datetime(valid_date)) & 
        (input_data['date'] < pd.to_datetime(test_date))
        ].copy()
    test_target = target_data[
        (target_data['date'] >= pd.to_datetime(valid_date)) &
        (target_data['date'] < pd.to_datetime(test_date))
        ]

    # Get columns
    index_cols = ['date', 'permno']
    firm_cols = list(info[info['Cat.Data'] != 'Analyst']['Acronym'].values)
    concept_cols = list(info[info['Cat.Data'] == 'Analyst']['Acronym'].values)
    
    # Macro columns are those that are not in firm_cols and not index columns
    macro_cols = [col for col in input_data.columns 
                 if col not in firm_cols 
                 and col not in concept_cols
                 and col not in index_cols]

    # Extract macro features from each dataset
    train_macro = train_input[['date'] + macro_cols].drop_duplicates('date').copy()
    valid_macro = valid_input[['date'] + macro_cols].drop_duplicates('date').copy()
    test_macro = test_input[['date'] + macro_cols].drop_duplicates('date').copy()
    model = None

    # Apply min-max normalization to macro features using min_max_norm function
    # First, normalize the training data
    train_macro_norm = min_max_norm(train_macro)
    
    # Then normalize validation and test data using training data statistics
    valid_macro_norm = min_max_norm(valid_macro, 
                                   min_dict={col: train_macro[col].min() for col in macro_cols},
                                   max_dict={col: train_macro[col].max() for col in macro_cols})
    test_macro_norm = min_max_norm(test_macro, 
                                  min_dict={col: train_macro[col].min() for col in macro_cols},
                                  max_dict={col: train_macro[col].max() for col in macro_cols})
    
    # --- Choose embedding method ---
    if embedding_method == 'autoencoder':
        train_macro_emb, valid_macro_emb, test_macro_emb, model = embed_macro_autoencoder(
            train_macro_norm, valid_macro_norm, test_macro_norm, macro_cols, model_path)
    elif embedding_method == 'pca':
        train_macro_emb, valid_macro_emb, test_macro_emb, model = embed_macro_pca(
            train_macro_norm, valid_macro_norm, test_macro_norm, macro_cols, n_components=32)
    elif embedding_method == 'none':
        train_macro_emb = train_macro_norm.copy()
        valid_macro_emb = valid_macro_norm.copy()
        test_macro_emb = test_macro_norm.copy()
        model = None
    else:
        raise ValueError("embedding_method must be one of 'autoencoder', 'pca', or 'none'.")

    # Choose columns from input data
    train_firm = train_input[index_cols + firm_cols]
    valid_firm = valid_input[index_cols + firm_cols]
    test_firm = test_input[index_cols + firm_cols]
    
    # Apply rank normalization to firm-level features using the rank_norm function
    train_firm = rank_norm(train_firm)
    valid_firm = rank_norm(valid_firm)
    test_firm = rank_norm(test_firm)

    # Choose columns from input data for concept features
    train_concept = train_input[index_cols + concept_cols]
    valid_concept = valid_input[index_cols + concept_cols]
    test_concept = test_input[index_cols + concept_cols]
    
    # Apply rank normalization to concept features using the min_max_norm function
    train_concept_norm = rank_norm(train_concept)
    valid_concept_norm = rank_norm(valid_concept)
    test_concept_norm = rank_norm(test_concept)
    
    # Merge normalized firm data with normalized macro data
    train_input = pd.merge(train_firm, train_macro_emb, on='date')
    valid_input = pd.merge(valid_firm, valid_macro_emb, on='date')
    test_input = pd.merge(test_firm, test_macro_emb, on='date')

    # Add concept columns back to input data for CB_Dataset
    train_input = pd.merge(train_input, train_concept_norm, on=['date', 'permno'])
    valid_input = pd.merge(valid_input, valid_concept_norm, on=['date', 'permno'])
    test_input = pd.merge(test_input, test_concept_norm, on=['date', 'permno'])

    # Create datasets for train, validation, and test sets
    train_dataset = CB_Dataset(train_input, train_target, info)
    valid_dataset = CB_Dataset(valid_input, valid_target, info)
    test_dataset = CB_Dataset(test_input, test_target, info)

    g = torch.Generator()
    g.manual_seed(0)

    # Create dataloaders for train, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, drop_last=False, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False, shuffle=False)

    # Return date and permno of test dataset for portfolio performance analysis
    test_index = test_target[['date', 'permno']]

    return train_loader, valid_loader, test_loader, test_index, model
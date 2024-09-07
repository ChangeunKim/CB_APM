import numpy as np
import pandas as pd
import torch

def drop_firms(firm_predictors, firm_info, method, thr):
    '''
        Drop firms (permno) if it doesn't contain full set of analysts consensus data.
    '''
    assert method in ['mean', 'max']

    # Drop firms without enough of analyst data
    to_drop_firm = list()
    analyst_col = list(firm_info[firm_info['Cat.Data']=='Analyst']['Acronym'].values)

    for firm in firm_predictors.groupby('permno'):
        # If some analyst data doesn't exist at all, drop that firm
        if method == 'mean':
            if (len(firm[1]))*thr <= (firm[1][analyst_col].isna().sum().mean()):
                to_drop_firm.append(firm[0])
        if method == 'max':
            if (len(firm[1]))*thr <= (firm[1][analyst_col].isna().sum().max()):
                to_drop_firm.append(firm[0])

    firm_predictors = firm_predictors[~firm_predictors['permno'].isin(to_drop_firm)]

    return firm_predictors

def create_return(firm_predictors, risk_free, horizon_r):
    '''
        Create asset return column
    '''

    # Calculate stock return
    firm_predictors['LogPrice'] = np.log(firm_predictors['Price'])
    firm_predictors['Return'] = firm_predictors.groupby('permno')['LogPrice'].diff(horizon_r)

    # Calculate risk-free rate
    risk_free['Rfree'] = (1 + risk_free['Rfree'])**horizon_r - 1
    risk_free['Rfree'] = np.log(1 + risk_free['Rfree']) # convert to log return

    # Calculate excess return
    firm_predictors = pd.merge(firm_predictors, risk_free, on='date')
    firm_predictors['Return'] = firm_predictors['Return'] - firm_predictors['Rfree']
    
    # Calculate n-month ahead return
    firm_predictors['Return'] =  firm_predictors['Return'].shift(-horizon_r)

    return firm_predictors

def get_na_summary(firm_predictors, firm_info):
    '''
        Counts missing values of firm characteristics and analysts data
    '''
    firm_na = pd.DataFrame(columns = firm_predictors['permno'].unique())
    analyst_na = pd.DataFrame(columns = firm_predictors['permno'].unique())

    firm_col = list(firm_info[firm_info['Cat.Data']!='Analyst']['Acronym'].values)
    analyst_col = list(firm_info[firm_info['Cat.Data']=='Analyst']['Acronym'].values)

    for firm in firm_predictors.groupby('permno'):
        temp1 = firm[1][firm_col].isna().sum()
        temp1['total'] = len(firm[1])
        temp2 = firm[1][analyst_col].isna().sum()
        temp2['total'] = len(firm[1])
        firm_na[firm[0]] = temp1
        analyst_na[firm[0]] = temp2

        del temp1
        del temp2

    firm_na['mean'] = firm_na.mean(axis=1)
    analyst_na['mean'] = analyst_na.mean(axis=1)
    firm_na['missing rate'] = firm_na['mean'][0:-1]/firm_na['mean'][-1]
    analyst_na['missing rate'] = analyst_na['mean'][0:-1]/analyst_na['mean'][-1]

    firm_period =firm_info[firm_info['Cat.Data']!='Analyst'][['Acronym', 'Frequency']].set_index('Acronym')
    analyst_period =firm_info[firm_info['Cat.Data']=='Analyst'][['Acronym', 'Frequency']].set_index('Acronym')
    firm_na = pd.DataFrame.merge(firm_na, firm_period, left_index=True, right_index=True, how = 'left')
    analyst_na = pd.DataFrame.merge(analyst_na, analyst_period, left_index=True, right_index=True, how = 'left')

    return firm_na, analyst_na

def fill_firm_na(firm_predictors, firm_info, method):
    '''
        Fill missing malues in firm characteristics
    '''
    assert method in ['time', 'cross']
    
    if method == 'time':
        index = ['permno', 'date']
        firm_predictors_index = firm_predictors[index]
        firm_predictors_index.index = range(len(firm_predictors_index))

        # Firm characteristics except analyst consensus
        # Not Change/Growth: ffill
        ffill_col = list(firm_info[(firm_info['Cat.Data']!='Analyst') & (firm_info['Cat.Change']==0)]['Acronym'].values)
        firm_predictors_ffill = firm_predictors.groupby('permno')[ffill_col].apply(lambda x: x.ffill())
        firm_predictors_ffill.index = range(len(firm_predictors_ffill))

        # Change/Growth: time-series mean
        ts_mean_col = list(firm_info[(firm_info['Cat.Data']!='Analyst') & (firm_info['Cat.Change']==1)]['Acronym'].values)
        firm_predictors_ts_mean = firm_predictors.groupby('permno')[ts_mean_col].apply(
            lambda x: x.fillna(x.rolling(24, min_periods=1).mean())
            )
        firm_predictors_ts_mean.index = range(len(firm_predictors_ts_mean))

        # Analyst consensus: linear interpolation
        analyst_col = list(firm_info[firm_info['Cat.Data']=='Analyst']['Acronym'].values)
        firm_predictors_analyst = firm_predictors.groupby('permno')[analyst_col].apply(
            lambda x: x.interpolate(method='linear', limit_direction='forward')
            )
        firm_predictors_analyst.index = range(len(firm_predictors_analyst))

        # Merge na filled dataframes
        firm_predictors_filled = pd.concat([firm_predictors_index, firm_predictors_ffill, firm_predictors_ts_mean, firm_predictors_analyst], axis=1)
        del(firm_predictors, firm_predictors_index, firm_predictors_ffill, firm_predictors_ts_mean, firm_predictors_analyst)

    if method == 'cross':
        # Firm characteristics except analyst consensus: cross-sectional mean
        analyst_col = list(firm_info[firm_info['Cat.Data']=='Analyst']['Acronym'].values)
        firm_col = firm_predictors.columns[~firm_predictors.columns.isin(analyst_col)]

        firm_predictors_firm = firm_predictors[firm_col].groupby('date').apply(lambda x: x.fillna(x.mean()))
        if isinstance(firm_predictors_firm.index, pd.MultiIndex):
            firm_predictors_firm.index = firm_predictors_firm.index.droplevel()
        firm_predictors_firm = firm_predictors_firm.sort_index()

        firm_predictors_analyst = firm_predictors[analyst_col]

        # Rows with missing analyst consensus data are dropped
        firm_predictors_filled = pd.concat([firm_predictors_firm, firm_predictors_analyst], axis=1).dropna()
        del(firm_predictors, firm_predictors_firm, firm_predictors_analyst)

    return firm_predictors_filled

def rank_norm(firm_predictors):
    # Check if input is a PyTorch tensor
    if isinstance(firm_predictors, torch.Tensor):
        # For tensor input, apply rank normalization across features for each sample
        # Convert to numpy for easier processing
        data_np = firm_predictors.detach().cpu().numpy()
        
        # Apply rank normalization to each sample (row)
        normalized_data = np.zeros_like(data_np)
        for i in range(data_np.shape[0]):
            # Rank normalize each row (sample)
            ranks = np.argsort(np.argsort(data_np[i]))
            normalized_data[i] = (ranks / (len(ranks) - 1)) * 2 - 1
        
        # Convert back to tensor with same device and dtype
        return torch.from_numpy(normalized_data).to(firm_predictors.device).to(firm_predictors.dtype)
    
    # Original pandas DataFrame processing
    index = ['permno', 'date']
    firm_predictors_index = firm_predictors[index]

    firm_predictors_firm = firm_predictors.groupby('date').apply(lambda x: x.drop(index, axis=1).rank(pct=True) * 2 - 1)
    if isinstance(firm_predictors_firm.index, pd.MultiIndex):
        firm_predictors_firm.index = firm_predictors_firm.index.droplevel()
    firm_predictors_firm = firm_predictors_firm.sort_index()

    firm_predictors = pd.concat([firm_predictors_index, firm_predictors_firm], axis = 1)

    return firm_predictors

def min_max_norm(macro_predictors, columns=None, min_dict=None, max_dict=None):
    result = macro_predictors.copy()

    for predictor in macro_predictors.columns:
        if predictor != 'date' and predictor != 'permno':
            # Find the minimum and maximum values for each feature
            if min_dict is not None and max_dict is not None:
                min_val = min_dict[predictor]
                max_val = max_dict[predictor]
            else:
                min_val = result[predictor].min()
                max_val = result[predictor].max()
            
            # Apply min-max normalization to each feature
            if max_val > min_val:  # Avoid division by zero
                result[predictor] = (result[predictor] - min_val) / (max_val - min_val)
                # Clip values to [0, 1] for out-of-range values
                result[predictor] = result[predictor].clip(0, 1)

    return result
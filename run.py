import argparse
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import torch
import os

from utils.data_utils import create_dataloaders

from models.train import train, set_random_seed
from models.test import test
from models.metrics import evaluate

def run(input, target, info, config, path):
    # Define device context manager
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define expanding windows
    train_dates = ['2011-01-01', '2012-01-01', '2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01']
    valid_dates = ['2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01']
    test_dates  = ['2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01']

    # Initialize test results
    actual_concept = np.array([[]]*9)
    actual_target = np.array([])
    forecast_concept = np.array([[]]*9)
    forecast_target = np.array([])

    index = pd.DataFrame()
    score_table = pd.DataFrame()


    for train_date, valid_date, test_date in tqdm(zip(train_dates, valid_dates, test_dates), desc='Expanding window training...'):
        # Define dataloaders
        train_loader, valid_loader, test_loader, test_index = create_dataloaders(input, target, info, train_date, valid_date, test_date, batch_size = config['batch_size'])
        
        # Train model and get forecast results with test scores
        ensemble = config['ensemble']
        models = list()
        for i in tqdm(range(ensemble), desc='Ensemble model training...'):
            # Fix random seeds for reproducibility
            set_random_seed(i)    
            model = train(config, train_loader, valid_loader, device, False)
            model_path = os.path.join(path, train_date + 'model_' + str(i) + '.pt')
            torch.save(model.state_dict(), model_path)
            models.append(model)

        # Get model inference on test dataset
        c, r, c_hat, r_hat = test(models, test_loader, config['ensemble'], device)
        del models
        
        # Evaluate test results for each window
        print('\n------------------------------------------------------------------------------------')
        print(f'Testing window {valid_date}-{test_date}')
        scores = evaluate(c, r, c_hat, r_hat, info)

        score_table[test_date] = scores
        del scores

        actual_concept = np.append(actual_concept, c, axis=1)
        actual_target = np.append(actual_target, r)
        forecast_concept = np.append(forecast_concept, c_hat, axis=1)
        forecast_target = np.append(forecast_target, r_hat)
        
        index = pd.concat([index, test_index], ignore_index=True)

    # Evaluate test results of entire window
    print('\n------------------------------------------------------------------------------------')
    print('Testing entire window')
    scores = evaluate(actual_concept, actual_target, forecast_concept, forecast_target, info)

    score_table['Whole periods'] = scores
    del scores

    # Create dataframes of concepts and excess returns
    index.columns = ['date', 'permno']

    analyst_col = info[info['Cat.Data'] == 'Analyst']['LongDescription'].values
    actual_concept = pd.DataFrame(columns = analyst_col, data = actual_concept.T)
    forecast_concept = pd.DataFrame(columns = analyst_col, data = forecast_concept.T)

    actual_target = pd.DataFrame(columns = ['Return'], data = actual_target.T)
    forecast_target = pd.DataFrame(columns = ['Return'], data = forecast_target.T)

    # Merge with permno and date
    actual_concept = pd.concat([index, actual_concept], axis=1, ignore_index=True)
    forecast_concept = pd.concat([index, forecast_concept], axis=1, ignore_index=True)
    actual_target = pd.concat([index, actual_target], axis=1, ignore_index=True)
    forecast_target = pd.concat([index, forecast_target], axis=1, ignore_index=True)

    
    output = {
        'actual_concept': actual_concept,
        'forecast_concept': forecast_concept,
        'actual_target': actual_target,
        'forecast_target': forecast_target
    }

    return score_table, output


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--horizon", dest="horizon", action="store", type=int)
    parser.add_argument("-w", "--weight",  dest="weight",  action="store", type=float)
    args = parser.parse_args()

    # Define save path
    if args.horizon:
        file_name = 'predict_' + str(args.horizon) + 'months_' + str(args.weight)
    else:
        file_name = 'approx_' + str(args.weight)
    # Model save path
    path = os.path.join('checkpoints', file_name)
    if not os.path.exists(path):
        os.makedirs(path)

    # Define networks and hyperparameters
    config = {
        # model
        'input_size': 240, 
        'concept_hidden_sizes': [40, 20],
        'final_hidden_sizes': [],
        'concept_output_size': 9,
        'final_output_size': 1,

        # learning
        'lr': 0.001,
        'epochs': 100,
        'early_stopping_patience': 5, 
        'scheduling_patience': 2,
        'scheduling_factor': 0.2, 
        'batch_size': 5000,
        'ensemble': 10,

        # Regularization
        'weight_decay': 0.005,
        'clip_value': 1,

        # weight lambda
        'weight_lambda': args.weight
    }

    # Load data
    if args.horizon:
        input = pd.read_csv('data/input_predict_3_' + str(args.horizon) +'month.csv')
        target = pd.read_csv('data/target_predict_3_' + str(args.horizon) +'month.csv')
    else:
        input = pd.read_csv('data/input_approx_1month.csv')
        target = pd.read_csv('data/target_approx_1month.csv')
    input['date']  = pd.to_datetime(input['date'])
    target['date'] = pd.to_datetime(target['date'])

    # Load info
    signal_info = pd.read_csv('data/info/SignalDoc.csv')
    info = signal_info[signal_info['Acronym'].isin(input.columns)]

    # Run CB-APM

    print('\n------------------------------------------------------------------------------------')
    print(f'Training lambda={str(args.weight)}')
            
    score, output = run(input, target, info, config, path)

    # Save training results
    score.to_csv('results/' + file_name + '.csv')

    with open('results/' + file_name + '.pickle', 'wb') as fw:
        pickle.dump(output, fw)

if __name__=="__main__":
    main()
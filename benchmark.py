import argparse
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import os
import joblib
from joblib import dump as joblib_dump
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
joblib.parallel_backend('loky', n_jobs=1)

from utils.data_utils import create_datasets
from models.train import train_benchmarks
from models.metrics import evaluate_benchmarks
from models.test import test_benchmark

def run_benchmarks(input_data, target_data, info, horizon, tune=False, n_trials=50):
    """Run benchmark models with expanding window"""
    # Define expanding windows
    train_dates = ['2011-01-01', '2012-01-01', '2013-01-01', '2014-01-01', '2015-01-01', 
                  '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01']
    valid_dates = ['2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01', 
                  '2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01']
    test_dates = ['2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', 
                 '2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01']
    
    all_predictions = {}
    all_metrics = {}
    test_indices = pd.DataFrame()
    
    for train_date, valid_date, test_date in tqdm(zip(train_dates, valid_dates, test_dates), 
                                                 desc='Running benchmark models...'):
        # Create datasets
        datasets = create_datasets(input_data, target_data, info, train_date, valid_date, test_date)
        (X_train, X_valid, X_test, 
         X_train_concept, X_valid_concept, X_test_concept,
         y_train, y_valid, y_test, test_index) = datasets
        
        # Train models with input data
        input_models = train_benchmarks(X_train, X_valid, y_train, y_valid, "input", tune, n_trials)
        
        # Train models with concept data
        concept_models = train_benchmarks(X_train_concept, X_valid_concept, y_train, y_valid, "concept", tune, n_trials)
        
        # Get predictions
        input_preds = test_benchmark(input_models, X_test)
        concept_preds = test_benchmark(concept_models, X_test_concept)
        
        # Evaluate models
        input_metrics = evaluate_benchmarks(input_models, X_test, y_test)
        concept_metrics = evaluate_benchmarks(concept_models, X_test_concept, y_test)
        
        # Store results
        all_predictions[test_date] = {**input_preds, **concept_preds}
        all_metrics[test_date] = {**input_metrics, **concept_metrics}
        test_indices = pd.concat([test_indices, test_index])
    
    return all_predictions, all_metrics, test_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--horizon", dest="horizon", action="store", type=int,
                        help="Prediction horizon in months (e.g., 1, 3, 6, 12)")
    parser.add_argument("-t", "--tune", dest="tune", action="store_true", 
                        help="Enable hyperparameter tuning")
    parser.add_argument("-n", "--trials", dest="trials", action="store", type=int, default=50, 
                        help="Number of trials for hyperparameter tuning")
    args = parser.parse_args()
    
    # Load data
    input_data = pd.read_csv(f'data/input_{args.horizon}month.csv')
    target_data = pd.read_csv(f'data/target_{args.horizon}month.csv')
    input_data['date'] = pd.to_datetime(input_data['date'])
    target_data['date'] = pd.to_datetime(target_data['date'])
    
    # Load info
    signal_info = pd.read_csv('data/info/SignalDoc.csv')
    info = signal_info[signal_info['Acronym'].isin(input_data.columns)]

    # Run benchmarks
    predictions, metrics, test_indices = run_benchmarks(
        input_data, target_data, info, args.horizon, 
        tune=args.tune, n_trials=args.trials
    )
    
    # Create a single DataFrame for R² values
    all_dates = sorted(metrics.keys())
    r2_df = pd.DataFrame(columns=all_dates + ['Whole periods'])
    
    # Format results for each model type
    for model_type in ['OLS', 'PLS', 'PCR', 'ElasticNet', 'GLM', 'RF', 'GBRT']:
        for data_type in ['input', 'concept']:
            model_name = f"{model_type}_{data_type}"
            file_name = f"{args.horizon}month_{model_name}"
            
            # Create output dictionary similar to run.py format
            output = {
                'actual_target': pd.DataFrame({
                    'date': test_indices['date'],
                    'permno': test_indices['permno'],
                    'actual': target_data[target_data['date'].isin(test_indices['date'])]['Return'].values
                }),
                'forecast_target': pd.DataFrame({
                    'date': test_indices['date'],
                    'permno': test_indices['permno'],
                    'forecast': predictions[list(predictions.keys())[-1]][model_name]
                })
            }
            
            # Extract R² values for each time period
            r2_values = {}
            for date in all_dates:
                r2_values[date] = metrics[date][model_name]['R2']
            
            # Add row to DataFrame
            r2_df.loc[model_name] = pd.Series(r2_values)
            
            # Save predictions and actual values
            with open(f'results/{file_name}.pickle', 'wb') as f:
                pickle.dump(output, f)
            
            # Save model objects in a separate directory
            model_dir = os.path.join('checkpoints', 'benchmarks', file_name)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                
            for date in predictions.keys():
                model_path = os.path.join(model_dir, f'model_{date}.joblib')
                joblib_dump(predictions[date][model_name], model_path)
    
    # Add average R² for the whole period
    r2_df['Whole periods'] = r2_df.mean(axis=1)
    
    # Save R² values to a single CSV file
    r2_df.to_csv(f'results/{args.horizon}month_benchmark.csv')

if __name__ == "__main__":
    main()
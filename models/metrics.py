import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def r2_score(actual_returns, forecast_returns):
    """
    Compute the R-squared (coefficient of determination) for stock return forecasts.
    
    Parameters:
        actual_returns (array-like): Array of actual stock returns.
        forecast_returns (array-like): Array of forecasted stock returns.
        
    Returns:
        float: The R-squared value.
    """
    total_sum_squares = np.sum(actual_returns ** 2)
    residual_sum_squares = np.sum((actual_returns - forecast_returns) ** 2)
    r_squared = 1 - (residual_sum_squares / total_sum_squares)
    return round(r_squared * 100, 2) 

def evaluate_benchmarks(models, X_test, y_test):
    """Evaluate models and return R² metrics"""
    metrics = {}
    
    # Standardize data for PLS and PCR
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test)
    
    for name, model in models.items():
        # Get predictions
        if 'PCR' in name:
            X_test_pca = model['pca'].transform(X_test_scaled)
            y_pred = model['regressor'].predict(X_test_pca)
        elif 'PLS' in name:
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
            
        # Calculate R²
        r2 = r2_score(y_test, y_pred)
        metrics[name] = {'R2': r2}
    
    return metrics

def evaluate(actual_concept, actual_target, forecast_concept, forecast_target, info):
    
    score = pd.Series()

    analyst_col = info[info['Cat.Data'] == 'Analyst']['LongDescription'].values
    
    # Consensus R2 score
    for i, col in enumerate(analyst_col):
        score[col] = r2_score(actual_concept[i], forecast_concept[i])
    
    # Consensus average R2 score
    score['Consensus average'] = score.mean()
    
    # Return R2 score
    score['Return'] = r2_score(actual_target, forecast_target)

    return score

def evaluate_mse(actual_concept, actual_target, forecast_concept, forecast_target, info):
    
    score = pd.Series()

    analyst_col = info[info['Cat.Data'] == 'Analyst']['LongDescription'].values
    
    # Consensus MSE
    for i, col in enumerate(analyst_col):
        score[col] = mean_squared_error(actual_concept[i], forecast_concept[i])
    
    # Consensus average MSE
    score['Consensus average MSE'] = score.mean()
    
    # Return R2 score
    score['Return'] = mean_squared_error(actual_target, forecast_target)

    return score
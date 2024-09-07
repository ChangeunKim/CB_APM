import numpy as np
import pandas as pd

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
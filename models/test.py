from models.metrics import r2_score
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def test_benchmark(models, X_test):
    """Get predictions from benchmark models"""
    predictions = {}
    
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
            
        predictions[name] = y_pred
    
    return predictions

def test(models, data_loader, ensemble, device):
    """ Inference using concept bottleneck model """

    forecast_concept = []
    forecast_target = []
    actual_concept = []
    actual_target = []

    # Set models to evaluation mode
    for i in range(ensemble):
        models[i].eval()

    with torch.no_grad():
        for inputs, concepts, targets in data_loader:
            inputs = inputs.to(device)
            concepts = concepts.to(device)
            targets = targets.to(device)

            concept_output = torch.zeros(concepts.shape)
            final_output = torch.zeros(targets.shape)
            
            for i in range(ensemble):
                concept_temp, final_temp = models[i](inputs)
                concept_output = concept_output + concept_temp.cpu()
                final_output   = final_output + final_temp.cpu()
            concept_output = concept_output / ensemble
            final_output   = final_output   / ensemble
            forecast_concept.append(concept_output)
            forecast_target.append(final_output)
            actual_concept.append(concepts)
            actual_target.append(targets)

    forecast_concept = torch.cat(forecast_concept, dim=0).detach().cpu().numpy().T
    forecast_target = torch.cat(forecast_target, dim=0).detach().cpu().numpy()
    actual_concept = torch.cat(actual_concept, dim=0).detach().cpu().numpy().T
    actual_target = torch.cat(actual_target, dim=0).detach().cpu().numpy()


    return actual_concept, actual_target, forecast_concept, forecast_target

from models.metrics import r2_score
import torch
import pandas as pd

def test(models, test_loader, ensemble, device):
    # Test concept bottleneck model

    forecast_concept = []
    forecast_target = []
    actual_concept = []
    actual_target = []

    # Set models to evaluation mode
    for i in range(ensemble):
        models[i].eval()

    with torch.no_grad():
        for inputs, concepts, targets in test_loader:
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
        
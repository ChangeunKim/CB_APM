import numpy as np
import random

from models.networks import ConceptBottleneckModel
from models.model_utils import EarlyStopping, init_weights
from models.losses import JointLoss

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(config, train_loader, valid_loader, device, verbose):
    '''
        Train concept bottleneck model
    '''

    # Define model, loss function
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'], verbose=False)
    criterion = JointLoss(option='mse', weight_lambda=config['weight_lambda'])

    model = ConceptBottleneckModel(config['input_size'], config['concept_hidden_sizes'], config['concept_output_size'], config['final_hidden_sizes'], config['final_output_size']).to(device)
    # initialize weights and optimizers
    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']) # apply regularization via weight decay
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['scheduling_factor'], patience=config['scheduling_patience'])

    epochs = config['epochs']
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, concepts, targets in train_loader:
            inputs = inputs.to(device)
            concepts = concepts.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            concept_output, final_output = model(inputs)
            loss = criterion(final_output, targets, concept_output, concepts)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=config['clip_value'])
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, concepts, targets in valid_loader:
                inputs = inputs.to(device)
                concepts = concepts.to(device)
                targets = targets.to(device)
                
                concept_output, final_output = model(inputs)
                loss = criterion(final_output, targets, concept_output, concepts)
                val_loss += loss.item()
            val_loss /= len(valid_loader)
            scheduler.step(val_loss)

        # Print training and validation loss
        if verbose:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Check if validation loss has improved
        early_stopping(val_loss)
        
        # Check if early stopping criteria met
        if early_stopping.early_stop and verbose:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    return model
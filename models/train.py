import numpy as np
import random
import copy

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import TweedieRegressor  # for GLM
from sklearn.pipeline import Pipeline
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error

from models.networks import Autoencoder
from models.networks import ConceptBottleneckModel
from models.model_utils import EarlyStopping, init_weights
from models.losses import JointLoss
from config import get_config

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def tune_hyperparameters(X_train, X_valid, y_train, y_valid, model_type, n_trials=50):
    """Tune hyperparameters for a specific model type using Optuna"""
    
    # Standardize data for models that need it
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    
    def objective_pls(trial):
        n_components = trial.suggest_int('n_components', 2, min(20, X_train.shape[1]))
        model = PLSRegression(n_components=n_components)
        model.fit(X_train_scaled, y_train)
        return mean_squared_error(y_valid, model.predict(X_valid_scaled))
    
    def objective_pcr(trial):
        n_components = trial.suggest_int('n_components', 2, min(20, X_train.shape[1]))
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_valid_pca = pca.transform(X_valid_scaled)
        model = LinearRegression()
        model.fit(X_train_pca, y_train)
        return mean_squared_error(y_valid, model.predict(X_valid_pca))
    
    def objective_elasticnet(trial):
        alpha = trial.suggest_float('alpha', 1e-4, 0.1, log=True)
        model = ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=42)
        model.fit(X_train_scaled, y_train)
        return mean_squared_error(y_valid, model.predict(X_valid_scaled))
    
    def objective_glm(trial):
        alpha = trial.suggest_float('alpha', 1e-4, 0.1, log=True)
        n_knots = trial.suggest_int('n_knots', 2, 10)
        
        # Create pipeline with spline transformation and GLM
        model = Pipeline([
            ('spline', SplineTransformer(n_knots=n_knots, degree=3)),
            ('glm', TweedieRegressor(power=0, alpha=alpha))
        ])
        model.fit(X_train, y_train)
        return mean_squared_error(y_valid, model.predict(X_valid))
    
    def objective_rf(trial):
        max_depth = trial.suggest_int('max_depth', 1, 6)
        max_features = trial.suggest_float('max_features', 0.1, 1.0)
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=max_depth,
            max_features=max_features,
            random_state=42
        )
        model.fit(X_train, y_train)
        return mean_squared_error(y_valid, model.predict(X_valid))
    
    def objective_gbrt(trial):
        n_estimators = trial.suggest_int('n_estimators', 1, 1000)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
        max_depth = trial.suggest_int('max_depth', 1, 2)
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        return mean_squared_error(y_valid, model.predict(X_valid))
    
    # Select the appropriate objective function based on model type
    objective_functions = {
        'PLS': objective_pls,
        'PCR': objective_pcr,
        'ElasticNet': objective_elasticnet,
        'GLM': objective_glm,
        'RF': objective_rf,
        'GBRT': objective_gbrt
    }
    
    # Create and run the study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective_functions[model_type], n_trials=n_trials)
    
    return study.best_params

def train_benchmarks(X_train, X_valid, y_train, y_valid, data_type="input", tune=False, n_trials=50):
    """Train various benchmark models with optional hyperparameter tuning"""
    models = {}
    
    # Standardize the data for PLS and PCR
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    
    # Train OLS
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    models[f'OLS_{data_type}'] = ols
    
    # Train PLS
    if tune:
        pls_params = tune_hyperparameters(X_train, X_valid, y_train, y_valid, 'PLS', n_trials)
        n_components = pls_params.get('n_components', 5)
    else:
        n_components = min(5, X_train.shape[1])
    
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train_scaled, y_train)
    models[f'PLS_{data_type}'] = pls
    
    # Train PCR
    if tune:
        pcr_params = tune_hyperparameters(X_train, X_valid, y_train, y_valid, 'PCR', n_trials)
        n_components_pcr = pcr_params.get('n_components', 5)
    else:
        n_components_pcr = min(5, X_train.shape[1])
    
    pca = PCA(n_components=n_components_pcr)
    X_train_pca = pca.fit_transform(X_train_scaled)
    pcr = LinearRegression()
    pcr.fit(X_train_pca, y_train)
    models[f'PCR_{data_type}'] = {'pca': pca, 'regressor': pcr}
    
    # Train Elastic Net
    if tune:
        enet_params = tune_hyperparameters(X_train, X_valid, y_train, y_valid, 'ElasticNet', n_trials)
        alpha = enet_params.get('alpha', 0.01)
    else:
        alpha = 0.01
    
    enet = ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=42)
    enet.fit(X_train_scaled, y_train)
    models[f'ElasticNet_{data_type}'] = enet
    
    # Train GLM (Tweedie regression with splines)
    if tune:
        glm_params = tune_hyperparameters(X_train, X_valid, y_train, y_valid, 'GLM', n_trials)
        alpha_glm = glm_params.get('alpha', 0.01)
    else:
        alpha_glm = 0.01
    
    glm = Pipeline([
        ('spline', SplineTransformer(n_knots=3, degree=3)),
        ('glm', TweedieRegressor(power=0, alpha=alpha_glm))
    ])
    glm.fit(X_train, y_train)
    models[f'GLM_{data_type}'] = glm
    
    # Train Random Forest
    if tune:
        rf_params = tune_hyperparameters(X_train, X_valid, y_train, y_valid, 'RF', n_trials)
        max_depth_rf = rf_params.get('max_depth', 3)
        max_features = rf_params.get('max_features', 0.5)
    else:
        max_depth_rf = 3
        max_features = 0.5
    
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=max_depth_rf,
        max_features=max_features,
        random_state=42
    )
    rf.fit(X_train, y_train)
    models[f'RF_{data_type}'] = rf
    
    # Train GBRT
    if tune:
        gbrt_params = tune_hyperparameters(X_train, X_valid, y_train, y_valid, 'GBRT', n_trials)
        n_estimators_gbrt = gbrt_params.get('n_estimators', 100)
        learning_rate = gbrt_params.get('learning_rate', 0.1)
        max_depth_gbrt = gbrt_params.get('max_depth', 2)
    else:
        n_estimators_gbrt = 100
        learning_rate = 0.1
        max_depth_gbrt = 2
    
    gbrt = GradientBoostingRegressor(
        n_estimators=n_estimators_gbrt,
        learning_rate=learning_rate,
        max_depth=max_depth_gbrt,
        random_state=42
    )
    gbrt.fit(X_train, y_train)
    models[f'GBRT_{data_type}'] = gbrt
    
    return models

def train_autoencoder(train_loader, valid_loader, input_dim, device):
    '''
        Train autoencoder for macro embedding
    '''

    # ------------------------------------------------------------------
    # Training loop with true early-stopping & best-model snapshot
    # ------------------------------------------------------------------

    #train an auto encoder for macro predictors
    autoencoder = Autoencoder(input_dim=input_dim, latent_dim=32)
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # ── 0) data loaders ───────────────────────────────────────────────

    #   train_loader, valid_loader

    # ── 1) early-stopping & run settings ───────────────────────────────
    config = get_config('autoencoder', model='autoencoder')
    num_runs   = config['num_runs']
    num_epochs = config['num_epochs']
    patience   = config['patience']
    min_delta  = config['min_delta']
    print_every = config['print_every']

    overall_best_loss  = float("inf")
    overall_best_state = None
    run_best_losses    = []

    #fix seed for reproducibility
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)

    print("\nTraining autoencoder (config-based runs) …")
    for run in range(1, num_runs + 1):
        print(f"\n────────── Run {run}/{num_runs} ──────────")

        # ── 1.1) model, optimiser, loss ───────────────────────────────
        autoencoder = Autoencoder(input_dim=input_dim, latent_dim=32).to(device)
        optimizer   = optim.Adam(autoencoder.parameters(), lr=1e-4)
        criterion   = nn.MSELoss()

        best_loss   = float("inf")
        best_state  = None
        epochs_no_improve = 0

        # ── 1.2) training loop ────────────────────────────────────────
        for epoch in range(1, num_epochs + 1):
            # ─ training ─
            autoencoder.train()
            tr_loss = 0.0
            for (x,) in train_loader:
                x = x.to(device)
                optimizer.zero_grad()
                recon = autoencoder(x)
                loss  = criterion(recon, x)
                loss.backward()
                optimizer.step()
                tr_loss += loss.item()
            tr_loss /= len(train_loader)

            # ─ validation ─
            autoencoder.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (x,) in valid_loader:
                    x = x.to(device)
                    recon = autoencoder(x)
                    loss  = criterion(recon, x)
                    val_loss += loss.item()
            val_loss /= len(valid_loader)

            # status prints
            if epoch % print_every == 0 or epoch == 1:
                print(f"  epoch {epoch:>6}/{num_epochs} │ "
                    f"train={tr_loss:.6f}  val={val_loss:.6f}")

            # early-stopping bookkeeping
            if val_loss + min_delta < best_loss:
                best_loss  = val_loss
                best_state = copy.deepcopy(autoencoder.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"early stop at epoch {epoch} "
                    f"(no val-improve for {patience} epochs)")
                break

        run_best_losses.append(best_loss)
        print(f"  best val-loss for run {run}: {best_loss:.6f}")

        # keep overall winner
        if best_loss < overall_best_loss:
            overall_best_loss  = best_loss
            overall_best_state = copy.deepcopy(best_state)

    # ── 2) summary & model restoration ────────────────────────────────
    print("\nRun-wise best losses:", [f"{l:.6f}" for l in run_best_losses])
    print(f"Lowest validation loss across runs: {overall_best_loss:.6f}")

    # load the champion weights and keep your original handle
    autoencoder.load_state_dict(overall_best_state)
    model = autoencoder        # so the rest of your code stays unchanged

    return model

def train(config, train_loader, valid_loader, device, verbose):
    '''
        Train concept bottleneck model
    '''

    # Define model, loss function
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'], verbose=False)
    criterion = JointLoss(weight_lambda=config['weight_lambda'])

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
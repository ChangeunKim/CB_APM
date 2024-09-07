## A Consensus Bottleneck Asset Pricing Model


This repository contains code implementation of Master's Thesis titled "*A Consensus Bottleneck Asset Pricing Model*.

## Data

Dataset used in this work are sourced from WRDS and not allowed to be used without subscription. Please refer to the appendix of my paper for a detailed list of the variables utilized in the study.

## Usage

To run the model, execute run.py with the appropriate options.

* The argument ```-w``` sets the hyperparameter $\lambda$.
* Setting ```-w 0``` trains a naive feedforward neural network.

```console
python run.py -w $LAMBDA$
```

* The argument ```-p``` defines the forecasting horizon of the CB_APM.
* The default setting is ```-p 1```, where the model predicts stock returns one month ahead.

```console
python run.py -w $LAMBDA$ -p $HORIZON$
```

The results are saved in the following directories:
1. results: Contains out-of-sample $R^2$ results (*.csv) and inference results for test set (*.pickle) of CB_APM for different $\lambda$ settings.
2. output & error: Contains text files with console outputs from run.py (provided by the SLURM sbatch command).
3. Checkpoints: Trained model weights of CB_APM (*.pt) for different training periods (expanding windows).
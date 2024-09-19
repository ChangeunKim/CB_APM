## A Consensus Bottleneck Asset Pricing Model


This repository contains code implementation of Master's Thesis titled "*A Consensus Bottleneck Asset Pricing Model*. (The link to the revised version of the paper will be soon provided via SSRN.)

## Data

Dataset used in this work is mainly sourced from WRDS and cannot be accessed without a subscription. However, you can download the dataset we used to train CB_APM from the Google Drive link below. Please contact us by email to request permission to access the shared link.

Data link: https://drive.google.com/drive/folders/1ff_VxjDY0O3sZwSY7uQEMR9veXlDeb4C?usp=drive_link

For a detailed list of the variables used in the study, please refer to the appendix of our paper. We highly recommend fully understanding the dataset before conducting further research.

## Usage

### ```load_data.py```
This script loads raw data from Chen and Zimmermann (2022), Welch and Goyal (2008) and Gu et al. (2020), and generates datasets after data preprocessing. 

The following arguments allow you to set the target prediction horizon for consensus variables and stock returns:
* ```-c```: prediction horizon for consensus variables.
* ```-r```: prediction horizon for stock returns.

For example:
```console
python load_data.py -c 3 -r 12 
```
This command generates input and target datasets named ```input_predict_3_12month.csv``` and ```target_predict_3_12month.csv```, respectively.

A special case occurs when ```-c``` is set to 0, which corresponds to the consensus approximation case discussed in the empirical studies of the paper. For example:
```console
python load_data.py -c 0 -r 1 
```
This will generate datasets named ```input_approx_1month.csv``` and ```target_approx_1month.csv``` .

### ```run.py```
To run the model, execute run.py with the appropriate options.

By setting these arguments, the script will load the appropriate data file created by running ```load_data.py```.

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
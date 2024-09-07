# CB-APM: Consensus-Bottleneck Asset Pricing Model

<p align="center">
  <b>Interpretable Deep Learning for Stock Returns</b><br>
  <i>A Consensus-Bottleneck Asset Pricing Model</i>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2512.16251">
    <img src="https://img.shields.io/badge/arXiv-2512.16251-b31b1b" alt="arXiv">
  </a>
  <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5165817">
    <img src="https://img.shields.io/badge/SSRN-5165817-blue" alt="SSRN">
  </a>
  <a href="#license">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </a>
</p>

---

## 📖 Overview

This repository contains the official implementation of the paper **"Interpretable Deep Learning for Stock Returns: A Consensus-Bottleneck Asset Pricing Model"**.

📄 **Paper Links**: [arXiv](https://arxiv.org/abs/2512.16251) | [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5165817)

**CB-APM** (Consensus-Bottleneck Asset Pricing Model) is a novel deep learning framework that bridges the gap between predictive accuracy and interpretability in asset pricing. By incorporating analyst consensus forecasts as intermediate concepts, our model achieves superior out-of-sample performance while maintaining economic interpretability.

### Key Features

- 🎯 **Interpretability**: Utilizes analyst consensus variables as a conceptual bottleneck layer
- 📈 **Performance**: Outperforms traditional machine learning benchmarks in stock return prediction
- 🔬 **Economic Insights**: Provides meaningful decomposition of return predictions through consensus concepts
- ⚡ **Ensemble Learning**: Employs ensemble methods for robust predictions
- 📊 **Comprehensive Evaluation**: Includes extensive analysis notebooks for model evaluation

---

## 🏗️ Architecture

The CB-APM architecture consists of two main components:

```
┌─────────────────────────────────────────────────────────────────┐
│                         CB-APM Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Input Features (146 firm characteristics)                     │
│              │                                                  │
│              ▼                                                  │
│   ┌──────────────────────┐                                      │
│   │   Concept Network     │  Hidden: [64, 32]                   │
│   │   (Feedforward NN)    │  Activation: GELU                   │
│   └──────────────────────┘  LayerNorm + Dropout                 │
│              │                                                  │
│              ▼                                                  │
│   ┌──────────────────────┐                                      │
│   │ Consensus Bottleneck  │  9 Analyst Consensus Targets        │
│   │     (Concepts)        │                                     │
│   └──────────────────────┘                                      │
│              │                                                  │
│              ▼                                                  │
│   ┌──────────────────────┐                                      │
│   │  Prediction Network   │  Linear Layer                       │
│   │   (Final Output)      │                                     │
│   └──────────────────────┘                                      │
│              │                                                  │
│              ▼                                                  │
│        Stock Return Prediction                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Loss Function

The model is trained with a joint loss function:

$$\mathcal{L} = \mathcal{L}_{pred} + \lambda \cdot \mathcal{L}_{concept}$$

Where:
- $\mathcal{L}_{pred}$: MSE loss for stock return prediction
- $\mathcal{L}_{concept}$: MSE loss for consensus variable prediction
- $\lambda$: Hyperparameter controlling the weight of concept supervision

---

## 📁 Repository Structure

```
CB_APM/
├── run.py                 # Main training script for CB-APM
├── benchmark.py           # Benchmark model training script
├── load_data.py           # Data preprocessing script
├── config.py              # Model hyperparameters configuration
│
├── models/                # Model implementations
│   ├── networks.py        # Neural network architectures
│   ├── train.py           # Training procedures
│   ├── test.py            # Testing procedures
│   ├── losses.py          # Custom loss functions
│   ├── metrics.py         # Evaluation metrics
│   └── model_utils.py     # Utility functions
│
├── utils/                 # Utility modules
│   ├── data_utils.py      # Data loading and processing
│   └── data_preprocessor.py  # Data preprocessing utilities
│
├── analysis/              # Jupyter notebooks for analysis
│   ├── r_squared.ipynb    # R² analysis across models
│   ├── mse.ipynb          # MSE analysis (in-sample & out-of-sample)
│   ├── portfolio.ipynb    # Portfolio performance analysis
│   ├── coefficient.ipynb  # Consensus coefficient analysis
│   ├── consensus.ipynb    # Consensus variable analysis
│   ├── GRS.ipynb          # GRS test analysis
│   └── HJD.ipynb          # Hansen-Jagannathan Distance analysis
│
├── data/                  # Data directory (not included)
├── checkpoints/           # Saved model weights
├── results/               # Model predictions and metrics
└── tables/                # Generated result tables
```

---

## 📊 Data

The dataset used in this study is primarily sourced from **WRDS (Wharton Research Data Services)** and requires a subscription for access.

### Data Sources

| Source | Description |
|--------|-------------|
| Chen and Zimmermann (2022) | Firm characteristic variables |
| Welch and Goyal (2008) | Macroeconomic predictors |
| FRED-MD (McCracken and Ng, 2016) | Monthly macroeconomic indicators |
| Gu et al. (2020) | Extended factor sets |

### Data Access

You can download the preprocessed dataset from the Google Drive link below (permission required):

🔗 [**Download Dataset**](https://drive.google.com/drive/folders/1ff_VxjDY0O3sZwSY7uQEMR9veXlDeb4C?usp=drive_link)

> **Note**: Please contact us via email to request access. For a detailed list of variables, refer to the appendix of our paper.

---

## 🚀 Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.6+ (for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/ChangeunKim/CB_APM.git
cd CB_APM

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install torch numpy pandas scikit-learn optuna tqdm joblib matplotlib
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥1.12 | Deep learning framework |
| `numpy` | ≥1.21 | Numerical computations |
| `pandas` | ≥1.3 | Data manipulation |
| `scikit-learn` | ≥1.0 | Benchmark models & utilities |
| `optuna` | ≥3.0 | Hyperparameter optimization |
| `tqdm` | ≥4.62 | Progress bars |
| `joblib` | ≥1.1 | Model persistence |
| `matplotlib` | ≥3.5 | Visualization |

---

## 💻 Usage

### 1. Data Preprocessing

Generate input and target datasets for a specific prediction horizon:

```bash
python load_data.py -p <HORIZON>
```

**Arguments:**
- `-p, --horizon`: Prediction horizon in months (e.g., 1, 3, 6, 12)

**Example:**
```bash
# Generate datasets for 12-month ahead prediction
python load_data.py -p 12
```

This creates:
- `data/input_12month.csv`: Feature matrix
- `data/target_12month.csv`: Target returns

---

### 2. Training CB-APM

Train the Consensus-Bottleneck Asset Pricing Model:

```bash
python run.py -p <HORIZON> -w <LAMBDA>
```

**Arguments:**
- `-p, --horizon`: Prediction horizon in months
- `-w, --weight`: Hyperparameter λ for concept supervision
  - `λ = 0`: Standard feedforward neural network (no concept supervision)
  - `λ > 0`: CB-APM with concept bottleneck

**Examples:**
```bash
# Train CB-APM with λ=1.0 for 1-month prediction
python run.py -p 1 -w 1.0

# Train standard neural network (baseline)
python run.py -p 1 -w 0
```

**Output:**
- `results/<horizon>month_<lambda>.csv`: Out-of-sample R² results
- `results/<horizon>month_<lambda>_mse.csv`: MSE metrics
- `results/<horizon>month_<lambda>.pickle`: Detailed predictions
- `checkpoints/<horizon>month_<lambda>/`: Model weights

---

### 3. Training Benchmark Models

Run benchmark models for comparison:

```bash
python benchmark.py -p <HORIZON> [-t] [-n TRIALS]
```

**Arguments:**
- `-p, --horizon`: Prediction horizon in months
- `-t, --tune`: Enable hyperparameter tuning (optional)
- `-n, --trials`: Number of Optuna trials for tuning (default: 50)

**Example:**
```bash
# Run benchmarks with hyperparameter tuning
python benchmark.py -p 1 -t -n 100

# Run benchmarks with default hyperparameters
python benchmark.py -p 1
```

**Benchmark Models:**

| Model | Description |
|-------|-------------|
| **OLS** | Ordinary Least Squares regression |
| **PLS** | Partial Least Squares regression |
| **PCR** | Principal Component Regression |
| **ElasticNet** | L1+L2 regularized regression |
| **GLM** | Generalized Linear Model with splines |
| **RF** | Random Forest regressor |
| **GBRT** | Gradient Boosted Regression Trees |

Each model is trained on both raw input features and consensus data.

**Output:**
- `results/benchmarks/<horizon>month_benchmark.csv`: R² results
- `results/benchmarks/<horizon>month_<model>.pickle`: Predictions
- `checkpoints/benchmarks/`: Saved model objects

---

## 📈 Analysis Notebooks

The `analysis/` directory contains Jupyter notebooks for comprehensive model evaluation:

| Notebook | Description |
|----------|-------------|
| `r_squared.ipynb` | Compare out-of-sample R² across models and time periods |
| `mse.ipynb` | Analyze in-sample and out-of-sample Mean Squared Error |
| `portfolio.ipynb` | Evaluate portfolio performance based on model predictions |
| `coefficient.ipynb` | Analyze the coefficients of consensus variable approximations |
| `consensus.ipynb` | Deep dive into consensus variable predictions |
| `GRS.ipynb` | Gibbons-Ross-Shanken test for factor model evaluation |
| `HJD.ipynb` | Hansen-Jagannathan Distance analysis |

---

## 🔧 Model Configuration

Default hyperparameters in `config.py`:

```python
{
    # Architecture
    'input_size': 146,
    'concept_hidden_sizes': [64, 32],
    'concept_output_size': 9,
    'final_output_size': 1,
    
    # Training
    'lr': 0.0005,
    'epochs': 100,
    'batch_size': 5000,
    'ensemble': 10,
    
    # Regularization
    'weight_decay': 0.005,
    'clip_value': 1,
    
    # Early Stopping
    'early_stopping_patience': 5,
    'scheduling_patience': 2,
    'scheduling_factor': 0.2,
    
    # Concept Weight
    'weight_lambda': <user_specified>
}
```

---

## 📝 Evaluation Methodology

### Training Protocol

- **Expanding Window**: Training starts from 2010 with expanding validation and test periods
- **Ensemble**: 10 models trained with different random seeds
- **Early Stopping**: Based on validation loss with patience of 5 epochs

### Evaluation Metrics

- **Out-of-sample R²**: Main performance metric for return prediction
- **MSE**: Mean Squared Error for both concepts and returns
- **Portfolio Metrics**: Sharpe ratio, returns for long-short portfolios
- **GRS Test**: Statistical test for factor model validity
- **HJD**: Hansen-Jagannathan Distance for model comparison

---

## 📄 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{jang2025consensus,
  title={A Consensus-Bottleneck Asset Pricing Model},
  author={Jang, Bong-Gyu and Jeong, Younwoo and Kim, Changeun},
  journal={Available at SSRN 5165817},
  year={2025}
}
```

---

## 📬 Contact

For questions, data access requests, or collaboration inquiries, please contact:

- **Email**: changeun120@postech.ac.kr
- **arXiv**: [arXiv:2512.16251](https://arxiv.org/abs/2512.16251)
- **SSRN**: [SSRN:5165817](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5165817)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Data sources: WRDS, Chen and Zimmermann (2022), Welch and Goyal (2008), FRED-MD (McCracken and Ng, 2016), Gu et al. (2020)

---

<p align="center">
  <i>Last updated: January 2026</i>
</p>
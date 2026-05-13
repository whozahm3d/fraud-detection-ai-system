<div align="center">

# 🛡️ TrustGuard AI
### AI-Powered Fraud Detection for Digital Banking
#### with Explainable AI & RAG-Based Policy Assistance

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-006400?style=for-the-badge)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-orange?style=for-the-badge)](https://shap.readthedocs.io)
[![RAG](https://img.shields.io/badge/RAG-ChromaDB%20%2B%20GPT--4o--mini-7B68EE?style=for-the-badge)](https://www.trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Academic%20Project-blueviolet?style=for-the-badge)](.)

<br>

**🚀 Live Demo →** [trustguard-ai-fraud-detection.streamlit.app](https://trustguard-ai-fraud-detection-c7um3xntqvxthahgld5ucm.streamlit.app/)

<br>

> An end-to-end academic AI system that detects financial fraud on severely imbalanced data (0.13% fraud rate, 6.3M transactions), explains every prediction with SHAP, and grounds risk reports in State Bank of Pakistan regulations via a RAG pipeline — all accessible through a live Streamlit dashboard.

</div>

---

## ⚡ Key Results at a Glance

<div align="center">

| Metric | XGBoost (Deployed) |
|:---|:---:|
| **AUC-ROC** | **0.9995** |
| **Recall** | **0.9976** |
| **Avg Precision** | **0.9358** |
| **F1 Score** | **0.5691** |
| **Test Accuracy** | **99.80%** |
| RAG Hallucinations | **0 / 4** |
| Fraud cases caught (6.3M test set) | **8,190 / 8,213** |

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Features](#-features)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
  - [Class Imbalance Handling](#class-imbalance-handling)
  - [Feature Engineering](#feature-engineering)
  - [Models Trained](#models-trained)
  - [XAI — SHAP Explanations](#xai--shap-explanations)
  - [RAG Pipeline](#rag-pipeline)
- [Results](#-results)
- [Live Dashboard](#-live-dashboard)
- [Quick Start](#-quick-start)
- [Full Setup Guide](#-full-setup-guide)
- [Project Structure](#-project-structure)
- [Academic Context](#-academic-context)
- [Team](#-team)
- [References](#-references)

---

## 🎯 Project Overview

Financial fraud causes trillions of dollars in annual losses globally. Detecting it is hard — not because the algorithms are weak, but because **fraud is statistically invisible**: only 0.13% of transactions in the PaySim dataset are fraudulent. A naïve model predicting "legitimate" on everything achieves 99.87% accuracy while catching zero fraud.

**TrustGuard AI** solves this through three integrated components:

1. **Fraud Detection Engine** — A two-stage imbalance pipeline (Fraud Simulation + SMOTE) trains four ML models under identical conditions. XGBoost is deployed with AUC-ROC = 0.9995 and Recall = 0.9976.

2. **Explainable AI (XAI)** — SHAP TreeExplainer generates per-transaction waterfall explanations, showing exactly which features drove each prediction and by how much.

3. **RAG Policy Assistant** — A hybrid retrieval system (BM25 + dense vectors + CrossEncoder reranking) retrieves relevant SBP regulatory provisions and feeds them to GPT-4o-mini, generating grounded regulatory risk reports with zero hallucinations.

> **Academic note:** This is a thirs-year university AI course project built on the PaySim synthetic dataset. Results are strong but not validated on real-world production transaction data.

---

## 🏗️ System Architecture

```
Raw PaySim CSV (6.36M rows)
        │
        ▼
┌─────────────────────┐
│   Data Cleaning &   │  Drop nameOrig, nameDest, isFlaggedFraud
│  Feature Engineering│  Add balanceDiff, amount_ratio, one-hot type
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Fraud Simulation   │  Inject synthetic full-drain transactions
│     Engine          │  0.13% → 1.26% fraud rate
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐     ┌──────────────────────────┐
│  80/20 Stratified   │     │   SBP Regulatory Corpus  │
│    Train/Test Split │     │  (5 PDFs → ~100 chunks)  │
└─────────┬───────────┘     └────────────┬─────────────┘
          │                              │
          ▼                              ▼
┌─────────────────────┐     ┌──────────────────────────┐
│  ImbPipeline × 5    │     │  ChromaDB Vector Store   │
│  (SMOTE 0.3 →       │     │  all-MiniLM-L6-v2        │
│  StandardScaler →   │     │  BM25 + Dense Retrieval  │
│  Classifier)        │     │  CrossEncoder Reranker   │
└─────────┬───────────┘     └────────────┬─────────────┘
          │                              │
          ▼                              ▼
┌─────────────────────┐     ┌──────────────────────────┐
│  XGBoost (deployed) │────▶│   GPT-4o-mini Generator  │
│  + SHAP TreeExplain │     │   Regulatory Risk Report │
└─────────┬───────────┘     └────────────┬─────────────┘
          │                              │
          └──────────────┬───────────────┘
                         ▼
              ┌─────────────────────┐
              │  Streamlit Dashboard│
              │  6 interactive pages│
              └─────────────────────┘
```

---

## ✨ Features

### 🔍 Fraud Detection
- Trains **4 models** (Logistic Regression, Random Forest, Neural Network, XGBoost) under identical pipeline conditions
- Two-stage imbalance handling raises training fraud rate from **0.13% → 23.07%** without data leakage
- 3-fold cross-validation with SMOTE applied strictly inside each fold
- Full ablation study across 7 conditions isolating the contribution of each pipeline component

### 🧠 Explainable AI
- **SHAP TreeExplainer** for per-transaction waterfall plots
- Feature importance comparison across all 4 models
- Bias & fairness check across transaction type subgroups
- Threshold optimisation curve showing precision/recall trade-offs

### 📜 RAG Policy Assistant
- **Hybrid retrieval**: BM25 (lexical) + sentence-transformers (semantic) combined
- **CrossEncoder reranking** (ms-marco-MiniLM-L-6-v2) for passage quality scoring
- **GPT-4o-mini** generation grounded in SBP regulatory documents
- Zero hallucinations across all evaluated transactions
- Average retrieval Precision = 0.855 across 10 regulatory queries

### 📊 Interactive Dashboard
- Single transaction prediction with real-time SHAP explanation
- Batch CSV upload and scoring
- Full EDA visualisation suite
- Model comparison, confusion matrices, ROC/PR curves
- Ablation study explorer
- Regulatory report generation (requires OpenAI API key)

---

## 📊 Dataset

**PaySim Synthetic Financial Dataset** — [Kaggle Link](https://www.kaggle.com/datasets/ealaxi/paysim1)

A multi-agent simulation calibrated against real mobile money transaction data from an African financial institution.

| Property | Value |
|:---|:---|
| Total transactions | 6,362,620 |
| Fraudulent transactions | 8,213 (**0.13%**) |
| Legitimate transactions | 6,354,407 |
| Simulation period | 30 days (743 hourly steps) |
| Transaction types | CASH_OUT, TRANSFER, PAYMENT, DEBIT, CASH_IN |
| Fraud types | CASH_OUT and TRANSFER **only** |
| Missing values | None |

> ⚠️ The raw dataset (~500MB) is not included. Download from Kaggle and place at `Data/original_dataset/` or use the dataset link provided above.

---

## 🔬 Methodology

### Class Imbalance Handling

The 0.13% fraud rate is too severe for SMOTE alone — minority folds in 5-fold CV can contain fewer than 10 fraud samples. TrustGuard uses a **two-stage strategy**:

```
Stage 1 — Fraud Simulation Engine (global, before split)
  • Sample 5% of legitimate TRANSFER & CASH_OUT transactions
  • Set amount = oldbalanceOrg (full drain), newbalanceOrig = 0
  • Recompute balanceDiff and amount_ratio
  • Label injected rows as fraud
  Result: 0.13% → 1.26% fraud rate

Stage 2 — SMOTE inside ImbPipeline (per CV fold, train only)
  • sampling_strategy = 0.3
  • Applied AFTER train/test split, INSIDE each fold
  • Validation fold never sees synthetic samples
  Result: 1.26% → 23.07% fraud rate in training
```

| Stage | Fraud Cases | Total Rows | Fraud Rate |
|:---|:---:|:---:|:---:|
| Original | 8,213 | 6,362,620 | 0.13% |
| After Fraud Simulation | ~87,500 | ~6,940,000 | 1.26% |
| After SMOTE (train folds) | ~1,600,000 | ~6,930,000 | **23.07%** |

### Feature Engineering

12 features used for model training (6 raw + 2 engineered + 4 one-hot):

| Feature | Type | Signal |
|:---|:---|:---|
| `step` | Raw | Temporal position (0–743 hours) |
| `amount` | Raw | Transaction size — higher in fraud |
| `oldbalanceOrg` | Raw | Origin balance before |
| `newbalanceOrig` | Raw | Drops to 0 in full-drain fraud |
| `oldbalanceDest` | Raw | Mule accounts often start at 0 |
| `newbalanceDest` | Raw | Spikes in fraud cases |
| `balanceDiff` | **Engineered** | `oldbalanceOrg − newbalanceOrig − amount` — detects inconsistencies |
| `amount_ratio` | **Engineered** | `amount / (oldbalanceOrg + 1)` — approaches 1.0 in drain attacks |
| `type_CASH_OUT` | One-hot | Binary type indicator |
| `type_DEBIT` | One-hot | Binary type indicator |
| `type_PAYMENT` | One-hot | Binary type indicator |
| `type_TRANSFER` | One-hot | Binary type indicator |

### Models Trained

All four models use `ImbPipeline(SMOTE → StandardScaler → Classifier)` with 5-fold stratified CV:

| Model | Role | Key Hyperparameters |
|:---|:---|:---|
| **XGBoost** ⭐ | Deployed model | `n_estimators=300`, `learning_rate=0.05`, `max_depth=6` |
| Random Forest | Ensemble baseline | `max_depth=12`, `n_estimators=30`, `max_samples=0.5` |
| Neural Network | Non-linear baseline | `hidden_layer_sizes=(32,16)`, `batch_size=256`, `alpha=0.03` |
| Logistic Regression | Linear baseline | `C=1.0`, `solver=saga` |

### XAI — SHAP Explanations

SHAP `TreeExplainer` decomposes each prediction into per-feature additive contributions. Top drivers:

1. **`amount_ratio`** — values near 1.0 (full drain) are the strongest fraud signal
2. **`newbalanceOrig`** — near-zero post-transaction balance confirms drain
3. **`balanceDiff`** — inconsistencies between amount and balance change
4. **`type_CASH_OUT` / `type_TRANSFER`** — transaction type as a gate
5. **`amount`** — absolute size, though weaker than the ratio

<div align="center">
<img src="https://raw.githubusercontent.com/whozahm3d/trustguard-ai-fraud-detection/main/Images/final_deliverable_images/feature_importance_comparison.png" width="85%" alt="Feature Importance Comparison across all 4 models"/>
<br><em>Feature importance comparison across all 4 trained models</em>
</div>

### RAG Pipeline

| Component | Technology |
|:---|:---|
| Vector store | ChromaDB (`~100 chunks`) |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Lexical retrieval | BM25 (rank-bm25) |
| Reranker | `ms-marco-MiniLM-L-6-v2` CrossEncoder |
| Generator | `gpt-4o-mini` (OpenAI) |
| Regulatory corpus | 5 SBP documents (C1-Annex, C2-Annex-A, CL33-Annex-B, C10-Branchless-Banking, SME-PRs-Jan-2025) |

---

## 📈 Results

### Model Comparison — Test Set

| Model | Precision | Recall | F1 | AUC-ROC | Avg Precision |
|:---|:---:|:---:|:---:|:---:|:---:|
| **XGBoost** ⭐ | **0.3981** | **0.9976** | **0.5691** | **0.9995** | **0.9358** |
| Random Forest | 0.1035 | 0.9976 | 0.1875 | 0.9995 | 0.8870 |
| Neural Network | 0.1437 | 0.9732 | 0.2505 | 0.9983 | 0.7081 |
| Logistic Regression | 0.0217 | 0.9860 | 0.0425 | 0.9946 | 0.5567 |

<div align="center">
<img src="https://raw.githubusercontent.com/whozahm3d/trustguard-ai-fraud-detection/main/Images/final_deliverable_images/model_comparison_all_metrics.png" width="85%" alt="Model Comparison — All Metrics"/>
<br><em>All-metric comparison across the four trained models</em>
</div>

### ROC & Precision-Recall Curves

<div align="center">
<img src="https://raw.githubusercontent.com/whozahm3d/trustguard-ai-fraud-detection/main/Images/final_deliverable_images/roc_pr_curves.png" width="85%" alt="ROC and Precision-Recall Curves"/>
<br><em>ROC and Precision-Recall curves — XGBoost achieves AUC-ROC = 0.9995</em>
</div>

### Confusion Matrices — All Models

<div align="center">
<img src="https://raw.githubusercontent.com/whozahm3d/trustguard-ai-fraud-detection/main/Images/final_deliverable_images/confusion_matrices_all_models.png" width="85%" alt="Confusion Matrices for all 4 models"/>
<br><em>Confusion matrices for all four models on the held-out test set</em>
</div>

### Cross-Validation — XGBoost (5-fold, mean ± std)

| CV Precision | CV Recall | CV F1 | CV AUC-ROC | CV Avg Precision |
|:---:|:---:|:---:|:---:|:---:|
| 0.905 ± 0.035 | 1.000 ± 0.000 | 0.949 ± 0.020 | 1.000 ± 0.000 | 0.985 ± 0.004 |

### Ablation Study — All 7 Conditions

| ID | Condition | CV F1 | Test Recall | Test F1 | Test AP |
|:---|:---|:---:|:---:|:---:|:---:|
| A1 | No Fraud Simulation | 0.671 | 0.9976 | 0.6247 | 0.9363 |
| **A2** | **Full Pipeline** ⭐ | **0.947** | **0.9976** | **0.5533** | **0.7317** |
| B1 | No SMOTE | 0.947 | 0.9951 | 0.9132 | 0.9639 |
| **B2** | **SMOTE ratio=0.3** ⭐ | **0.947** | **0.9976** | **0.5557** | **0.7322** |
| B3 | SMOTE ratio=0.5 | 0.947 | 0.9976 | 0.5280 | 0.7180 |
| C1 | No Engineered Features | 0.636 | 0.9976 | 0.1538 | 0.6061 |
| **C2** | **With Features** ⭐ | **0.947** | **0.9976** | **0.5533** | **0.7317** |

> Removing engineered features (C1) drops Test F1 by **73%**. Removing Fraud Simulation (A1) drops CV F1 by **29%**. Both are critical.

### Cost-Benefit Analysis

<div align="center">
<img src="https://raw.githubusercontent.com/whozahm3d/trustguard-ai-fraud-detection/main/Images/final_deliverable_images/cost_benefit_analysis.png" width="85%" alt="Cost-Benefit Analysis"/>
<br><em>Cost-benefit analysis across decision thresholds</em>
</div>

### RAG Evaluation

| Transaction | Fraud Prob | Risk Tier | Latency | Hallucination |
|:---|:---:|:---:|:---:|:---:|
| TXN-CRITICAL-001 | 97% | 🔴 CRITICAL | 6.45s | ✅ None |
| TXN-HIGH-001 | 94% | 🔴 CRITICAL | 4.67s | ✅ None |
| TXN-HIGH-002 | 88% | 🔴 CRITICAL | 3.40s | ✅ None |
| TXN-MEDIUM-001 | 73% | 🟠 HIGH | 3.64s | ✅ None |

Retrieval: **Avg Precision@5 = 0.855**, Term Hit Rate = 0.90, Expected doc found: **10/10**

---

## 🖥️ Live Dashboard

**URL:** [https://trustguard-ai-fraud-detection-c7um3xntqvxthahgld5ucm.streamlit.app/](https://trustguard-ai-fraud-detection-c7um3xntqvxthahgld5ucm.streamlit.app/)

The dashboard has **6 pages**, navigated from the sidebar:

| Page | What it does |
|:---|:---|
| **① Predict Transaction** | Enter transaction attributes → get fraud probability, risk tier, SHAP waterfall plot, and (with API key) a RAG regulatory report |
| **② Batch CSV Analysis** | Upload a CSV of transactions → score all rows, download results |
| **③ Dataset & Imbalance** | Interactive EDA: class distribution, fraud patterns, feature distributions, imbalance pipeline visualisation |
| **④ Model Performance** | ROC/PR curves, confusion matrices, threshold optimisation, cost-benefit analysis, bias check |
| **⑤ Ablation Study** | All 7 conditions across all metrics — bar chart, heatmap, delta plot, PR scatter |
| **⑥ About** | Team info, architecture overview, links |

> ⚠️ The RAG regulatory report on Page 1 requires an OpenAI API key. All other pages work without one.

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/whozahm3d/trustguard-ai-fraud-detection.git
cd trustguard-ai-fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app.py
```

The dashboard loads the **pre-trained XGBoost model** from `outputs/deployment/model.pkl` — no data or retraining required.

---

## 🔧 Full Setup Guide

### Prerequisites

- Python 3.10+
- pip
- (Optional) OpenAI API key for the RAG regulatory report feature

### 1. Clone the Repository

```bash
git clone https://github.com/whozahm3d/trustguard-ai-fraud-detection.git
cd trustguard-ai-fraud-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>Core dependencies (click to expand)</summary>

| Package | Version | Purpose |
|:---|:---|:---|
| `streamlit` | 1.45.0 | Dashboard UI |
| `xgboost` | 2.1.4 | Primary deployed model |
| `scikit-learn` | 1.5.2 | ML pipeline & baselines |
| `imbalanced-learn` | 0.12.4 | SMOTE / ImbPipeline |
| `shap` | ≥0.45, <0.48 | XAI explanations |
| `chromadb` | 0.5.23 | RAG vector store |
| `sentence-transformers` | 3.0.1 | RAG dense embeddings |
| `rank-bm25` | 0.2.2 | RAG lexical retrieval |
| `openai` | ≥1.30, <2.0 | RAG generation (GPT-4o-mini) |
| `pandas` | 2.2.3 | Data processing |
| `numpy` | 1.26.4 | Numerics |
| `scipy` | 1.13.1 | Statistical utilities |
| `matplotlib` | 3.9.4 | Plotting |
| `seaborn` | 0.13.2 | Statistical visualisation |
| `joblib` | 1.4.2 | Model serialisation |
| `psutil` | 6.1.1 | System resource monitoring |
| `tqdm` | 4.66.4 | Progress bars |

</details>

### 3. Configure OpenAI API Key (Optional — RAG only)

Copy the secrets template and add your key:

```bash
cp .streamlit/secrets.example.toml .streamlit/secrets.toml
# Edit secrets.toml and replace sk-proj-YOUR_KEY_HERE with your actual key
```

Or set it as an environment variable:

```bash
export OPENAI_API_KEY="sk-proj-your-key-here"
```

> Without this key, all dashboard pages except the RAG report on Page 1 work normally.

### 4. Download the Dataset (for retraining only)

The pre-trained model is already included in `outputs/deployment/`. If you want to retrain from scratch:

1. Download from [Kaggle — PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1)
2. Place the CSV at `Data/original_dataset/`
3. Run the full training notebook:

```bash
jupyter notebook notebooks/final_deliverable_notebooks/trustguard_ai_final.ipynb
```

### 5. Run the Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### 6. Run Individual Notebooks

```bash
# EDA and preprocessing (Deliverable 1)
jupyter notebook notebooks/deliverable_1_notebooks/exploratory_data_analysis.ipynb

# Full preprocessing + model training pipeline (Deliverable 2)
jupyter notebook notebooks/deliverable_2_notebooks/deliverable_1_and_2_pipeline.ipynb

# Final model, ablation, XAI (Deliverable 3)
jupyter notebook notebooks/final_deliverable_notebooks/trustguard_ai_final.ipynb

# RAG pipeline build + evaluation
jupyter notebook notebooks/final_deliverable_notebooks/rag_system.ipynb

# Model export + dashboard preparation
jupyter notebook notebooks/final_deliverable_notebooks/deployment.ipynb
```

---

## 📁 Project Structure

```
trustguard-ai-fraud-detection/
│
├── app.py                              # Streamlit dashboard (6 pages, ~1600 lines)
├── rag_module.py                       # RAG pipeline: retrieval + reranking + generation
├── requirements.txt                    # All dependencies with pinned versions
├── .python-version                     # Python 3.10+ specifier
├── LICENSE                             # MIT License
├── CONTRIBUTING.md                     # Contribution guidelines
│
├── .streamlit/
│   ├── config.toml                     # Streamlit theme config
│   └── secrets.example.toml           # API key template (never commit secrets.toml)
│
├── .devcontainer/
│   └── devcontainer.json              # GitHub Codespaces / VS Code dev container config
│
├── Data/
│   ├── original_dataset/
│   │   └── Paysim_dataset             # Raw PaySim CSV placeholder (not included — download from Kaggle)
│   └── processed_data/
│       ├── processed_paysim           # Cleaned + feature-engineered dataset
│       ├── train_original             # Pre-SMOTE training split
│       ├── train_after_simulation     # Post fraud simulation
│       ├── train_final_smote          # Post SMOTE (final training data)
│       └── test_original              # Held-out test set
│
├── notebooks/
│   ├── deliverable_1_notebooks/
│   │   ├── deliverable_1_pipeline.ipynb        # Data loading, cleaning, EDA
│   │   └── exploratory_data_analysis.ipynb     # Full EDA with 8+ visualisations
│   ├── deliverable_2_notebooks/
│   │   └── deliverable_1_and_2_pipeline.ipynb  # Feature engineering + model training
│   └── final_deliverable_notebooks/
│       ├── trustguard_ai_final.ipynb            # Full pipeline, ablation, XAI
│       ├── rag_system.ipynb                     # RAG build + evaluation
│       └── deployment.ipynb                     # Model export + dashboard prep
│
├── scripts/
│   ├── deliverable_1_scripts/          # Modular preprocessing scripts
│   │   ├── load_dataset.py
│   │   ├── basic_inspection.py
│   │   ├── data_cleaning.py
│   │   ├── feature_engineering.py
│   │   ├── fraud_simulation_engine.py
│   │   ├── preprocessing_pipeline.py
│   │   ├── scaling_dataset.py
│   │   ├── smote.py
│   │   ├── train_test_split.py
│   │   ├── save_cleaned_dataset.py
│   │   ├── libraries_loaded.py
│   │   └── eda_analysis_and_visualization.py
│   └── deliverable_2_scripts/          # Model training & evaluation scripts
│       ├── logistic_regression.py
│       ├── random_forest.py
│       ├── nueral_networks.py
│       ├── xgboost_regression.py
│       ├── compare_models.py
│       ├── hyperparameter_config.py
│       ├── kfold_validation.py
│       ├── feature_importance.py
│       ├── roc_curves.py
│       ├── visualize_confusion_matrix.py
│       ├── error_analysis.py
│       ├── bias_check.py
│       └── ablation_study.py
│
├── outputs/
│   ├── deployment/                     # Artefacts loaded by app.py at runtime
│   │   ├── model.pkl                  # Trained XGBoost model
│   │   ├── scaler.pkl                 # Fitted StandardScaler
│   │   ├── model_meta.json            # Model metadata + test metrics
│   │   ├── feature_names.json         # Feature list in training order
│   │   └── model_comparison_deploy.png
│   ├── models/                         # All 4 trained model pkl files
│   │   ├── logistic_regression.pkl
│   │   ├── neural_network.pkl
│   │   ├── random_forest.pkl
│   │   ├── xgboost.pkl
│   │   └── scaler.pkl
│   ├── metrics/                        # Per-model metrics JSON + CSV
│   │   ├── model_comparison.json      # All 4 models, CV + test metrics
│   │   ├── model_comparison.csv
│   │   ├── hyperparameter_table.csv
│   │   ├── logistic_regression_metrics.{json,csv}
│   │   ├── neural_network_metrics.{json,csv}
│   │   ├── random_forest_metrics.{json,csv}
│   │   └── xgboost_metrics.{json,csv}
│   ├── ablation/                       # Ablation study results + plots
│   │   ├── ablation_results.csv
│   │   ├── ablation_study.png
│   │   ├── ablation_heatmap.png
│   │   ├── ablation_delta.png
│   │   ├── ablation_pr_scatter.png
│   │   └── ablation_smote_trend.png
│   ├── plots/                          # Full set of EDA, model, and evaluation plots
│   ├── experiments/                    # Raw experiment logs
│   │   ├── experiment_results.csv
│   │   └── cleaned_experiment_results.csv
│   └── rag/                            # RAG evaluation outputs
│       ├── retrieval_evaluation.csv
│       ├── response_evaluation.csv
│       ├── TXN-CRITICAL-001.txt
│       ├── TXN-HIGH-001.txt
│       ├── TXN-HIGH-002.txt
│       └── TXN-MEDIUM-001.txt
│
├── Images/
│   ├── Deliverable_1_images/           # EDA & preprocessing plots
│   ├── deliverable_2_images/           # Model training & evaluation plots
│   └── final_deliverable_images/       # Full final report plots (used in README)
│
├── SBP_Documents/                      # State Bank of Pakistan regulatory PDFs
│   ├── C1-Annex.pdf                   # AML/CFT regulations
│   ├── C2-Annex-A.pdf                 # Customer Due Diligence
│   ├── CL33-Annex-B.pdf               # Reporting obligations
│   ├── C10-Branchless-Banking-Regulations.pdf
│   └── SME-PRs-Updtd-Jan-2025.pdf
│
├── chroma_db/                          # Pre-built ChromaDB vector store (SBP chunks)
│   └── chroma.sqlite3
│
└── reports/
    ├── Fraud_Detection_Deliverable 1_Report.pdf
    ├── Fraud_Detection_Deliverable_2_Report.pdf
    ├── Fraud_Detection_Final_Report.pdf
    └── Fraud_Detection_Final_Report.tex   # LaTeX source for final report
```

---

## 🎓 Academic Context

This project was developed as the final deliverable for the **Artificial Intelligence** course at:

> **National University of Computer & Emerging Sciences (FAST-NUCES)**
> Department of Data Science & Artificial Intelligence — Lahore Campus
> Spring 2026 | Instructor: **Hajra Waheed**

It spans three deliverables:

| Deliverable | Scope |
|:---|:---|
| **Deliverable 1** | Dataset acquisition, preprocessing pipeline, EDA (8 visualisations) |
| **Deliverable 2** | Feature engineering, model training, class imbalance handling, evaluation |
| **Deliverable 3** | Ablation study, XAI (SHAP), RAG pipeline, Streamlit deployment |

**Important:** This is an academic prototype. The PaySim dataset is synthetic. Results have not been validated on real production transaction data and should not be used as-is in any financial system.

---

## 👨‍💻 Team

| Name | Student ID |
|:---|:---|
| Taha Nawaz | 23L-2644 |
| Ali Ahmad | 23L-2619 |
| Shahzeb Imran | 23L-2506 |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). Academic use, learning, and extension are encouraged. Attribution appreciated.

---

## 📚 References

- Lopez-Rojas, E.A. et al. (2016). *PaySim: A financial mobile money simulator for fraud detection.* EMSS.
- Chawla, N.V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR, 16, 321–357.
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD 2016.
- Lundberg, S.M. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 30.
- State Bank of Pakistan. (2023–2025). *AML/CFT Regulations, Branchless Banking Regulations, SME Payment Regulations.* [sbp.org.pk](https://www.sbp.org.pk)

---

<div align="center">

**🛡️ TrustGuard AI** — Built with purpose, explained with transparency, grounded in regulation.

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Click%20Here-FF4B4B?style=for-the-badge)](https://trustguard-ai-fraud-detection-c7um3xntqvxthahgld5ucm.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-whozahm3d-181717?style=for-the-badge&logo=github)](https://github.com/whozahm3d/trustguard-ai-fraud-detection)

</div>

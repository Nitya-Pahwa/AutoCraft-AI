#  AutoCraft AI — Agentic AutoML Pipeline

> An end-to-end **Agentic AutoML System** that automatically analyzes tabular datasets, selects optimal machine learning models, performs hyperparameter tuning, evaluates performance, and iteratively improves results through a critic-driven feedback loop.

Built using **LangGraph**, **Optuna**, **MLflow**, **Scikit-Learn**, **XGBoost**, **LightGBM**, and **LLM-powered reasoning via Groq**.

---

##  Overview

AutoCraft AI transforms raw tabular datasets into production-ready machine learning models with minimal human intervention.

The system combines traditional AutoML techniques with an **agent-based workflow**, where specialized agents collaborate to:

* Understand the dataset
* Detect the ML problem type
* Engineer preprocessing strategies
* Select suitable algorithms
* Optimize hyperparameters
* Evaluate model quality
* Critique results and retrain when necessary
* Generate experiment reports

Both a **Streamlit Web Interface** and **Command-Line Interface (CLI)** are supported.

---

##  Key Features

### Agentic Workflow

Multiple AI agents collaborate through a LangGraph workflow:

* Memory Agent
* Planner Agent
* Analyst Agent
* Model Selector Agent
* Trainer Agent
* Evaluator Agent
* Critic Agent
* Reporter Agent

---

### Automated Data Understanding

AutoCraft AI automatically analyzes:

* Missing values
* Duplicate records
* Outliers
* Column data types
* Class imbalance
* Target distribution
* Numeric-looking string columns
* High-cardinality categorical features
* Free-text features

---

### Intelligent Model Selection

Automatically detects:

* Classification Problems
* Regression Problems

Then selects and optimizes an appropriate model from a diverse model pool.

---

### Hyperparameter Optimization

Uses **Optuna** to:

* Search optimal hyperparameters
* Improve validation performance
* Reduce manual experimentation
* Support critic-guided retraining

---

### Experiment Tracking

Integrated with **MLflow** for:

* Run tracking
* Metric logging
* Parameter logging
* Experiment comparison

---

### Automated Reporting

Generates detailed experiment reports containing:

* Dataset analysis
* Feature preprocessing summary
* Model selection rationale
* Hyperparameter configuration
* Evaluation metrics
* Critic feedback
* Final recommendations

---

## System Architecture

```text
Memory Retrieve
        │
        ▼
    Planner
        │
        ▼
    Analyst
        │
        ▼
    Selector
        │
        ▼
    Trainer
        │
        ▼
   Evaluator
        │
        ▼
     Critic
   ┌────┴────┐
   │         │
 Retry    Approve
   │         │
   ▼         ▼
Trainer   Memory Store
              │
              ▼
          Reporter
```

---

## Tech Stack

| Category                    | Technologies       |
| --------------------------- | ------------------ |
| Workflow Orchestration      | LangGraph          |
| Frontend                    | Streamlit          |
| Data Processing             | Pandas, NumPy      |
| Machine Learning            | Scikit-Learn       |
| Hyperparameter Optimization | Optuna             |
| Experiment Tracking         | MLflow             |
| Visualization               | Plotly, Matplotlib |
| Serialization               | Joblib             |
| Validation                  | Pydantic           |
| Gradient Boosting           | XGBoost, LightGBM  |
| LLM Integration             | Groq               |

---

## Project Structure

```text
.
├── run_pipeline.py
├── requirements.txt
├── README.md
└── automl_agent
    ├── agents
    │   ├── analyst.py
    │   ├── critic.py
    │   ├── evaluator.py
    │   ├── memory.py
    │   ├── planner.py
    │   ├── reporter.py
    │   ├── selector.py
    │   └── trainer.py
    │
    ├── core
    │   ├── graph.py
    │   ├── llm.py
    │   ├── models.py
    │   └── state.py
    │
    ├── ui
    │   └── app.py
    │
    └── utils
        ├── data_loader.py
        ├── memory.py
        ├── mlflow_tracker.py
        ├── preprocessing.py
        └── visualizer.py
```

---

## Workflow

### 1. Memory Retrieval

Searches previous experiments and retrieves relevant historical runs.

### 2. Planning

Determines:

* Optuna trial count
* Retry budget
* Training strategy

### 3. Dataset Analysis

Profiles the dataset and identifies:

* Feature types
* Missing values
* Outliers
* Data quality issues
* Target characteristics

### 4. Model Selection

Chooses an appropriate baseline model and initial hyperparameters.

### 5. Training & Optimization

Runs:

* Data preprocessing
* Hyperparameter tuning
* Final model training

### 6. Evaluation

Computes task-specific metrics and overfitting diagnostics.

### 7. Critique & Retry

The Critic Agent:

* Reviews model quality
* Approves successful models
* Requests retraining when necessary

### 8. Report Generation

Creates a detailed Markdown experiment report.

---

## Supported Models

### Classification

* Logistic Regression
* Linear SVC (Calibrated)
* Decision Tree
* Random Forest
* Extra Trees
* Gradient Boosting
* Histogram Gradient Boosting
* K-Nearest Neighbors
* Gaussian Naive Bayes
* XGBoost
* LightGBM

### Regression

* Ridge
* Lasso
* ElasticNet
* Decision Tree Regressor
* Random Forest Regressor
* Extra Trees Regressor
* Gradient Boosting Regressor
* Histogram Gradient Boosting Regressor
* KNN Regressor
* XGBoost Regressor
* LightGBM Regressor

---

## Automatic Preprocessing

| Feature Type                | Strategy                             |
| --------------------------- | ------------------------------------ |
| Numeric                     | Median Imputation + Standard Scaling |
| Numeric-Like Strings        | Value Extraction + Scaling           |
| Low Cardinality Categories  | One-Hot Encoding                     |
| High Cardinality Categories | Frequency Encoding                   |
| Free Text                   | TF-IDF Vectorization                 |
| ID Columns                  | Automatically Removed                |
| Duplicate Rows              | Automatically Removed                |

Examples of supported numeric-like strings:

```text
8GB
1.5 kg
2 BHK
1000-1500
```

---

## Evaluation Metrics

### Classification

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix
* Overfitting Analysis

### Regression

* RMSE
* MAE
* R² Score
* Overfitting Analysis

---

## Installation

### Clone Repository

```bash
[git clone https://github.com/<username>/AutoCraft-AI.git](https://github.com/Nitya-Pahwa/AutoCraft-AI.git)
cd AutoCraft-AI
```

### Create Virtual Environment

```bash
python -m venv .venv
```

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## LLM Setup

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

Without an API key, AutoCraft AI automatically falls back to rule based reasoning.

---

## Running the Application

### Streamlit UI

```bash
streamlit run automl_agent/ui/app.py
```

### CLI

```bash
python run_pipeline.py \
--data data/heart.csv \
--target target \
--name heart_disease
```

With custom retry count:

```bash
python run_pipeline.py \
--data data/heart.csv \
--target target \
--name heart_disease \
--retries 2
```

---

## Generated Artifacts

```text
artifacts/
├── dataset_best.joblib
├── dataset_preprocessor.joblib
└── experiment_memory.jsonl
```

---

## Current Limitations

* Supports only supervised tabular learning
* No forecasting support
* No clustering support
* No image/audio pipelines
* Large datasets may require reducing Optuna search space

---

## Future Roadmap

* REST API for model inference
* Cross-validation support
* SHAP explainability
* Feature importance dashboard
* Automated deployment pipelines
* Distributed hyperparameter optimization
* Unit & integration tests
* Docker support

---

## Why AutoCraft AI?

Unlike traditional AutoML tools, AutoCraft AI introduces an **agentic decision-making layer**, enabling:

* Context-aware model selection
* Critic-driven iterative improvement
* Experiment memory and learning
* Human-readable reasoning and reports
* Hybrid symbolic + LLM-assisted workflows

This makes AutoCraft AI not just an AutoML system, but an intelligent machine learning assistant capable of reasoning about the entire model development lifecycle.

## Application Demo

<img width="1920" height="907" alt="Screenshot (2467)" src="https://github.com/user-attachments/assets/bcda05d3-4ba9-4d36-97dd-adc0ac2c5fcb" />

<img width="1920" height="901" alt="Screenshot (2470)" src="https://github.com/user-attachments/assets/9437916e-f413-4225-b27e-1d524f4c70d3" />

<img width="1920" height="909" alt="Screenshot (2276)" src="https://github.com/user-attachments/assets/d9332927-a397-48a2-9b9b-28517010d4e9" />

<img width="1920" height="903" alt="Screenshot (2277)" src="https://github.com/user-attachments/assets/55ec2ef6-44e7-4557-9ee9-5e27f9d34c3a" />

<img width="1920" height="901" alt="Screenshot (2471)" src="https://github.com/user-attachments/assets/a33e50e2-9a81-4511-8f8d-9d9287958017" />


<img width="1920" height="902" alt="Screenshot (2279)" src="https://github.com/user-attachments/assets/ca4112c8-dc82-485b-bee7-7578f486a2eb" />

<img width="1920" height="910" alt="Screenshot (2280)" src="https://github.com/user-attachments/assets/a11742c2-f2fa-4a2f-a1b7-6eb9ff9a5662" />

<img width="1920" height="878" alt="Screenshot (2281)" src="https://github.com/user-attachments/assets/cc77f825-840d-43fa-b7e4-08a423dd0420" />

<img width="1920" height="902" alt="Screenshot (2282)" src="https://github.com/user-attachments/assets/167e1114-d46f-4adf-b058-04770625f985" />

<img width="1920" height="783" alt="Screenshot (2283)" src="https://github.com/user-attachments/assets/77c21d4e-110d-4a66-a2d4-116c487257f4" />

<img width="1920" height="890" alt="Screenshot (2284)" src="https://github.com/user-attachments/assets/7a1613b9-a2d7-4bf0-8d28-af1a6cc16c4f" />

<img width="1920" height="897" alt="Screenshot (2285)" src="https://github.com/user-attachments/assets/b1c6d0b3-7280-4e68-8408-1f54dc5d8f7d" />

<img width="1920" height="895" alt="Screenshot (2286)" src="https://github.com/user-attachments/assets/807d4f0c-1183-4c99-b833-31feda6ef8c3" />

<img width="1920" height="893" alt="Screenshot (2287)" src="https://github.com/user-attachments/assets/8433def7-f1bd-473b-9d0a-a8bfcdcf9339" />

<img width="1920" height="885" alt="Screenshot (2288)" src="https://github.com/user-attachments/assets/54e5b9f2-021c-4fc6-9bc6-dc66630a974b" />

<img width="1920" height="905" alt="Screenshot (2289)" src="https://github.com/user-attachments/assets/2b176b38-981f-475f-ae78-8b1969d8c04e" />

<img width="1920" height="909" alt="Screenshot (2290)" src="https://github.com/user-attachments/assets/bfe51729-1697-445e-b1a6-051911b01830" />

<img width="1920" height="902" alt="Screenshot (2291)" src="https://github.com/user-attachments/assets/b37242c6-fa04-4334-a903-a63bf2c63bd1" />







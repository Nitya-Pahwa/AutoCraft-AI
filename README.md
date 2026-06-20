<<<<<<< HEAD
# Agentic AutoML Pipeline

A minimal working agentic AutoML system for tabular CSV/Excel datasets.

## What It Does

1. Profiles the dataset.
2. Detects classification or regression.
3. Selects a PyTorch model.
4. Tunes hyperparameters with Optuna.
5. Trains and evaluates the model.
6. Detects overfitting.
7. Uses a critic step to retry with improved hyperparameters.
8. Stores experiment summaries and logs runs with MLflow.

## Project Structure

```text
automl_agent/
  agents/
    analyst.py
    selector.py
    trainer.py
    evaluator.py
    critic.py
    memory.py
  core/
    graph.py
    state.py
    models.py
    llm.py
  utils/
    data_loader.py
    preprocessing.py
    visualizer.py
    mlflow_tracker.py
  ui/
    app.py
scripts/
  make_architecture_png.py
run_pipeline.py
requirements-minimal.txt
```

## Run

```bash
pip install -r requirements-minimal.txt
streamlit run automl_agent/ui/app.py
```

If LangGraph raises an import error involving `langgraph.checkpoint`, refresh the paired packages:

```bash
pip install --upgrade langgraph langgraph-checkpoint
```

## CLI Example

```bash
python run_pipeline.py --data path/to/data.csv --target target_column --name demo
```

## Architecture Diagram

Generate the PNG:

```bash
python scripts/make_architecture_png.py
```

Output:

```text
docs/architecture.png
```
=======
# AutoCraft-AI
>>>>>>> 8cf1599b812f1ab50d8c6c024f43f413e41ad8f5

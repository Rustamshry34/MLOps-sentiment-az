# End-to-End MLOps Pipeline for Azerbaijani Sentiment Analysis

[![Train and Deploy Azerbaijani Sentiment Model](https://github.com/Rustamshry34/MLOps-sentiment-az/actions/workflows/train-deploy.yml/badge.svg)](https://github.com/Rustamshry34/MLOps-sentiment-az/actions/workflows/train-deploy.yml)

This project implements a production-grade, fully automated MLOps pipeline for sentiment classification of Azerbaijani-language text. Built with reproducibility, CI/CD, and cloud deployment in mind, the system leverages modern open-source tools to manage data, training, evaluation, model registry, and inference serving—all on CPU-based infrastructure.

## 🔧 Core Components & Architecture

### Data Management with DVC
 - Raw data (from Hugging Face Hub) is fetched via a get_data stage.
 - Processed datasets are tracked using DVC, ensuring full reproducibility without storing large files in Git.
 - The pipeline (dvc.yaml) defines three stages: get_data → preprocess → train.

### Model Training & Evaluation
 - A Multinomial Naive Bayes classifier is trained on TF-IDF features.
 - Text preprocessing includes custom Azerbaijani-aware cleaning (lowercasing, punctuation normalization, etc.).
 - Comprehensive metrics (accuracy, cross-validation scores, classification report) are logged.

### Experiment Tracking & Model Registry with MLflow
 - All runs are tracked in MLflow (remote server hosted on AWS EC2).
 - Models are registered in the MLflow Model Registry under the name az_sentiment_nb.
 - Artifacts (including the TF-IDF vectorizer) are stored in Amazon S3.
 - Post-training, a script (promote_model.py) automatically promotes the model to the @production alias if test accuracy exceeds a threshold (e.g., 0.85).

### CI/CD with GitHub Actions
 - On every push to main, the pipeline:
 - Installs dependencies
 - Runs dvc repro to train the model
 - Executes unit tests (e.g., test_inference.py)
 - Builds and pushes a Docker image to GitHub Container Registry (GHCR)
 - Promotes the model to @production if metrics pass
 - Deploys the new image to an AWS EC2 instance via SSH
 
### Inference Serving with FastAPI
 - A lightweight FastAPI service serves predictions at /predict.
 - At startup, it loads the latest production model from MLflow using the models:/az_sentiment_nb@production URI.
 - The vectorizer is also fetched from MLflow artifacts, ensuring training-serving consistency.
 - Includes health checks (/health) and robust error handling.

### Infrastructure & Security
 - No GPU required: Entire pipeline runs on CPU.
 - Secure deployment:
    - EC2 instance uses an IAM role (no hardcoded credentials)
    - Docker container runs as a non-root user
Reproducibility: Fixed Python version (3.10), pinned dependencies, and DVC-managed data.
📦 Key Files & Structure

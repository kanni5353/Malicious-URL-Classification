# Trying-ML-

# URL Classifier using Batch ML Models

This project classifies URLs as malicious or benign using batch-trained machine learning models. It handles sparse input in SVM-light format and supports evaluation across multiple days.

## Features
- Batch loading and padding of sparse data
- Models: Logistic Regression, SVM, LightGBM, Ridge, SGD, Passive Aggressive
- Confusion matrix and performance visualization
- Easily extensible to CI/CD (SonarQube, Jenkins)

## Run It
```bash
pip install -r requirements.txt
python src/batch_url_classifier.py

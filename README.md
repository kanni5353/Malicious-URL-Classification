# Malicious URL Classifier using Batch Machine Learning

This project helps classify URLs as **malicious** or **benign** using machine learning. The dataset is large and stored in SVM-light format, so we process it in **batches by day** to manage memory and speed up training.

---

## What This Project Does

- Loads URL data from SVM-light format files.
- Prepares batches of data (like Day0, Day1, etc.)
- Trains different ML models like Logistic Regression, SVM, and LightGBM
- Evaluates how each model performs on accuracy, F1-score, and more
- Saves the results and shows graphs to compare models

---

## How It Works

1. Data is loaded day-by-day (small files like `Day0_mini.svm`)
2. Features are padded so all batches have the same number of columns
3. We train models and compare them using test data
4. The best model's confusion matrix is shown
5. Results are saved to a CSV file and plotted in graphs

---

## Models Used

- Logistic Regression
- Linear SVM
- LightGBM
- SGD Classifier
- Ridge Classifier
- Passive Aggressive Classifier

---

## How to Run It

1. Put your `.svm` files in a folder called `url_svmlight`
2. Make sure your file names are like `Day0_mini.svm`, `Day1_mini.svm`, etc.
3. Run the script using:

```python
run_classification()
```

This will:
- Load and process the data in batches
- Train models
- Show results and save them in a CSV

---

## What You Need

Install required libraries first:

```bash
pip install scikit-learn lightgbm imbalanced-learn matplotlib seaborn tqdm pandas numpy scipy
```

---

## Outputs You’ll See

- Accuracy, F1, and AUC scores
- Confusion matrix of best model
- Plots of model performance
- A CSV with all results

---

## Notes

This is a beginner-friendly project for working with high-dimensional URL data. It's useful for learning how to handle sparse data and batch processing.

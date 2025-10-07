# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold 
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgb
from scipy.sparse import csr_matrix, hstack, vstack
from tqdm import tqdm
import os
import gc
import traceback
from datetime import datetime


class BatchURLClassifier:
    def __init__(self,feature_selection="chi2", data_dir="url_svmlight", batch_size=30,k_best=300):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.models = {}
        self.results = {}
        self.batch_results = []
        self.max_features = None
        self.feature_selection = feature_selection
        self.k_best = k_best
        print(f"Initializing URL Classifier with Feature Selection: {feature_selection.upper()} | Top K: {k_best}")

    def get_max_features(self):
        if self.max_features is None:
            max_dim = 0
            print("Scanning dataset for maximum feature dimension...")
            for day in tqdm(range(6)):
                try:
                    X, _ = self.load_day_data(day)
                    max_dim = max(max_dim, X.shape[1])
                    del X
                    _ = gc.collect()
                except Exception as e:
                    print(f"Warning: Could not load Day {day}: {str(e)}")
            self.max_features = max_dim
        return self.max_features

    def load_day_data(self, day_number):
        file_path = os.path.join(self.data_dir, f"Day{day_number}_mini.svm")
        try:
            X, y = load_svmlight_file(file_path)
            return X, y
        except Exception as e:
            raise RuntimeError(f"Error loading {file_path}: {str(e)}")

    def pad_features(self, X, max_features):
        if X.shape[1] < max_features:
            padding = csr_matrix((X.shape[0], max_features - X.shape[1]))
            X = hstack([X, padding])
        elif X.shape[1] > max_features:
            X = X[:, :max_features]
        return X

    def apply_feature_selection(self, X_train, y_train, X_test):
        print("🧹 Applying VarianceThreshold to remove near-zero variance features...")
        vt = VarianceThreshold(threshold=1e-5)
        X_train = vt.fit_transform(X_train)
        X_test = vt.transform(X_test)
        print(f"🔢 Features after VarianceThreshold: {X_train.shape[1]}")
        if self.feature_selection == 'l1':
            selector = SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000), max_features=self.k_best)
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)
        elif self.feature_selection == 'l2':
            selector = SelectFromModel(LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000), max_features=self.k_best)
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)
        return X_train, X_test

    def load_batch_data(self, start_day, end_day):
        X_list = []
        y_list = []
        max_features = self.get_max_features()

        for day in tqdm(range(start_day, end_day + 1), desc=f"Loading days {start_day}-{end_day}"):
            try:
                X_day, y_day = self.load_day_data(day)
                X_day = self.pad_features(X_day, max_features)
                y_day = (y_day + 1) / 2
                X_list.append(X_day)
                y_list.append(y_day)
            except Exception as e:
                print(f"Warning: Could not load Day {day}: {str(e)}")

        if not X_list:
            raise ValueError("No data could be loaded for this batch")

        X = vstack(X_list)
        y = np.concatenate(y_list)
        return X, y

    def train_evaluate_models(self, X_train, X_test, y_train, y_test):
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Linear SVM': LinearSVC(max_iter=1000),
            'LightGBM': lgb.LGBMClassifier(n_estimators=100, force_row_wise=True),
            'SGD (Logistic)': SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3),
            'Ridge Classifier': RidgeClassifier(),
            'Passive Aggressive': PassiveAggressiveClassifier(max_iter=1000, tol=1e-3),
        }

        results = {}
        for name, model in models.items():
            try:
                print(f"\nTraining {name}...")
                start_time = datetime.now()
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
                y_train_pred = model.predict(X_train)

                train_f1 = f1_score(y_train, y_train_pred)
                test_f1 = f1_score(y_test, y_test_pred)

                print(f"{name} Results:")
                print(f"- accuracy: {accuracy_score(y_test, y_test_pred):.3f}")
                print(f"- precision: {precision_score(y_test, y_test_pred):.3f}")
                print(f"- recall: {recall_score(y_test, y_test_pred):.3f}")
                print(f"- f1: {test_f1:.3f}")
                print(f"- roc_auc: {roc_auc_score(y_test, y_test_pred):.3f}")
                print(f"- training_time: {(datetime.now() - start_time).total_seconds():.2f} seconds")
                print(f"Overfitting Test: train_f1 = {train_f1:.3f}, test_f1 = {test_f1:.3f}, gap = {train_f1 - test_f1:.3f}")

                results[name] = {
                    'test_accuracy': accuracy_score(y_test, y_test_pred),
                    'test_precision': precision_score(y_test, y_test_pred),
                    'test_recall': recall_score(y_test, y_test_pred),
                    'test_f1': test_f1,
                    'overfit_gap': train_f1 - test_f1,
                    'roc_auc': roc_auc_score(y_test, y_test_pred),
                    'training_time': (datetime.now() - start_time).total_seconds(),
                    'confusion_matrix': confusion_matrix(y_test, y_test_pred)
                }

            except Exception as e:
                results[name] = {'error': str(e)}
        return results

    def process_batch(self, start_day, end_day):
        try:
            X, y = self.load_batch_data(start_day, end_day)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
            X_train, X_test = self.apply_feature_selection(X_train, y_train, X_test)
            results = self.train_evaluate_models(X_train, X_test, y_train, y_test)
            return results
        except Exception as e:
            print(f"Error in batch {start_day}-{end_day}: {e}")
            traceback.print_exc()
            return None

    def process_all_batches(self):
        for batch_start in range(0, 6, self.batch_size):
            batch_end = min(batch_start + self.batch_size - 1, 5)
            result = self.process_batch(batch_start, batch_end)
            if result:
                self.batch_results.append({
                    'batch_start': batch_start,
                    'batch_end': batch_end,
                    'results': result
                })
                print(f"\n✅ Finished Batch: Days {batch_start}-{batch_end}")
        return self.batch_results

    def plot_batch_results(self):
        if not self.batch_results:
            print("No results to plot")
            return

        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'overfit_gap', 'roc_auc']
        models = list(self.batch_results[0]['results'].keys())

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()

        for i, metric in enumerate(metrics):
            ax = axes[i]
            for model in models:
                values = [batch['results'][model][metric] for batch in self.batch_results if 'error' not in batch['results'][model]]
                ax.plot(range(1, len(values)+1), values, label=model, marker='o')
            ax.set_title(f'{metric.upper()} across batches')
            ax.set_xlabel('Batch Number')
            ax.set_ylabel(metric)
            ax.legend()

        plt.tight_layout()
        plt.show()

def run_classification():
    try:
        classifier = BatchURLClassifier(
            data_dir="data",
            batch_size=2,
            feature_selection='l1',
            k_best=300
        )

        print(f"Starting URL Classification")
        print(f"User: kanni5353")
        print("-" * 50)

        results = classifier.process_all_batches()

        if results and len(results) > 0:
            print("\nAll batches processed successfully!")
            classifier.plot_batch_results()
        else:
            print("\u26a0\ufe0f No results were generated. Please check the error messages above.")

    except Exception as e:
        print(f"An error occurred during classification: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    run_classification()

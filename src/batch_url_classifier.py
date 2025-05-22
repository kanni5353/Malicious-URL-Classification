# Import all required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
import lightgbm as lgb
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from datetime import datetime
import os
from tqdm import tqdm
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.linear_model import RidgeClassifier, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from google.colab import files
import traceback
import gc  # Make sure gc is imported


class BatchURLClassifier:
    def __init__(self, data_dir="url_svmlight", batch_size=30):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.models = {}
        self.results = {}
        self.batch_results = []
        self.max_features = None
        print(f"Initializing URL Classifier")
        print(f"Batch Size: {batch_size} days")
        print("-" * 50)

    def get_max_features(self):
        if self.max_features is None:
            max_dim = 0
            print("Scanning dataset for maximum feature dimension...")
            for day in tqdm(range(121)):
                try:
                    X, _ = self.load_day_data(day)
                    max_dim = max(max_dim, X.shape[1])
                    del X
                    _ = gc.collect()
                except Exception as e:
                    print(f"Warning: Could not load Day {day}: {str(e)}")
            self.max_features = max_dim
            print(f"Maximum feature dimension found: {self.max_features}")
        return self.max_features

    def load_day_data(self, day_number):
        file_path = os.path.join(self.data_dir, f"Day{day_number}.svm")
        try:
            X, y = load_svmlight_file(file_path)
            return X, y
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            raise

    def pad_features(self, X, max_features):
        try:
            if X.shape[1] < max_features:
                padding = csr_matrix((X.shape[0], max_features - X.shape[1]))
                X = hstack([X, padding])
            elif X.shape[1] > max_features:
                X = X[:, :max_features]
            return X
        except Exception as e:
            print(f"Error in padding features: {str(e)}")
            raise

    def load_batch_data(self, start_day, end_day):
        X_list = []
        y_list = []
        max_features = self.get_max_features()

        for day in tqdm(range(start_day, end_day + 1), desc=f"Loading days {start_day}-{end_day}"):
            try:
                X_day, y_day = self.load_day_data(day)
                selected_indices = list(range(min(max_features, 10000)))
                X_day = X_day[:, selected_indices]
                X_day = self.pad_features(X_day, len(selected_indices))
                y_day = (y_day + 1) / 2
                X_list.append(X_day)
                y_list.append(y_day)
            except Exception as e:
                print(f"Warning: Could not load Day {day}: {str(e)}")
                continue

        if not X_list:
            raise ValueError("No data could be loaded for this batch")

        X = vstack(X_list)
        y = np.concatenate(y_list)

        print(f"Batch data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Class distribution: {np.bincount(y.astype(int))}")

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
        best_model = None
        best_f1 = 0

        for name, model in models.items():
            try:
                print(f"\nTraining {name}...")
                start_time = datetime.now()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred)
                rec = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred)

                results[name] = {
                    'accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'f1': f1,
                    'roc_auc': auc,
                    'training_time': (datetime.now() - start_time).total_seconds()
                }

                print(f"{name} Results:")
                for metric, value in results[name].items():
                    print(f"- {metric}: {value:.3f}" if metric != 'training_time' else f"- {metric}: {value:.2f} seconds")

                if f1 > best_f1:
                    best_model = name
                    best_f1 = f1
                    print("\nConfusion Matrix for best-performing model so far:")
                    cm = confusion_matrix(y_test, y_pred)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                    disp.plot(cmap='Blues')
                    plt.title(f"Confusion Matrix: {name}")
                    plt.show()

                self.models[name] = model

            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue

        print(f"\n✅ Best model this batch based on F1 score: {best_model} ({best_f1:.3f})")
        print("\n📌 Model Comparison Insight:")
        print("- Tree-based models like LightGBM often perform well because they capture non-linear relationships and feature interactions.")
        print("- Linear models (e.g., Logistic Regression, SGD) are fast and effective when feature spaces are high-dimensional but not very complex.")
        print("- Passive Aggressive may perform poorly if classes overlap too much or there's noise in features.")

        return results

    def process_batch(self, start_day, end_day):
        print(f"\nProcessing Batch: Days {start_day} to {end_day}")
        batch_start_time = datetime.now()

        try:
            X, y = self.load_batch_data(start_day, end_day)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            print(f"\nTrain set shape: {X_train.shape}")
            print(f"Test set shape: {X_test.shape}")
            results = self.train_evaluate_models(X_train, X_test, y_train, y_test)
            batch_time = (datetime.now() - batch_start_time).total_seconds()
            print(f"\nBatch processing time: {batch_time:.2f} seconds")
            del X, y, X_train, X_test, y_train, y_test
            _ = gc.collect()
            return results
        except Exception as e:
            print(f"Error processing batch {start_day}-{end_day}: {str(e)}")
            traceback.print_exc()
            return None

    def process_all_batches(self):
        total_days = 121
        total_start_time = datetime.now()
        print(f"\nStarting processing of all batches")
        print(f"Total days: {total_days}")
        print(f"Batch size: {self.batch_size}")
        print(f"Expected number of batches: {(total_days + self.batch_size - 1) // self.batch_size}")
        for batch_start in range(0, total_days, self.batch_size):
            batch_end = min(batch_start + self.batch_size - 1, total_days - 1)
            try:
                results = self.process_batch(batch_start, batch_end)
                if results:
                    self.batch_results.append({
                        'batch_start': batch_start,
                        'batch_end': batch_end,
                        'results': results
                    })
            except Exception as e:
                print(f"Error processing batch {batch_start}-{batch_end}: {str(e)}")
                continue
            _ = gc.collect()
        total_time = (datetime.now() - total_start_time).total_seconds()
        print(f"\nTotal processing time: {total_time:.2f} seconds")
        return self.batch_results

    def plot_batch_results(self):
          """Plot results across all batches"""
          if not self.batch_results:
              print("No results to plot")
              return

          metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
          models = list(self.batch_results[0]['results'].keys())

          fig, axes = plt.subplots(2, 3, figsize=(20, 12))
          axes = axes.ravel()

          for i, metric in enumerate(metrics):
              ax = axes[i]

              for model in models:
                  values = [batch['results'][model][metric]
                          for batch in self.batch_results]
                  batch_numbers = range(1, len(self.batch_results) + 1)

                  ax.plot(batch_numbers, values, label=model, marker='o')

              ax.set_title(f'{metric.upper()} across batches')
              ax.set_xlabel('Batch Number')
              ax.set_ylabel(metric)
              ax.legend()

          # Add training time plot
          ax = axes[5]
          for model in models:
              values = [batch['results'][model]['training_time']
                      for batch in self.batch_results]
              batch_numbers = range(1, len(self.batch_results) + 1)
              ax.plot(batch_numbers, values, label=model, marker='o')

          ax.set_title('Training Time across batches')
          ax.set_xlabel('Batch Number')
          ax.set_ylabel('Time (seconds)')
          ax.legend()

          plt.tight_layout()
          plt.show()
def run_classification():
    try:
        # Initialize classifier with batch size of 5
        classifier = BatchURLClassifier(data_dir="url_svmlight", batch_size=30)

        # Process all batches
        print(f"Starting URL Classification")
        print(f"Current Date and Time (UTC): 2025-05-20 18:25:29")
        print(f"User: kanni5353")
        print("-" * 50)

        results = classifier.process_all_batches()

        if results and len(results) > 0:
            # Plot results
            classifier.plot_batch_results()

            # Save results
            results_df = pd.DataFrame()
            for batch in classifier.batch_results:
                batch_df = pd.DataFrame(batch['results']).T
                batch_df['batch'] = f"Days {batch['batch_start']}-{batch['batch_end']}"
                results_df = pd.concat([results_df, batch_df])

            results_df.to_csv('url_classification_results.csv')
            files.download('url_classification_results.csv')
        else:
            print("No results were generated. Please check the error messages above.")

    except Exception as e:
        print(f"An error occurred in classification: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()

# Run the classification
run_classification()

import unittest
import os
import numpy as np
from scipy.sparse import csr_matrix
from src.batch_url_classifier import BatchURLClassifier  # Adjust this import path as needed

class TestBatchURLClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mini_data_dir = "url_svmlight"  # Directory with Day*_mini.svm files
        cls.classifier = BatchURLClassifier(data_dir=cls.mini_data_dir, batch_size=2)

    def test_max_features_positive_int(self):
        """Test max feature extraction returns valid value."""
        max_feat = self.classifier.get_max_features()
        self.assertIsInstance(max_feat, int)
        self.assertGreater(max_feat, 0)

    def test_padding_features_expand(self):
        """Ensure feature padding expands columns as expected."""
        X = csr_matrix(np.ones((5, 40)))
        padded = self.classifier.pad_features(X, 100)
        self.assertEqual(padded.shape, (5, 100))

    def test_padding_features_truncate(self):
        """Ensure padding trims down feature space if oversized."""
        X = csr_matrix(np.ones((3, 150)))
        padded = self.classifier.pad_features(X, 100)
        self.assertEqual(padded.shape, (3, 100))

    def test_load_single_day_valid(self):
        """Test if a mini dataset day file loads correctly."""
        try:
            X, y = self.classifier.load_day_data(0)
            self.assertEqual(X.shape[0], len(y))
        except Exception as e:
            self.fail(f"Failed loading day 0 mini dataset: {e}")

    def test_batch_data_loading(self):
        """Verify structure when loading a mini batch."""
        X, y = self.classifier.load_batch_data(0, 1)
        self.assertEqual(X.shape[0], len(y))
        self.assertLessEqual(X.shape[0], 400)  # each file has 200 rows

    def test_process_batch_results(self):
        """Run the full batch training pipeline with 2-day mini set."""
        results = self.classifier.process_batch(0, 1)
        self.assertIsInstance(results, dict)
        self.assertIn("Logistic Regression", results)

    def test_process_all_batches(self):
        """Ensure multiple mini batches are processed cleanly."""
        self.classifier.batch_size = 2  # keep small for test
        results = self.classifier.process_all_batches()
        self.assertTrue(results)
        self.assertGreaterEqual(len(results), 1)

    def test_plot_batch_results_runs(self):
        """Test plot generation doesn't crash with minimal batch results."""
        self.classifier.batch_results = [
            {
                'batch_start': 0,
                'batch_end': 1,
                'results': {
                    'Logistic Regression': {
                        'test_accuracy': 0.9,
                        'test_precision': 0.91,
                        'test_recall': 0.89,
                        'test_f1': 0.9,
                        'overfit_gap': 0.02,
                        'roc_auc': 0.95,
                        'training_time': 0.5,
                        'confusion_matrix': np.array([[85, 15], [10, 90]])
                    }
                }
            }
        ]
        try:
            self.classifier.plot_batch_results()  # Just ensure it runs without error
        except Exception as e:
            self.fail(f"Plotting failed with error: {e}")

if __name__ == '__main__':
    unittest.main()

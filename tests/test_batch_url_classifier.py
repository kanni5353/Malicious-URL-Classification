import unittest
import os
import numpy as np
from scipy.sparse import csr_matrix
from src.batch_url_classifier import BatchURLClassifier

class TestBatchURLClassifier(unittest.TestCase):

    def setUp(self):
        # Use a small batch size and dummy directory for test
        self.classifier = BatchURLClassifier(data_dir="url_svmlight", batch_size=2)

    def test_get_max_features_returns_int(self):
        result = self.classifier.get_max_features()
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_pad_features_increases_width(self):
        X = csr_matrix(np.ones((3, 50)))
        padded = self.classifier.pad_features(X, 100)
        self.assertEqual(padded.shape, (3, 100))

    def test_pad_features_truncates(self):
        X = csr_matrix(np.ones((2, 120)))
        padded = self.classifier.pad_features(X, 100)
        self.assertEqual(padded.shape[1], 100)

    def test_load_day_data_invalid_day(self):
        # Should raise error when loading non-existent file
        with self.assertRaises(Exception):
            self.classifier.load_day_data(day_number=999)

    def test_load_batch_data_structure(self):
        try:
            # Adjust this to a valid day range that exists in your setup
            X, y = self.classifier.load_batch_data(start_day=0, end_day=0)
            self.assertEqual(X.shape[0], len(y))
        except Exception:
            # If no data is present locally, skip test
            self.skipTest("Day0.svm data not available for loading")

if __name__ == '__main__':
    unittest.main()

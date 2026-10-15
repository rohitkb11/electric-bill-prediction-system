"""Check that the saved artifact matches the reported holdout evaluation."""
import json
import unittest
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from ml.preprocess import ROOT, FEATURES, NUMERIC, TARGET, load_dataset


class ModelArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, _ = load_dataset()
        cls.model = joblib.load(ROOT / 'models/consumption.joblib')
        cls.report = json.loads((ROOT / 'models/report.json').read_text(encoding='utf-8'))
        cls.Xtrain, cls.Xtest, cls.ytrain, cls.ytest = train_test_split(
            cls.data[FEATURES], cls.data[TARGET], test_size=.2, random_state=42)

    def test_saved_pipeline_uses_linear_regression_and_training_only_scaling(self):
        self.assertIsInstance(self.model.named_steps['model'], LinearRegression)
        numeric = self.model.named_steps['preprocess'].named_transformers_['numeric']
        np.testing.assert_allclose(numeric.named_steps['standardscaler'].mean_,
                                   self.Xtrain[NUMERIC].mean().to_numpy())
        self.assertFalse(any('Bill' in name for name in FEATURES))
        self.assertNotIn(TARGET, FEATURES)

    def test_saved_predictions_match_reported_test_scores(self):
        predicted = np.maximum(0, self.model.predict(self.Xtest))
        selected = next(row for row in self.report['models']
                        if row['name'] == self.report['selected_variant'])
        self.assertAlmostEqual(mean_absolute_error(self.ytest, predicted), selected['test']['mae'], places=3)
        self.assertAlmostEqual(r2_score(self.ytest, predicted), selected['test']['r2'], places=3)
        self.assertEqual(len(self.Xtest), 200)


if __name__ == '__main__':
    unittest.main()

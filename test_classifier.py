import unittest
from classifier import IntentClassifier, normalize, tokenize, split, evaluate, load_rows


class ClassifierTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize('كِتاب'), 'کتاب')
        self.assertEqual(tokenize('سلام! 123'), ['سلام'])

    def test_unknown(self):
        model = IntentClassifier().fit([('سلام', 'greeting'), ('مدد', 'help')])
        self.assertEqual(model.predict('xyz')['label'], 'unknown')
        self.assertEqual(model.predict('')['label'], 'unknown')
        self.assertEqual(model.predict('مدد')['label'], 'help')

    def test_split_is_reproducible_and_disjoint(self):
        rows = load_rows('sample_data.csv')
        self.assertEqual(split(rows), split(rows))
        train, test = split(rows)
        self.assertFalse(set(train) & set(test))
        self.assertEqual(set(l for _, l in train), set(l for _, l in test))

    def test_metrics(self):
        model = IntentClassifier().fit([('سلام', 'greeting'), ('مدد', 'help')])
        metrics = evaluate(model, [('سلام', 'greeting'), ('مدد', 'help')])
        self.assertEqual(metrics['macro_f1'], 1)
        self.assertEqual(metrics['accuracy'], 1)

    def test_empty_fit_rejected(self):
        with self.assertRaises(ValueError):
            IntentClassifier().fit([])


if __name__ == '__main__':
    unittest.main()

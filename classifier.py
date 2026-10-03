"""Dependency-free multinomial Naive Bayes Urdu intent demo. Not a research benchmark."""
from collections import Counter
import argparse
import csv
import json
import math
import random
import re
import unicodedata


def normalize(text):
    text = unicodedata.normalize('NFKC', text).replace('ي', 'ی').replace('ك', 'ک')
    return ''.join(c for c in text if unicodedata.category(c) != 'Mn')


def tokenize(text):
    return re.findall(r'[^\W\d_]+', normalize(text).lower(), re.UNICODE)


class IntentClassifier:
    def fit(self, rows):
        if not rows:
            raise ValueError('Training rows must not be empty')
        self.docs = Counter()
        self.counts = {}
        self.vocab = set()
        for text, label in rows:
            tokens = tokenize(text)
            if not tokens or not label.strip():
                raise ValueError('Training rows require text tokens and a label')
            self.docs[label] += 1
            self.counts.setdefault(label, Counter()).update(tokens)
            self.vocab.update(tokens)
        self.total = sum(self.docs.values())
        return self

    def predict(self, text):
        tokens = [t for t in tokenize(text) if t in self.vocab]
        if not tokens:
            return {'label': 'unknown', 'score': None, 'known_tokens': 0}
        scores = {}
        for label, counts in self.counts.items():
            score = math.log(self.docs[label] / self.total)
            denominator = sum(counts.values()) + len(self.vocab)
            scores[label] = score + sum(math.log((counts[t] + 1) / denominator) for t in tokens)
        peak = max(scores.values())
        weights = {label: math.exp(score - peak) for label, score in scores.items()}
        norm = sum(weights.values())
        label = max(sorted(weights), key=weights.get)
        return {'label': label, 'score': round(weights[label] / norm, 4), 'known_tokens': len(tokens)}


def load_rows(path):
    with open(path, encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        if not {'text', 'label'} <= set(reader.fieldnames or []):
            raise ValueError('CSV requires text,label columns')
        rows, seen = [], {}
        for line, row in enumerate(reader, 2):
            text, label = row['text'].strip(), row['label'].strip()
            key = ' '.join(tokenize(text))
            if not key or not label:
                raise ValueError(f'Empty text/label on row {line}')
            if key in seen:
                raise ValueError(f'Duplicate normalized text on row {line}; avoid split leakage')
            seen[key] = label
            rows.append((text, label))
        return rows


def split(rows, seed=42):
    rng = random.Random(seed)
    groups = {}
    for row in rows:
        groups.setdefault(row[1], []).append(row)
    if len(groups) < 2 or any(len(g) < 4 for g in groups.values()):
        raise ValueError('Need at least 2 labels and 4 distinct rows per label')
    train, test = [], []
    for label in sorted(groups):
        group = list(groups[label])
        rng.shuffle(group)
        size = max(1, round(len(group) * 0.25))
        test.extend(group[:size])
        train.extend(group[size:])
    return train, test


def evaluate(model, rows):
    labels = sorted(model.docs)
    matrix = {true: {pred: 0 for pred in labels + ['unknown']} for true in labels}
    for text, label in rows:
        matrix[label][model.predict(text)['label']] += 1
    f1s = []
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[other][label] for other in labels if other != label)
        fn = sum(matrix[label][other] for other in labels + ['unknown'] if other != label)
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return {'test_rows': len(rows), 'accuracy': sum(matrix[l][l] for l in labels) / len(rows),
            'macro_f1': sum(f1s) / len(f1s), 'confusion_matrix': matrix}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='sample_data.csv')
    parser.add_argument('--text')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    try:
        rows = load_rows(args.data)
        if args.text is not None:
            result = IntentClassifier().fit(rows).predict(args.text)
            result['warning'] = 'Uncalibrated model score; toy training data, not production confidence.'
        else:
            train, test = split(rows, args.seed)
            result = evaluate(IntentClassifier().fit(train), test)
            result.update(seed=args.seed, train_rows=len(train), warning='Synthetic demo dataset. Scores are not research benchmark results.')
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

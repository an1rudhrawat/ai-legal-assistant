# Local domain-classifier evaluation

## Role in the app

The domain classifier answers a narrow routing question: does this request appear legal, non-legal, or ambiguous? It does not identify an offence, verify a legal claim, or determine which law applies. An ambiguous classification is intended to lead to clarification or evidence checks, not an automatic legal conclusion.

The model uses `all-MiniLM-L6-v2` sentence embeddings and a scikit-learn Logistic Regression classifier. Classifier training examples are stored in `data/domain_classifier/{train,validation,test}.csv`; generated model files are stored in `data/domain_classifier/model` and are not committed.

At runtime, the configured thresholds are:

- Legal probability at or above `DOMAIN_LEGAL_THRESHOLD` (default `0.70`) → `legal`.
- Legal probability at or below `DOMAIN_NON_LEGAL_THRESHOLD` (default `0.35`) → `non_legal`.
- Probability between the two thresholds → `ambiguous`.

If the classifier or its local embedding model cannot be loaded, runtime classification conservatively returns `ambiguous`. The app does not download classifier models while serving requests.

## Train and evaluate

From `backend`:

```powershell
python scripts\train_domain_classifier.py --allow-download
```

`--allow-download` permits Sentence Transformers to download the named embedding model if it is not cached. If the model is already cached, omit the flag to keep the command local:

```powershell
python scripts\train_domain_classifier.py
```

The script trains on `train.csv`, reports validation and held-out test metrics, then writes `classifier.joblib` under `data/domain_classifier/model`.

## Recorded result and limits

The previous recorded seed-data run used 72 training, 20 validation, and 18 held-out test examples. It reported validation accuracy/precision/recall/F1 of `1.000`; test accuracy `0.889` and legal-class precision, recall, and F1 of `0.857`. Its test confusion matrix in `[legal, non_legal]` order was `[[6, 1], [1, 10]]`.

These numbers come from a very small seed set and are not a reliability claim. A held-out set of 18 examples can expose obvious errors but cannot represent all regions, languages, user styles, or legal circumstances. The classifier can still misroute both legal and non-legal questions. Update the dataset with reviewed, representative examples and inspect class-specific errors before relying on it beyond a prototype.

# Local domain classifier evaluation

The versioned seed dataset has 72 train examples, 20 validation examples, and 18 held-out test examples. It balances ordinary-language first-responder scenarios (violence, property, police, housing, work, identity, consumer, cyber, family, traffic, and government-service contexts) with programming, study, cooking, sports, entertainment, weather, travel, shopping, and technical-support requests.

The model is `all-MiniLM-L6-v2` plus `LogisticRegression(class_weight="balanced")`. Legal probability >= 0.75 is legal, <= 0.25 is non-legal, and the middle band is ambiguous. This reduces false negatives by sending uncertainty to clarification/evidence retrieval rather than declaring it out of scope.

Run the evaluation locally:

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\train_domain_classifier.py --allow-download
```

On the current seed-data training run, validation accuracy/precision/recall/F1 were `1.000`; held-out test accuracy was `0.889` and legal-class precision, recall, and F1 were each `0.857`. The held-out confusion matrix, ordered `[legal, non_legal]`, was `[[6, 1], [1, 10]]`. This means one legal false negative and one non-legal false positive in only 18 examples, so it is not a reliability claim. The seed corpus is small and not representative of Indian languages, regional phrasing, or all legal circumstances; false negatives remain a material risk and require reviewed evaluation data.

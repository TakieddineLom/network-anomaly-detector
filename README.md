# Network Anomaly Detector

A machine learning prototype that flags anomalous network traffic — built during Thirduni's AI program, trained on real intrusion-detection data.

## The interesting part

The first version of this model looked almost perfect: **99%+ precision and recall** on a held-out test split. But once evaluated on data it had never seen at all (NSL-KDD's official test set), performance dropped to **76% accuracy**, with recall on rare attack types (R2L, U2R) falling as low as **9–10%**, while common attack types (DoS, Probe) stayed above 70%.

That gap turned out to be more useful than the perfect score. It's a real, well-documented challenge in network intrusion detection: models trained on aggregate connection statistics learn to spot *loud* attacks (traffic floods, port scans) easily, but struggle with *quiet* ones (password guessing, privilege escalation) that look statistically similar to normal traffic.

| Attack category | Held-out recall | External test recall |
|---|---|---|
| DoS | 99.8% | 76.4% |
| Probe | 99.4% | 71.7% |
| R2L | 95.1% | 10.1% |
| U2R | 50.0% | 9.0% |

## What's in this repo

- **`src/data_loader.py`** — loads NSL-KDD (auto-detects `.txt`/`.arff`/`.csv`, both the 42- and 43-column formats), or falls back to synthetic data if no dataset is present
- **`src/classifier.py`** — trains a Random Forest on the labeled data, evaluates on a held-out split *and* an external test set, and breaks results down by attack category (DoS/Probe/R2L/U2R) rather than just one aggregate number

## The dataset: NSL-KDD

[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) is a widely-used benchmark for network intrusion detection — a cleaned-up version of the older KDD'99 dataset, built to remove redundant records that biased older models. Each row is a network connection described by 41 features (duration, byte counts, protocol type, service, error rates, etc.), labeled as either `normal` or a specific attack type (grouped into four categories: DoS, Probe, R2L, U2R).

One property worth knowing: NSL-KDD's training set is roughly **46% labeled anomalies** — built that way intentionally to train classifiers, not to mimic real-world traffic where attacks are rare. That's part of why a supervised model (which uses the labels directly) performs far better here than an unsupervised one.

## Setup

```bash
pip install -r requirements.txt
```

Download NSL-KDD's training file (`KDDTrain+`) and, optionally, the test file (`KDDTest+`) from Kaggle, and drop them into `data/`. Then run:

```bash
python src/classifier.py
```

This prints precision/recall/F1, a confusion matrix, feature importances, and the attack-category breakdown shown above.

## What's next

- Investigate why R2L/U2R recall collapses on unseen data — likely candidates: class imbalance (a few hundred examples vs. tens of thousands of DoS) and attack subtypes present in the test set but absent from training
- Try rebalancing training data toward rare categories and measure the effect
- Build out an interactive dashboard for exploring flagged traffic

## Team

Open to collaborators, especially anyone with a cybersecurity or networking background interested in digging into the R2L/U2R gap. Reach out if you want to build on this together.

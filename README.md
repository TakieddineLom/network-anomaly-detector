# Network Anomaly Detection — Starter Project

A working prototype that flags anomalous network traffic, with two
models to compare: an unsupervised baseline (Isolation Forest) and a
supervised classifier (Random Forest) that performs much better on
NSL-KDD specifically, since real labels are available.

## Project structure

```
network_anomaly_detector/
├── README.md
├── requirements.txt
├── data/                  # put your dataset file(s) here
├── src/
│   ├── data_loader.py     # loads + cleans traffic data (NSL-KDD or synthetic)
│   └── classifier.py      # supervised Random Forest + attack-category breakdown
└── output/                # results get saved here
```

## Setup

```bash
cd network_anomaly_detector
pip install -r requirements.txt
```

## Getting data: NSL-KDD

1. Search "NSL-KDD dataset" on Kaggle and download the training file
   (`KDDTrain+.txt`, `.arff`, or `.csv` — any format works).
2. Drop it into `data/`. Common filenames are auto-detected.
3. Optionally, also download **`KDDTest+`** and drop it in `data/` for
   external validation on data the model never saw during training.
4. Run `python src/classifier.py`.

Both the 42-column (`class`: normal/anomaly) and 43-column (`label`:
specific attack name + `difficulty`) NSL-KDD formats are handled
automatically, including raw `.arff` files.

If no file is found, `data_loader.py` falls back to synthetic data so
the pipeline always runs.

## Running it

```bash
python src/classifier.py
```

This will:
- Train a Random Forest on 80% of the data, evaluate on the held-out 20%
- Print precision/recall/F1, a confusion matrix, and feature importances
- **Break results down by attack category** (DoS/Probe/R2L/U2R) so you
  can see which types of attacks the model catches well vs poorly, not
  just one aggregate number
- List the specific attack types it missed most often
- If a `KDDTest+` file is present, run a second, more honest evaluation
  on data the model never saw during training at all
- Save full results to `output/classifier_results.csv`

## Notes

- NSL-KDD's training set is roughly 46% labeled anomalies (built that
  way intentionally for training classifiers, not to mimic real-world
  traffic where attacks are rare) — this is why a Random Forest
  (supervised, uses the labels) works far better here than Isolation
  Forest (unsupervised, has to guess an anomaly rate).
- Expect a gap between held-out and external-test performance — NSL-KDD's
  real test set deliberately includes attack variations not seen in
  training, so a drop there is expected and is a genuine sign the
  evaluation is working, not a bug.

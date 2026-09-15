# Key Finding: Why the Model's "99% Accuracy" Wasn't the Real Story

## The investigation

The first version of this model, trained on NSL-KDD, hit 99%+ precision and recall on a held-out split of the training data — a result that looked almost too good to be true. Testing it on NSL-KDD's official external test set (data the model never saw during training) told a different story: accuracy dropped to 76%, and recall on rare attack types (R2L, U2R) collapsed to 9–10%, while common attack types (DoS, Probe) stayed above 70%.

## Hypothesis: class imbalance

The initial explanation was straightforward: R2L and U2R attacks make up a small fraction of the training data (a few hundred examples vs. tens of thousands of DoS examples), so the model likely under-learns them. The fix should be simple — give rare classes more weight during training.

## Test: weighted training

I retrained the model giving R2L examples 8x weight and U2R examples 15x weight relative to normal traffic, then compared recall on the external test set before and after.

**Result: no meaningful change.** R2L recall moved from 10.1% to 9.2% — within noise, arguably slightly worse. The weighting hypothesis was falsified.

## Re-investigation: what's actually different about the missed attacks

Looking more closely at held-out performance revealed the real clue: **R2L recall was already 95.6% on the held-out split**, dropping to only 9.2% on the external test. If this were purely an imbalance problem, held-out performance should have been limited too — it wasn't. Something about the *external test set specifically* was different.

Cross-referencing the external test set's attack labels against the training set's attack labels confirmed it: **17 distinct attack types in the test set never appear anywhere in the training data** (`apache2`, `mscan`, `processtable`, `mailbomb`, `sqlattack`, and 12 others) — accounting for 3,750 of the 12,833 anomalies in the test set (29%).

Splitting recall by whether the specific attack type was seen during training confirms the cause directly:

| | Recall |
|---|---|
| Attack types seen in training | **77.0%** |
| Attack types never seen in training | **18.7%** |

## Conclusion

The performance gap isn't a class imbalance problem — it's a **distribution shift** problem. NSL-KDD's test set is deliberately constructed to include novel attack variants absent from training, specifically to test whether a model generalizes or memorizes. Reweighting only changes how much a model cares about mistakes on data it has already seen; it cannot teach a model to recognize a pattern it has never encountered in any form.

This points to a real limitation of the supervised classification approach used here: it can only recognize attacks similar to what it was trained on. In a real deployment, new attack techniques appear constantly — meaning a system relying purely on this method would need continuous retraining as new attack types are catalogued, or would benefit from being paired with an anomaly-detection approach (flagging "this looks unlike anything normal," rather than "this matches a known attack signature") to catch genuinely novel threats.
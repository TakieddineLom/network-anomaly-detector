"""
Supervised anomaly classifier using Random Forest, with a breakdown of
results by original attack category (DoS/Probe/R2L/U2R).

Run directly: `python src/classifier.py`
"""

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import (
    load_data, load_test_data, get_feature_columns, get_attack_category,
    LABEL_COLUMN, ATTACK_TYPE_COLUMN,
)

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "classifier_results.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42


def train_and_evaluate(df: pd.DataFrame, feature_columns: list):
    X = df[feature_columns]
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test, attack_train, attack_test = train_test_split(
        X, y, df[ATTACK_TYPE_COLUMN], test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results = X_test.copy()
    results[LABEL_COLUMN] = y_test.values
    results[ATTACK_TYPE_COLUMN] = attack_test.values
    results["predicted"] = y_pred
    results["anomaly_probability"] = y_proba

    return model, results, y_test, y_pred


def print_feature_importance(model, feature_columns: list, top_n: int = 10):
    importances = sorted(zip(feature_columns, model.feature_importances_), key=lambda x: x[1], reverse=True)
    print(f"\nTop {top_n} most important features:")
    for name, score in importances[:top_n]:
        print(f"  {name:30s} {score:.3f}")


def print_attack_category_breakdown(results: pd.DataFrame, label: str = "held-out"):
    """
    Breaks down recall by broad attack category (DoS/Probe/R2L/U2R),
    so you can see which types of attacks the model catches well vs
    poorly, instead of only a single aggregate recall number.
    """
    results = results.copy()
    results["category"] = results[ATTACK_TYPE_COLUMN].apply(get_attack_category)

    print(f"\n--- Recall by attack category ({label} set) ---")
    for category in ["DoS", "Probe", "R2L", "U2R", "Unknown"]:
        subset = results[results["category"] == category]
        if len(subset) == 0:
            continue
        caught = (subset["predicted"] == 1).sum()
        total = len(subset)
        recall = caught / total if total else 0
        print(f"  {category:10s}  {caught:>5d} / {total:<5d} caught  ({recall:.1%} recall)")

    # Also show the specific attack types the model missed most, within this split
    missed = results[(results[LABEL_COLUMN] == 1) & (results["predicted"] == 0)]
    if len(missed) > 0:
        print(f"\n  Most-missed specific attack types ({label} set):")
        top_missed = missed[ATTACK_TYPE_COLUMN].value_counts().head(5)
        for name, count in top_missed.items():
            print(f"    {name:20s} missed {count} times")


def evaluate_on_external_test(model, test_df, feature_columns: list):
    X_ext = test_df[feature_columns]
    y_ext = test_df[LABEL_COLUMN]
    y_pred = model.predict(X_ext)

    print(f"\n=== Evaluation on EXTERNAL test set ({len(test_df)} rows, never seen during training) ===")
    print(classification_report(y_ext, y_pred, target_names=["normal", "anomaly"]))

    cm = confusion_matrix(y_ext, y_pred)
    print("Confusion matrix:")
    print("                 predicted normal   predicted anomaly")
    print(f"actual normal    {cm[0][0]:>15d}   {cm[0][1]:>17d}")
    print(f"actual anomaly   {cm[1][0]:>15d}   {cm[1][1]:>17d}")

    ext_results = test_df.copy()
    ext_results["predicted"] = y_pred
    print_attack_category_breakdown(ext_results, label="EXTERNAL test")


def main():
    df = load_data()
    feature_columns = get_feature_columns(df)
    print(f"Loaded {len(df)} rows of traffic data.")
    print(f"Using features: {feature_columns}")

    model, results, y_test, y_pred = train_and_evaluate(df, feature_columns)

    print(f"\n--- Evaluation on held-out test set ({len(y_test)} rows) ---")
    print(classification_report(y_test, y_pred, target_names=["normal", "anomaly"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print("                 predicted normal   predicted anomaly")
    print(f"actual normal    {cm[0][0]:>15d}   {cm[0][1]:>17d}")
    print(f"actual anomaly   {cm[1][0]:>15d}   {cm[1][1]:>17d}")

    print_feature_importance(model, feature_columns)
    print_attack_category_breakdown(results, label="held-out")

    test_df = load_test_data(feature_columns)
    if test_df is not None:
        evaluate_on_external_test(model, test_df, feature_columns)
    else:
        print("\nNo external test file found in data/ (e.g. KDDTest+.txt). Add one to validate on truly unseen data.")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results.sort_values("anomaly_probability", ascending=False).to_csv(OUTPUT_PATH, index=False)
    print(f"\nFull results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

"""
predict_en.py

Reusable scoring script for the fraud detection model, independent of the
exploratory notebook. Loads the already-trained model (fraud_model.pkl)
and exposes a scoring function ready to integrate into another system
(API, batch job, etc.).

Use as a script:
    python predict_en.py path/to/transactions.csv

Use as a module:
    from predict_en import score_transaction, load_model
    model = load_model("fraud_model.pkl")
    result = score_transaction(row, model, threshold=0.05)
"""

import sys
import joblib
import pandas as pd

MODEL_PATH = "fraud_model.pkl"
DEFAULT_THRESHOLD = 0.05  # update with the real optimal threshold from the notebook


def load_model(path: str = MODEL_PATH):
    """Loads the trained pipeline (scaler + model) from disk."""
    return joblib.load(path)


def score_transaction(row: pd.Series, model, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """
    Runs the full pipeline on a single transaction and returns a decision.

    Parameters
    ----------
    row : pd.Series
        A row with the same columns used at training time
        (Time, V1..V28, Amount).
    model : Pipeline
        Trained pipeline (scaler + classification model).
    threshold : float
        Probability threshold above which a transaction is flagged for review.

    Returns
    -------
    dict with the fraud probability and the decision ("REVIEW" / "APPROVE").
    """
    row_df = pd.DataFrame([row.values], columns=row.index)
    fraud_probability = model.predict_proba(row_df)[0][1]
    decision = "REVIEW" if fraud_probability >= threshold else "APPROVE"
    return {"fraud_probability": float(fraud_probability), "decision": decision}


def score_batch(df: pd.DataFrame, model, threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """Applies score_transaction to every row of a transactions DataFrame."""
    results = df.apply(lambda row: score_transaction(row, model, threshold), axis=1)
    return pd.DataFrame(list(results), index=df.index)


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_en.py path/to/transactions.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    model = load_model()
    df = pd.read_csv(csv_path)

    # If the CSV includes the Class column, drop it — it's not an input feature.
    df_features = df.drop(columns=["Class"], errors="ignore")

    results = score_batch(df_features, model)
    output = pd.concat([df_features, results], axis=1)

    output_path = "scoring_results.csv"
    output.to_csv(output_path, index=False)
    print(f"Scoring complete. {len(output)} transactions processed.")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

"""Trains Linear Regression + Random Forest, keeps the better one."""
import json
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join("data", "student_data.csv")
MODEL_PATH = os.path.join("model", "model.pkl")
METRICS_PATH = os.path.join("model", "metrics.json")

FEATURES = ["hours_studied", "attendance", "previous_score",
            "sleep_hours", "tutoring_sessions"]
TARGET = "final_score"


def main():
    os.makedirs("model", exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    candidates = {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)),
        ]),
    }

    print("Model comparison on the held-out test set:\n")
    results, fitted = {}, {}
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        results[name] = {
            "r2": round(float(r2_score(y_test, preds)), 4),
            "mae": round(float(mean_absolute_error(y_test, preds)), 4),
            "rmse": round(float(root_mean_squared_error(y_test, preds)), 4),
        }
        fitted[name] = pipe
        print(f"{name:<18} R2={results[name]['r2']:.3f}  "
              f"MAE={results[name]['mae']:.2f}  RMSE={results[name]['rmse']:.2f}")

    best = max(results, key=lambda k: results[k]["r2"])
    print(f"\nBest model: {best} -> saved to {MODEL_PATH}")

    joblib.dump({"model": fitted[best], "features": FEATURES}, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump({"best_model": best, "results": results, "features": FEATURES},
                  f, indent=2)


if __name__ == "__main__":
    main()
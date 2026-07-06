"""Trains and tunes three models with cross-validation, keeps the best."""
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join("data", "student_data.csv")
MODEL_PATH = os.path.join("model", "model.pkl")
METRICS_PATH = os.path.join("model", "metrics.json")

FEATURES = ["hours_studied", "attendance", "previous_score",
            "sleep_hours", "tutoring_sessions"]
TARGET = "final_score"
CV_FOLDS = 5


def build_candidates():
    """Each entry: a pipeline + a grid of hyperparameters to search."""
    return {
        "LinearRegression": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]),
            "grid": {},  # nothing to tune
        },
        "RandomForest": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
            ]),
            "grid": {
                "model__n_estimators": [200, 400],
                "model__max_depth": [8, 12, None],
                "model__min_samples_leaf": [1, 2, 4],
            },
        },
        "GradientBoosting": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("model", GradientBoostingRegressor(random_state=42)),
            ]),
            "grid": {
                "model__n_estimators": [200, 400],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        },
    }


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "r2": round(float(r2_score(y_test, preds)), 4),
        "mae": round(float(mean_absolute_error(y_test, preds)), 4),
        "rmse": round(float(root_mean_squared_error(y_test, preds)), 4),
    }


def main():
    os.makedirs("model", exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    results, fitted = {}, {}
    print(f"Tuning models with {CV_FOLDS}-fold cross-validation...\n")

    for name, spec in build_candidates().items():
        if spec["grid"]:
            search = GridSearchCV(
                spec["pipeline"], spec["grid"],
                cv=CV_FOLDS, scoring="r2", n_jobs=-1)
            search.fit(X_train, y_train)
            best_est = search.best_estimator_
            cv_r2 = round(float(search.best_score_), 4)
            best_params = {k.replace("model__", ""): v
                           for k, v in search.best_params_.items()}
        else:
            best_est = spec["pipeline"].fit(X_train, y_train)
            cv_r2 = round(float(np.mean(
                cross_val_score(best_est, X_train, y_train,
                                cv=CV_FOLDS, scoring="r2"))), 4)
            best_params = {}

        test_metrics = evaluate(best_est, X_test, y_test)
        results[name] = {"cv_r2": cv_r2, **test_metrics, "best_params": best_params}
        fitted[name] = best_est
        print(f"{name:<18} CV-R2={cv_r2:.3f}  test-R2={test_metrics['r2']:.3f}  "
              f"MAE={test_metrics['mae']:.2f}")
        if best_params:
            print(f"                   best params: {best_params}")

    # Choose the winner by cross-validated R2 (the trustworthy number).
    best = max(results, key=lambda k: results[k]["cv_r2"])
    best_model = fitted[best]
    print(f"\nBest model: {best} (CV-R2={results[best]['cv_r2']:.3f}) -> {MODEL_PATH}")

    joblib.dump({"model": best_model, "features": FEATURES}, MODEL_PATH)

    # Feature importance from whichever model won.
    inner = best_model.named_steps["model"]
    if hasattr(inner, "coef_"):
        raw = dict(zip(FEATURES, inner.coef_))
    elif hasattr(inner, "feature_importances_"):
        raw = dict(zip(FEATURES, inner.feature_importances_))
    else:
        raw = {}
    total = sum(abs(v) for v in raw.values()) or 1
    importances = {k: round(abs(v) / total * 100, 1)
                   for k, v in sorted(raw.items(),
                                      key=lambda kv: abs(kv[1]), reverse=True)}

    with open(METRICS_PATH, "w") as f:
        json.dump({"best_model": best, "results": results,
                   "features": FEATURES, "importances": importances,
                   "cv_folds": CV_FOLDS}, f, indent=2)

    print("\nFeature importance (%):")
    for k, v in importances.items():
        print(f"  {k:<18} {v}")


if __name__ == "__main__":
    main()
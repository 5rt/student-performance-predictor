"""Creates a realistic synthetic dataset of students -> data/student_data.csv"""
import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_STUDENTS = 1000
OUT_PATH = os.path.join("data", "student_data.csv")


def generate():
    rng = np.random.default_rng(RANDOM_SEED)

    hours_studied = np.clip(rng.normal(5.0, 2.0, N_STUDENTS), 0, 12)
    attendance = np.clip(rng.normal(80, 12, N_STUDENTS), 40, 100)
    previous_score = np.clip(rng.normal(65, 15, N_STUDENTS), 20, 100)
    sleep_hours = np.clip(rng.normal(7.0, 1.2, N_STUDENTS), 4, 10)
    tutoring_sessions = np.clip(rng.poisson(1.5, N_STUDENTS), 0, 8)

    # Sleep helps most around 7-8 hours, then flattens.
    sleep_effect = -1.2 * (sleep_hours - 7.5) ** 2 + 4

    final_score = (
        3
        + 1.9 * hours_studied
        + 0.15 * attendance
        + 0.42 * previous_score
        + sleep_effect
        + 1.3 * tutoring_sessions
        + rng.normal(0, 7, N_STUDENTS)
    )
    final_score = np.clip(final_score, 0, 100).round(1)

    df = pd.DataFrame({
        "hours_studied": hours_studied.round(1),
        "attendance": attendance.round(1),
        "previous_score": previous_score.round(1),
        "sleep_hours": sleep_hours.round(1),
        "tutoring_sessions": tutoring_sessions.astype(int),
        "final_score": final_score,
    })
    df["passed"] = (df["final_score"] >= 50).astype(int)
    return df


def main():
    os.makedirs("data", exist_ok=True)
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")
    print(f"Pass rate: {df['passed'].mean():.1%}")


if __name__ == "__main__":
    main()
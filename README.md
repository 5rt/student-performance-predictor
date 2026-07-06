# Student Performance Prediction System

A web-based machine learning tool that predicts a student's final exam score from their study habits, then flags whether they're on track to pass. It trains and compares three models with cross-validation and hyperparameter tuning, then serves the best one through a Flask app with a web UI and a JSON API.

**Stack:** Python · scikit-learn · Flask · pandas

<p align="center">
  <img src="https://github.com/user-attachments/assets/88cf8874-879f-498e-98cd-b6c39959c26b" width="49%" />
  <img src="https://github.com/user-attachments/assets/b12b6dfc-1b0c-40a7-9db3-05d9bfa2da29" width="49%" />
</p>

---

## Features

- Predicts a final exam score (0–100) and a pass/fail verdict from five inputs.
- Compares **Linear Regression**, **Random Forest**, and **Gradient Boosting**, selected by 5-fold cross-validated R² with `GridSearchCV` tuning.
- Explains each prediction with a **feature-importance** breakdown.
- **Server-side input validation** with clear error messages.
- **JSON API** (`/api/predict`) and a **health endpoint** (`/health`) for monitoring and deployment.

---

## How it works

1. `generate_data.py` builds a reproducible synthetic dataset of 1,000 students (fixed random seed, so results are identical on any machine).
2. `train.py` splits the data, runs 5-fold cross-validation with hyperparameter tuning across three models, and saves the best one to `model/model.pkl`.
3. `app.py` loads the saved model and serves predictions via a web form and a JSON API.

The dataset carries deliberate noise, so there's an irreducible ceiling on accuracy — the honest cross-validated R² is around 0.52. On this data the relationship is mostly linear, so Linear Regression edges out the more complex models: a reminder that the simplest model that fits is often the right choice.

---

## Inputs

| Input | Meaning |
|-------|---------|
| Hours studied | Study time per week |
| Attendance | Class attendance percentage |
| Previous term score | Last term's score (0–100) |
| Sleep hours | Average sleep per night |
| Tutoring sessions | Extra help sessions per month |

---

## Run it locally

Requires Python 3.10+.

```bash
# create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# install dependencies
pip install -r requirements.txt

# build the dataset, train the models, then start the app
python generate_data.py
python train.py
python app.py
```

Open http://localhost:5000 in your browser.

---

## API

```bash
curl -X POST http://localhost:5000/api/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"hours_studied\":8,\"attendance\":95,\"previous_score\":78,\"sleep_hours\":7.5,\"tutoring_sessions\":4}"
```

Response:

```json
{
  "predicted_score": 72.4,
  "passed": true,
  "model_used": "LinearRegression"
}
```

Out-of-range or missing values return a `400` with a clear message.

---

## Project structure

```
student-performance-predictor/
├── app.py              # Flask app: web UI + JSON API + validation
├── train.py            # Cross-validation, tuning, model comparison
├── generate_data.py    # Reproducible dataset generator
├── requirements.txt
├── data/               # Generated dataset
├── model/              # Saved model + metrics
├── templates/          # index.html
└── static/             # style.css
```

---

## Possible next steps

- Swap the synthetic data for a real academic dataset.
- Add automated tests around the prediction and validation logic.
- Deploy to a cloud host with the existing `/health` endpoint for monitoring.

---

*Built with Python, scikit-learn, and Flask.*

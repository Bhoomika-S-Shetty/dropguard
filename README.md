# DropGuard AI — Student Dropout Prediction System

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the ML model (first time only)
```bash
python train_model.py
```

### 3. Run the application
```bash
python app.py
```

### 4. Open in browser
Navigate to: **http://localhost:5050**

---

## Project Structure
```
dropguard/
├── app.py              # Flask backend + all API routes
├── train_model.py      # ML model training (Random Forest)
├── requirements.txt    # Python dependencies
├── data/
│   └── students.csv    # Generated synthetic dataset (1200 students)
├── models/
│   ├── rf_model.pkl    # Trained Random Forest model
│   ├── encoders.pkl    # Label encoders for categorical features
│   ├── metrics.json    # Model evaluation metrics
│   └── features.json   # Feature list
└── templates/
    └── index.html      # Full UI (single-page app)
```

## Features

### Dashboard
- Live student counts (total, high/medium/low risk)
- Risk distribution donut chart
- Department-wise dropout risk bar chart
- Monthly attendance vs risk trend line chart
- Random Forest feature importance visualization (Explainable AI)

### Student List
- Searchable, filterable table of 60 top-risk students
- Risk score progress bars with color coding
- Click "Predict" to pre-fill the prediction form

### Predict Risk (ML Engine)
- 10 input features: attendance, marks, assignments, backlogs, income, travel, internet, participation, stress, dept
- Real Random Forest model (scikit-learn)
- Risk score with probability output
- SHAP-style factor contribution bars (Explainable AI)
- Personalized intervention recommendations

### Interventions
- 7 intervention categories mapped to ML-detected risk signals
- Student count per intervention type

### Model Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix (horizontal bar chart)
- Algorithm comparison: Logistic Regression vs Decision Tree vs Random Forest vs XGBoost

## Tech Stack
| Component | Technology |
|-----------|------------|
| Frontend | HTML5, CSS3, Vanilla JS, Chart.js |
| Backend | Python + Flask |
| ML | Scikit-learn (Random Forest) |
| Dataset | Synthetic (1,200 students, 11 features) |
| Charts | Chart.js 4.4 |
| Icons | Font Awesome 6 |

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Main UI |
| GET | `/api/dashboard` | Summary stats + chart data |
| GET | `/api/students` | Student list (top 60 by risk) |
| POST | `/api/predict` | Run ML prediction |
| GET | `/api/interventions` | Intervention recommendations |
| GET | `/api/metrics` | Model evaluation metrics |

## Input Features
| Feature | Type | Range |
|---------|------|-------|
| Attendance | Numeric | 0–100% |
| Internal Marks | Numeric | 0–100 |
| Assignments Submitted | Integer | 0–10 |
| Active Backlogs | Integer | 0–10 |
| Family Income | Categorical | low/medium/high |
| Travel Distance | Numeric | km |
| Internet Access | Binary | yes/no |
| Participation | Categorical | none/some/active |
| Stress Level | Integer | 1–10 |
| Department | Categorical | CS/Mech/Civil/ECE/IT/EEE |
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle, json, os

# â”€â”€ Paths (works on Windows, Mac, Linux) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, 'data')
MODELS = os.path.join(BASE, 'models')
os.makedirs(DATA,   exist_ok=True)
os.makedirs(MODELS, exist_ok=True)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

np.random.seed(42)
N = 1200

def gen_dataset(n):
    attendance      = np.clip(np.random.normal(72, 15, n), 30, 100)
    internal_marks  = np.clip(np.random.normal(58, 16, n), 10, 100)
    assignments     = np.random.randint(0, 11, n)
    backlogs        = np.random.choice([0,1,2,3,4,5,6], n, p=[0.35,0.25,0.18,0.10,0.06,0.04,0.02])
    family_income   = np.random.choice(['low','medium','high'], n, p=[0.30,0.45,0.25])
    travel_distance = np.clip(np.random.exponential(15, n), 1, 80)
    internet        = np.random.choice(['yes','no'], n, p=[0.72,0.28])
    participation   = np.random.choice(['none','some','active'], n, p=[0.40,0.38,0.22])
    stress          = np.random.randint(1, 11, n)
    dept            = np.random.choice(['CS','Mechanical','Civil','ECE','IT','EEE'], n)

    risk = np.zeros(n)
    risk += np.where(attendance < 60, 35, np.where(attendance < 75, 20, 5))
    risk += np.where(internal_marks < 50, 25, np.where(internal_marks < 65, 12, 2))
    risk += backlogs * 7
    risk += np.where(family_income == 'low', 14, np.where(family_income == 'medium', 5, 0))
    risk += np.where(travel_distance > 30, 9, np.where(travel_distance > 15, 4, 0))
    risk += np.where(internet == 'no', 7, 0)
    risk += np.where(participation == 'none', 5, np.where(participation == 'some', 2, 0))
    risk += np.maximum(0, stress - 5) * 3
    risk += np.random.normal(0, 5, n)

    dropout = (risk >= 55).astype(int)

    return pd.DataFrame({
        'attendance':      attendance.round(1),
        'internal_marks':  internal_marks.round(1),
        'assignments':     assignments,
        'backlogs':        backlogs,
        'family_income':   family_income,
        'travel_distance': travel_distance.round(1),
        'internet':        internet,
        'participation':   participation,
        'stress':          stress,
        'dept':            dept,
        'risk_score':      np.clip(risk, 0, 100).round(1),
        'dropout':         dropout
    })

# â”€â”€ Generate or load existing dataset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
csv_path = os.path.join(DATA, 'students.csv')
if os.path.exists(csv_path):
    print(f"Loading existing dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} rows loaded.")
else:
    print("No existing dataset found â€” generating synthetic data...")
    df = gen_dataset(N)
    df.to_csv(csv_path, index=False)
    print(f"  {len(df)} rows saved to: {csv_path}")

# â”€â”€ Encode categoricals â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
le_income        = LabelEncoder().fit(['high','low','medium'])
le_internet      = LabelEncoder().fit(['no','yes'])
le_participation = LabelEncoder().fit(['active','none','some'])
le_dept          = LabelEncoder().fit(['CS','Civil','ECE','EEE','IT','Mechanical'])

df_enc = df.copy()
df_enc['family_income']  = le_income.transform(df['family_income'])
df_enc['internet']       = le_internet.transform(df['internet'])
df_enc['participation']  = le_participation.transform(df['participation'])
df_enc['dept']           = le_dept.transform(df['dept'])

features = ['attendance','internal_marks','assignments','backlogs',
            'family_income','travel_distance','internet','participation','stress','dept']
X = df_enc[features]
y = df_enc['dropout']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# â”€â”€ Train models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def get_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred).tolist()
    return {
        'accuracy':         round(accuracy_score(y_test, y_pred) * 100, 1),
        'precision':        round(precision_score(y_test, y_pred, zero_division=0) * 100, 1),
        'recall':           round(recall_score(y_test, y_pred, zero_division=0) * 100, 1),
        'f1':               round(f1_score(y_test, y_pred, zero_division=0) * 100, 1),
        'confusion_matrix': cm
    }

print("\nTraining models...")
rf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
rf.fit(X_train, y_train)

lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_train, y_train)

dt = DecisionTreeClassifier(max_depth=8, random_state=42)
dt.fit(X_train, y_train)

metrics = {
    'random_forest':       get_metrics(rf, X_test, y_test),
    'logistic_regression': get_metrics(lr, X_test, y_test),
    'decision_tree':       get_metrics(dt, X_test, y_test),
}
fi = dict(zip(features, rf.feature_importances_.tolist()))
metrics['feature_importance'] = fi

# â”€â”€ Save artifacts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with open(os.path.join(MODELS, 'rf_model.pkl'), 'wb') as f:
    pickle.dump(rf, f)
with open(os.path.join(MODELS, 'encoders.pkl'), 'wb') as f:
    pickle.dump({'income': le_income, 'internet': le_internet,
                 'participation': le_participation, 'dept': le_dept}, f)
with open(os.path.join(MODELS, 'metrics.json'), 'w') as f:
    json.dump(metrics, f)
with open(os.path.join(MODELS, 'features.json'), 'w') as f:
    json.dump(features, f)

print("\nTraining complete!")
for k, v in metrics.items():
    if k != 'feature_importance':
        print(f"  {k:22s}: acc={v['accuracy']}%  prec={v['precision']}%  rec={v['recall']}%  f1={v['f1']}%")
print(f"\nModel files saved to: {MODELS}")
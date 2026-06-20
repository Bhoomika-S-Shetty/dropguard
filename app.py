from flask import Flask, render_template, request,redirect, jsonify,session
import pickle, json, os
import pandas as pd
import numpy as np
import csv
import os

app = Flask(__name__)
app.secret_key = "dropguard_secret"
BASE = os.path.dirname(__file__)
faculty_accounts=[]
# Load model & encoders
with open(os.path.join(BASE,'models','rf_model.pkl'),'rb') as f:
    model = pickle.load(f)
with open(os.path.join(BASE,'models','encoders.pkl'),'rb') as f:
    encoders = pickle.load(f)
print(encoders['income'].classes_)
print(encoders['internet'].classes_)
print(encoders['participation'].classes_)
print(encoders['dept'].classes_)
with open(os.path.join(BASE,'models','metrics.json')) as f:
    metrics = json.load(f)
with open(os.path.join(BASE,'models','features.json')) as f:
    features = json.load(f)
df_all = pd.read_csv(os.path.join(BASE,'data','students.csv'))

def encode_input(data):
    row = {
        'attendance':     float(data['attendance']),
        'internal_marks': float(data['internal_marks']),
        'assignments':    int(data['assignments']),
        'backlogs':       int(data['backlogs']),
        'family_income':  encoders['income'].transform([data['family_income']])[0],
        'travel_distance':float(data['travel_distance']),
        'internet':       encoders['internet'].transform([data['internet']])[0],
        'participation':  encoders['participation'].transform([data['participation']])[0],
        'stress':         int(data['stress']),
        'dept':           encoders['dept'].transform([data['dept']])[0],
    }
    return pd.DataFrame([row])[features]

def get_interventions(data, risk_pct, level):
    intv = []
    attend = float(data['attendance'])
    marks  = float(data['internal_marks'])
    backlogs = int(data['backlogs'])
    income = data['family_income']
    stress = int(data['stress'])
    travel = float(data['travel_distance'])
    internet = data['internet']

    if attend < 65:
        intv.append({'icon':'phone','color':'#E24B4A','title':'Immediate Parent Meeting',
            'desc':f"Attendance is critically low at {attend:.0f}%. Schedule urgent parent/guardian meeting and create a weekly attendance recovery plan.",'priority':'Critical'})
    elif attend < 75:
        intv.append({'icon':'calendar','color':'#EF9F27','title':'Attendance Monitoring',
            'desc':f"Attendance at {attend:.0f}% needs improvement. Assign a faculty mentor for bi-weekly check-ins and track punctuality.",'priority':'High'})

    if marks < 50:
        intv.append({'icon':'book','color':'#378ADD','title':'Remedial Tutoring Enrollment',
            'desc':f"Internal marks at {marks:.0f}/100 indicate academic difficulty. Enroll in peer tutoring and faculty-led remedial sessions for weak subjects.",'priority':'Critical'})
    elif marks < 65:
        intv.append({'icon':'pencil','color':'#7F77DD','title':'Study Skills Workshop',
            'desc':f"Marks at {marks:.0f}/100 can improve. Recommend time-management workshops and structured study groups.",'priority':'Medium'})

    if backlogs >= 3:
        intv.append({'icon':'alert-triangle','color':'#E24B4A','title':'Backlog Clearance Plan',
            'desc':f"{backlogs} active backlogs detected. Create a structured re-exam schedule, inform HOD, and assign dedicated subject mentors.",'priority':'Critical'})
    elif backlogs >= 1:
        intv.append({'icon':'clipboard-list','color':'#EF9F27','title':'Backlog Resolution',
            'desc':f"{backlogs} backlog(s) pending. Prepare a study plan targeting supplementary exams.",'priority':'High'})

    if income == 'low':
        intv.append({'icon':'cash','color':'#1D9E75','title':'Scholarship & Financial Aid',
            'desc':'Family income below ₹2L/yr. Refer to NSP, state merit scholarships, college fee waivers, and welfare fund emergency grants.','priority':'High'})

    if stress >= 7:
        intv.append({'icon':'heart','color':'#D4537E','title':'Counselor Session',
            'desc':f"Stress level at {stress}/10. Schedule confidential counseling session. Share mental wellness resources and stress management techniques.",'priority':'High'})

    if travel > 30:
        intv.append({'icon':'bus','color':'#854F0B','title':'Transport / Hostel Support',
            'desc':f"Travel distance of {travel:.0f}km is significant. Evaluate eligibility for hostel accommodation or transport allowance.",'priority':'Medium'})

    if internet == 'no':
        intv.append({'icon':'wifi','color':'#533AB7','title':'Digital Access Support',
            'desc':'No internet access detected. Connect to college computer lab extended hours and digital literacy programs or subsidized data schemes.','priority':'Medium'})

    if not intv:
        intv.append({'icon':'check','color':'#3B6D11','title':'Continue Regular Monitoring',
            'desc':'Student appears to be on track. Schedule monthly check-ins and encourage continued participation in academics and extracurriculars.','priority':'Low'})

    return intv

def shap_contributions(data):
    attend = float(data['attendance'])
    marks  = float(data['internal_marks'])
    backlogs = int(data['backlogs'])
    income = data['family_income']
    stress = int(data['stress'])
    travel = float(data['travel_distance'])
    internet = data['internet']

    contrib = {
        'Attendance':     round(max(0, (75 - attend) * 0.85), 1) if attend < 75 else 0,
        'Internal Marks': round(max(0, (65 - marks) * 0.45), 1) if marks < 65 else 0,
        'Backlogs':       round(backlogs * 5.5, 1),
        'Family Income':  12 if income == 'low' else (4 if income == 'medium' else 0),
        'Stress Level':   round(max(0, stress - 5) * 3.2, 1),
        'Travel Distance':round(max(0, (travel - 15) * 0.18), 1) if travel > 15 else 0,
        'No Internet':    6 if internet == 'no' else 0,
    }
    total = sum(contrib.values()) or 1
    return {k: round(v / total * 100, 1) for k, v in contrib.items()}

@app.route('/')
def home():
    return render_template('login.html')
@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('username')
    password = request.form.get('password')

    with open('data/faculty.csv', 'r', newline='') as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row['email'] == email and row['password'] == password:
                session['faculty_name'] = row['name']
                return redirect('/dashboard')

    return "Invalid Email or Password"
@app.route('/logout')
def logout():
    return redirect('/')
@app.route('/dashboard')
def dashboard_page():
    if 'faculty_name' not in session:
        return redirect('/')

    return render_template('index.html')
@app.route('/signup',methods=['GET'])
def signup():
    return render_template('signup.html')
@app.route('/signup',methods=['POST'])
def signup_post():
    name = request.form['name']
    faculty_id = request.form['faculty_id']
    department = request.form['department']
    email = request.form['email']
    password = request.form['password']

    with open('data/faculty.csv', 'a', newline='') as file:
        writer = csv.writer(file)

        writer.writerow([
            request.form['name'],
            request.form['faculty_id'],
            request.form['department'],
            request.form['email'],
            request.form['password']
        ])

    return redirect('/')
@app.route('/forgot-password')
def forgot_password():
    return render_template('forgot_password.html')
@app.route('/reset-password', methods=['POST'])
def reset_password():

    email = request.form['email']
    new_password = request.form['new_password']

    df = pd.read_csv('data/faculty.csv')

    if email in df['email'].values:

        df.loc[df['email'] == email, 'password'] = new_password
        df.to_csv('data/faculty.csv', index=False)

        return "Password Reset Successful. <a href='/'>Login</a>"

    return "Email Not Found"
@app.route('/api/dashboard')
def dashboard():
    df = df_all.copy()
    def classify(r):
        if r >= 60: return 'High'
        elif r >= 35: return 'Medium'
        return 'Low'
    df['risk_level'] = df['risk_score'].apply(classify)

    dept_risk = df.groupby('dept')['dropout'].mean().mul(100).round(1).to_dict()
    risk_dist  = df['risk_level'].value_counts().to_dict()
    monthly_attendance = [82.1,78.4,74.3,70.8,67.5,72.1,75.3,71.9]
    monthly_risk       = [21.4,27.8,34.1,41.5,47.9,44.2,39.6,43.1]

    fi = metrics['feature_importance']
    fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    feature_importance = [{'name': k.replace('_',' ').title(), 'value': round(v*100,1)} for k,v in fi_sorted]

    return jsonify({
        'totals': {
            'students': len(df),
            'high_risk': int((df['risk_level']=='High').sum()),
            'medium_risk': int((df['risk_level']=='Medium').sum()),
            'low_risk': int((df['risk_level']=='Low').sum()),
        },
        'risk_distribution': risk_dist,
        'dept_risk': dept_risk,
        'monthly_attendance': monthly_attendance,
        'monthly_risk': monthly_risk,
        'feature_importance': feature_importance,
    })

@app.route('/api/students')
def students():
    df=pd.read_csv(os.path.join(BASE,"data","students.csv"))
    df['risk_level'] = df['risk_score'].apply(
        lambda r: 'High' if r >= 65 else ('Medium' if r >= 35 else 'Low')
    )
    if 'student_name' in df.columns:
        df['name'] = df['student_name']
    #df['name'] = [f"Student {chr(65 + i%26)}{i+100}" for i in range(len(df))]
    if 'student_id' in df.columns:
        df['id'] = df['student_id']
    #df['id'] = [f"{row.dept}{2300+i}" for i, row in enumerate(df.itertuples())]
    df=df.fillna(0)
    df = df.fillna('')
    records=df.to_dict(orient='records')
    return jsonify(records)
@app.route('/add_student', methods=['POST'])
def add_student():

    data = request.json
    prediction_data = {
    'attendance': data['attendance'],
    'internal_marks': data['marks'],
    'assignments': data['assignments'],
    'backlogs': data['backlogs'],
    'family_income': data['income'],
    'travel_distance': data['travel'],
    'internet': data['internet'],
    'participation': data['participation'],
    'stress': data['stress'],
    'dept': data['department']
}
    x = encode_input(prediction_data)
    prob = model.predict_proba(x)[0]
    risk_pct = round(prob[1] * 100, 1)
    dropout = 1 if risk_pct >= 50 else 0
    new_student = {
        "student_name": data['name'],
        "student_id": data['usn'],
        "dept": data['department'],
        "attendance": float(data['attendance']),
        "internal_marks": float(data['marks']),
        "backlogs": int(data['backlogs']),
        "assignments": int(data['assignments']),
        "family_income": data['income'],
        "travel_distance": float(data['travel']),
        "internet": data['internet'],
        "participation": data['participation'],
        "stress": int(data['stress']),
        "risk_score":risk_pct,
        "dropout":dropout
    }

    global df_all
    df_all = pd.concat(
        [df_all, pd.DataFrame([new_student])],
        ignore_index=True
    )
    df_all.to_csv(
        os.path.join(BASE, 'data', 'students.csv'),
        index=False
    )

    return jsonify({
        "message": "Student added successfully"
    })
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    X = encode_input(data)
    prob = model.predict_proba(X)[0]
    risk_pct = round(prob[1] * 100, 1)
    level = 'High' if risk_pct >= 65 else ('Medium' if risk_pct >= 35 else 'Low')
    interventions = get_interventions(data, risk_pct, level)
    contributions = shap_contributions(data)
    return jsonify({
        'risk_pct': risk_pct,
        'level': level,
        'interventions': interventions,
        'contributions': contributions,
    })

@app.route('/api/metrics')
def get_metrics():
    return jsonify(metrics)

@app.route('/api/interventions')
def interventions_summary():
    df = df_all.copy()
    low_attend = int((df['attendance'] < 65).sum())
    poor_marks  = int((df['internal_marks'] < 50).sum())
    backlogs_3  = int((df['backlogs'] >= 3).sum())
    low_income  = int((df['family_income'] == 'low').sum())
    high_stress = int((df['stress'] >= 7).sum())
    far_travel  = int((df['travel_distance'] > 30).sum())
    no_internet = int((df['internet'] == 'no').sum())
    return jsonify({'interventions':[
        {'type':'attendance','color':'#E24B4A','tag':'Attendance Alert','title':'Parent Meeting',
         'desc':'Critical attendance below 65%. Initiate parent/guardian contact and co-create an attendance improvement plan with weekly check-ins.','count': low_attend},
        {'type':'academic','color':'#378ADD','tag':'Academic Support','title':'Remedial Tutoring',
         'desc':'Internal marks below 50%. Assign to peer mentoring or faculty-led remedial sessions for identified weak subjects.','count': poor_marks},
        {'type':'academic','color':'#7F77DD','tag':'Backlog Alert','title':'Backlog Clearance Plan',
         'desc':'3+ active backlogs. Structured re-exam prep with dedicated subject mentors and HOD notification.','count': backlogs_3},
        {'type':'financial','color':'#EF9F27','tag':'Financial Aid','title':'Scholarship Referral',
         'desc':'Family income below ₹2L/yr. Connect to NSP, state merit scholarships, fee waivers, and welfare fund emergency grants.','count': low_income},
        {'type':'social','color':'#D4537E','tag':'Mental Health','title':'Counselor Session',
         'desc':'Stress level 7+/10. Schedule confidential counseling session and share mental wellness resources.','count': high_stress},
        {'type':'attendance','color':'#854F0B','tag':'Travel Issue','title':'Transport Assistance',
         'desc':'Travel distance >30km. Evaluate hostel accommodation, bus route, or transport allowance eligibility.','count': far_travel},
        {'type':'digital','color':'#533AB7','tag':'Digital Access','title':'Internet Support',
         'desc':'No internet access. Connect to extended computer lab hours and subsidized data schemes.','count': no_internet},
    ]})
@app.route('/api/intervention_students/<intervention>')
def intervention_students(intervention):

    df = df_all.copy()

    # Fix NaN values
    df = df.fillna("")

    if intervention == "attendance":
        students = df[df['attendance'] < 65]

    elif intervention == "academic":
        students = df[df['internal_marks'] < 50]

    elif intervention == "backlog":
        students = df[df['backlogs'] >= 3]

    elif intervention == "financial":
        students = df[df['family_income'] == 'low']

    elif intervention == "mental":
        students = df[df['stress'] >= 7]

    elif intervention == "travel":
        students = df[df['travel_distance'] > 30]

    elif intervention == "digital":
        students = df[df['internet'] == 'no']

    else:
        students = pd.DataFrame()

    result = students[['student_name', 'student_id', 'dept']].fillna("").to_dict('records')

    return jsonify(result)
if __name__ == '__main__':
    app.run(debug=True, port=5050)
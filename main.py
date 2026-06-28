from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Loading the Data
df = pd.read_csv('dataset.csv')

#  Map User Type to Numbers
u_map = {'Student': 1, 'Career Changer': 2, 'Job Seeker': 3}
df['User_Type_Num'] = df['User_Type'].map(u_map)

#  Define Features (No Data Score)
features_list = [
    'User_Type_Num', 'Years_Exp', 'Coding_Score', 
    'Design_Score', 'Marketing_Score', 'Communication_Score', 
    'Leadership_Score', 'Problem_Solving_Score'
]

X = df[features_list]
y = df['Target_Career']

# 4. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 5. Skill Requirements for Gap Analysis
skill_requirements = {
    'Software Engineer': {'Coding_Score': 9, 'Problem_Solving_Score': 9},
    'Project Manager': {'Leadership_Score': 8, 'Communication_Score': 8},
    'UX Designer': {'Design_Score': 8, 'Communication_Score': 7},
    'Digital Marketer': {'Marketing_Score': 8, 'Communication_Score': 7}
}

@app.route('/')
def index():
    return render_template('inputpage.html')

@app.route('/profile')
def profile():
    return render_template('profileform.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    u_type = request.form.get('user_type')
    
    user_stats = {
        'Coding_Score': int(request.form.get('coding', 5)),
        'Design_Score': int(request.form.get('design', 5)),
        'Marketing_Score': int(request.form.get('marketing', 5)),
        'Communication_Score': int(request.form.get('comm', 5)),
        'Leadership_Score': int(request.form.get('lead', 5)),
        'Problem_Solving_Score': int(request.form.get('prob', 5))
    }
    
    input_data = [
        u_map.get(u_type, 1),
        int(request.form.get('exp', 0)),
        user_stats['Coding_Score'],
        user_stats['Design_Score'],
        user_stats['Marketing_Score'],
        user_stats['Communication_Score'],
        user_stats['Leadership_Score'],
        user_stats['Problem_Solving_Score']
    ]

    probs = model.predict_proba([input_data])[0]
    classes = model.classes_
    top_indices = np.argsort(probs)[-3:][::-1]
    
    results = [{'career': classes[i], 'score': round(probs[i]*100, 1)} for i in top_indices]
    
    top_career = results[0]['career']
    top_score = results[0]['score']
    gaps = []
    
    # Logic: If confidence is not 100% (or 98%), check for gaps
    if top_career in skill_requirements:
        for skill, min_val in skill_requirements[top_career].items():
            if user_stats[skill] < min_val:
                name = skill.replace('_Score', '').replace('_', ' ')
                gaps.append(f"Enhance your {name} skills to reach 100% readiness.")
    
    # Strict check: If score isn't perfect, ensure at least one gap is shown
    if top_score < 98.0 and not gaps:
        gaps.append("Profile is strong, but industry-specific certifications are recommended.")

    return render_template('output.html', results=results, top_career=top_career, gaps=gaps, top_score=top_score)

if __name__ == '__main__':
    app.run(debug=True)
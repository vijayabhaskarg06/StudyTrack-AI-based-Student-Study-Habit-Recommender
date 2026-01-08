import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('dataset/student_data.csv')

# Features and targets
X = data[['StudyHours', 'SleepHours', 'SocialMedia', 'Exercise']]
y_testmarks = data['TestMarks']
y_attention = data['AttentionLevel']

# Scale features for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Data preprocessed successfully.")

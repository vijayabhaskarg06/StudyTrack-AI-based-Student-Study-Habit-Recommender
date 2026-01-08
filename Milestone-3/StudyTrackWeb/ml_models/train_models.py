import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from StudyTrackWeb.settings import BASE_DIR

# Load dataset
data = pd.read_csv(BASE_DIR / 'dataset' / 'student_data.csv')

# Features and target for regression
X = data[['StudyHours', 'SleepHours', 'SocialMedia', 'Exercise']]
y = data['TestMarks']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Regression Models ---
# 1. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 2. Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save models
joblib.dump(lr_model, 'D:/BHASKAR/milestone 3/StudyTrackWeb/ml_models/linear_regression_model.pkl')
joblib.dump(rf_model, 'D:/BHASKAR/milestone 3/StudyTrackWeb/ml_models/random_forest_model.pkl')
print("Regression models trained and saved.")

# --- Clustering Model ---
# Scale features for clustering (exclude target)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[['StudyHours', 'SleepHours', 'SocialMedia']])

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Save clustering model and scaler
joblib.dump(kmeans, 'D:/BHASKAR/milestone 3/StudyTrackWeb/ml_models/kmeans_model.pkl')
joblib.dump(scaler, 'D:/BHASKAR/milestone 3/StudyTrackWeb/ml_models/scaler.pkl')
print("KMeans clustering model trained and saved.")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# Required columns from CSV
required_columns = [
    "Name",
    "StudyHours",
    "SleepHours",
    "SocialMediaHours",
    "ExerciseHours",
    "PlayHours"
]

# CSV file path
file_path = r"E:\Infosys Springboard Internship\Task 1\bulk_test_data.csv"

# Read CSV
df = pd.read_csv(file_path, usecols=required_columns)

print("\n==== DATASET USED ====")
print(df.head())

# ==================================================
# 1️⃣ LINEAR REGRESSION – Predict Study Performance
# ==================================================

print("\n==== LINEAR REGRESSION (Prediction) ====")

# Feature and target
X = df[['StudyHours']]
y = df['SleepHours']   # Example target (you can change if needed)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
test_hours = 7
pred = model.predict([[test_hours]])

print(f"Predicted Sleep Hours for {test_hours} Study Hours: {pred[0]:.2f}")

# ==================================================
# 2️⃣ K-MEANS CLUSTERING – Student Behavior Analysis
# ==================================================

print("\n==== K-MEANS CLUSTERING (Unsupervised) ====")

# Features for clustering
cluster_features = df[
    ['StudyHours', 'SleepHours', 'SocialMediaHours', 'ExerciseHours', 'PlayHours']
]

# KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(cluster_features)

print("\n=== Clustered Data ===")
print(df.head())

print("\nCluster Centers:")
print(kmeans.cluster_centers_)

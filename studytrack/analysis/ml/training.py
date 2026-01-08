import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score

def train_model(df):

    FEATURES = [
        "StudyHours",
        "SleepHours",
        "SocialMediaHours",
        "ExerciseHours",
        "PlayHours",
        "AttendancePercentage",
        "PreviousMarks",
        "AttentionLevel"
    ]

    X = df[FEATURES]
    y = df["TestMarks"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Regression model
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)

    y_pred = reg_model.predict(X_test)
    accuracy = r2_score(y_test, y_pred)

    # Clustering model
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)

    # Save models
    os.makedirs("trained_models", exist_ok=True)
    joblib.dump(reg_model, "trained_models/studytrack_regression.pkl")
    joblib.dump(kmeans, "trained_models/studytrack_kmeans.pkl")

    return round(accuracy * 100, 2)

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of students
n_students = 200

# Generate synthetic features
study_hours = np.random.normal(5, 2, n_students).clip(0, 12)
sleep_hours = np.random.normal(7, 1.5, n_students).clip(4, 10)
social_media = np.random.normal(2, 1, n_students).clip(0, 8)
exercise = np.random.normal(1, 0.5, n_students).clip(0, 3)

# Simulate TestMarks and AttentionLevel
# Assume positive correlation with study and sleep, negative with social media
test_marks = (study_hours * 8 + sleep_hours * 5 - social_media * 3 + np.random.normal(0,5,n_students)).clip(0,100)
attention_level = (study_hours * 5 + sleep_hours * 6 - social_media * 4 + exercise * 2 + np.random.normal(0,5,n_students)).clip(0,100)

# Create DataFrame
data = pd.DataFrame({
    'StudyHours': study_hours,
    'SleepHours': sleep_hours,
    'SocialMedia': social_media,
    'Exercise': exercise,
    'TestMarks': test_marks,
    'AttentionLevel': attention_level
})

# Save to CSV
data.to_csv('dataset/student_data.csv', index=False)
print("Synthetic dataset created: dataset/student_data.csv")

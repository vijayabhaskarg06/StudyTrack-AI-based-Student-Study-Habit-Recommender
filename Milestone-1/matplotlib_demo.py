import pandas as pd
import matplotlib.pyplot as plt

# Load Google Sheet Data

sheet_id = "1qJPi0ukmwwT3qWEvI5or8jcsp1FbXJfe"
gid = "1624515444"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

df = pd.read_csv(url)
print("Data Loaded Successfully!")
print(df.head())

# 1️⃣ Line Plot — Study Hours Trend
plt.figure(figsize=(6,4))
plt.plot(df["Study_Hours_Per_Day"], marker='o')
plt.title("Study Hours Per Day")
plt.xlabel("Student Index")
plt.ylabel("Study Hours")
plt.grid(True)
plt.show()

# 2️⃣ Bar Plot — Assignments Completed

plt.figure(figsize=(6,4))
plt.bar(df["Student_ID"], df["Assignments_Completed"])
plt.title("Assignments Completed by Students")
plt.xticks(rotation=45)
plt.ylabel("Assignments Completed")
plt.show()

# 3️⃣ Scatter Plot — Study Hours vs Test Score

plt.figure(figsize=(6,4))
plt.scatter(df["Study_Hours_Per_Day"], df["Test_Score"])
plt.title("Study Hours vs Test Score")
plt.xlabel("Study Hours")
plt.ylabel("Test Score")
plt.grid(True)
plt.show()

# 4️⃣ Histogram — Sleep Hours Distribution

plt.figure(figsize=(6,4))
plt.hist(df["Sleep_Hours"], bins=8)
plt.title("Distribution of Sleep Hours")
plt.xlabel("Sleep Hours")
plt.ylabel("Number of Students")
plt.show()

# 5️⃣ Pie Chart — Attendance Category Distribution

labels = ["Low (<70)", "Medium (70-90)", "High (>90)"]
sizes = [
    (df["Attendance_Percentage"] < 70).sum(),
    ((df["Attendance_Percentage"] >= 70) & (df["Attendance_Percentage"] <= 90)).sum(),
    (df["Attendance_Percentage"] > 90).sum()
]

plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, autopct="%1.1f%%")
plt.title("Attendance Percentage Categories")
plt.show()

print("All Matplotlib Plots Generated Successfully!")

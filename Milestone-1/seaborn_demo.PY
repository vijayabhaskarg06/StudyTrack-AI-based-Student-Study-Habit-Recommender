import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Google Sheet
sheet_id = "1qJPi0ukmwwT3qWEvI5or8jcsp1FbXJfe"
gid = "1624515444"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
df = pd.read_csv(url)

print("Columns in your sheet:")
print(df.columns)

# 1️⃣ Scatter Plot — Study Hours vs Test Score

plt.figure(figsize=(6,4))
sns.scatterplot(x="Study_Hours_Per_Day", y="Test_Score", data=df)
plt.title("Study Hours vs Test Score")
plt.show()

# 2️⃣ Histogram — Distribution of Sleep Hours

plt.figure(figsize=(6,4))
sns.histplot(df["Sleep_Hours"], kde=True)
plt.title("Distribution of Sleep Hours")
plt.show()

# 3️⃣ Box Plot — Test Score Spread

plt.figure(figsize=(6,4))
sns.boxplot(x=df["Test_Score"])
plt.title("Test Score Box Plot")
plt.show()

# 4️⃣ Bar Plot — Assignments Completed vs Test Score

plt.figure(figsize=(6,4))
sns.barplot(x="Assignments_Completed", y="Test_Score", data=df)
plt.title("Assignments Completed vs Test Score")
plt.show()

# 5️⃣ Heatmap — Correlation Between All Numeric Columns

plt.figure(figsize=(8,6))
corr = df.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


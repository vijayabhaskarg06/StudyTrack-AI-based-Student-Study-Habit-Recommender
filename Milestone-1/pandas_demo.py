import pandas as pd
import numpy as np

data = {
    "Name": ["Asha", "Rahul", "Vijay", "Asha"],
    "Department": ["IT", "HR", "IT", "Finance"],
    "Salary": [50000, 62000, 58000, None]
}

df = pd.DataFrame(data)

print("\nOriginal DataFrame:")
print(df)

# Fill missing values
print("\nFillna (Salary):")
print(df.fillna({"Salary": df["Salary"].mean()}))

# Drop missing values
print("\nDrop Rows with Null:")
print(df.dropna())

# Filtering
print("\nFilter IT Department:")
print(df[df["Department"] == "IT"])

# Groupby
print("\nGroupby Department - Mean Salary:")
print(df.groupby("Department")["Salary"].mean())

# Concat
df2 = pd.DataFrame({
    "Name": ["Kiran"],
    "Department": ["IT"],
    "Salary": [54000]
})

print("\nConcat DataFrames:")
print(pd.concat([df, df2], ignore_index=True))

# Merge
extra = pd.DataFrame({
    "Department": ["IT", "HR", "Finance"],
    "Location": ["BLR", "HYD", "PUNE"]
})

print("\nMerge on Department:")
print(df.merge(extra, on="Department"))

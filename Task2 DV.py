# Import libraries
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Display basic info
print("First 5 rows:")
print(df.head())

print("\nSummary:")
print(df.describe(include="all"))

# Clean some columns (drop rows with nulls in key columns for visualization)
df_clean = df.dropna(subset=["age", "fare", "embarked", "sex", "class", "survived"])

# -------------------------------
# 1. Pie Chart: Survival Rate
# -------------------------------
survival_counts = df_clean['survived'].value_counts()
labels = ['Did Not Survive', 'Survived']
colors = ['lightcoral', 'lightgreen']

plt.figure(figsize=(6,6))
plt.pie(survival_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Survival Distribution on the Titanic')
plt.tight_layout()
plt.show()

# -------------------------------
# 2. Bar Plot: Survival by Sex
# -------------------------------
plt.figure(figsize=(8, 5))
sns.countplot(data=df_clean, x='sex', hue='survived', palette='Set2')
plt.title('Survival Count by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.tight_layout()
plt.show()

# -------------------------------
# 3. Box Plot: Age Distribution by Class
# -------------------------------
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_clean, x='class', y='age', palette='Pastel1')
plt.title('Age Distribution by Passenger Class')
plt.xlabel('Class')
plt.ylabel('Age')
plt.tight_layout()
plt.show()

# -------------------------------
# 4. Violin Plot: Age vs Survival by Sex
# -------------------------------
plt.figure(figsize=(10, 6))
sns.violinplot(data=df_clean, x='sex', y='age', hue='survived', split=True, palette='coolwarm')
plt.title('Age vs Survival by Gender')
plt.tight_layout()
plt.show()

# -------------------------------
# 5. Heatmap: Correlation Matrix
# -------------------------------
plt.figure(figsize=(8, 6))
numeric_df = df_clean[["survived", "pclass", "age", "sibsp", "parch", "fare"]]
sns.heatmap(numeric_df.corr(), annot=True, cmap="YlGnBu", linewidths=0.5)
plt.title('Correlation Matrix of Titanic Features')
plt.tight_layout()
plt.show()

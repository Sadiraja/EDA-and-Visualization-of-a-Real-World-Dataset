import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set plot style for better visualization
sns.set(style="whitegrid")

# 1. Load the Dataset
# Using the Titanic dataset from a publicly available source
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Display basic information about the dataset
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())

# 2. Data Cleaning
# 2.1 Handle Missing Values
# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Impute 'Age' with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Impute 'Embarked' with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' due to high missing values (not critical for this analysis)
df.drop('Cabin', axis=1, inplace=True)

# 2.2 Remove Duplicates
df.drop_duplicates(inplace=True)
print("\nShape after removing duplicates:", df.shape)

# 2.3 Identify and Manage Outliers
# Using IQR for 'Fare' to detect outliers
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['Fare'] < lower_bound) | (df['Fare'] > upper_bound)]['Fare']
print("\nNumber of outliers in Fare:", len(outliers))

# Cap outliers in 'Fare'
df['Fare'] = df['Fare'].clip(lower=lower_bound, upper=upper_bound)

# 3. Visualizations
# Save all plots to a single figure
plt.figure(figsize=(15, 10))

# 3.1 Bar Charts for Categorical Variables
plt.subplot(2, 2, 1)
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')

plt.subplot(2, 2, 2)
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival by Passenger Class')

# 3.2 Histograms for Numeric Distributions
plt.subplot(2, 2, 3)
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution')

plt.subplot(2, 2, 4)
sns.histplot(df['Fare'], bins=20, kde=True)
plt.title('Fare Distribution')

plt.tight_layout()
plt.savefig('titanic_visualizations.png')
plt.close()

# 3.3 Correlation Heatmap for Numeric Features
plt.figure(figsize=(8, 6))
numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'Survived', 'Pclass']
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.savefig('titanic_correlation_heatmap.png')
plt.close()

# 4. Summarize Insights
insights = """
# Titanic Dataset EDA Insights

1. **Dataset Overview**:
   - The dataset contains {} rows and {} columns after cleaning.
   - Key features include Survived (target), Pclass, Sex, Age, Fare, and Embarked.

2. **Missing Values**:
   - Age: ~19% missing, imputed with median ({}).
   - Embarked: Few missing, imputed with mode ('{}').
   - Cabin: Dropped due to ~77% missing values.

3. **Outliers**:
   - Fare had {} outliers, capped using IQR method (lower: {}, upper: {}).

4. **Visualizations**:
   - **Survival Count**: {}% of passengers survived.
   - **Survival by Class**: Higher survival rates in 1st class vs. 3rd class.
   - **Age Distribution**: Most passengers were aged 20-40, with a median of {}.
   - **Fare Distribution**: Right-skewed, most fares below {} after capping.
   - **Correlation Heatmap**: Strong negative correlation between Pclass and Survived ({:.2f}), indicating lower classes had lower survival rates.

5. **Key Observations**:
   - Passenger class and fare are strongly linked to survival, with wealthier passengers (1st class, higher fares) more likely to survive.
   - Age shows a slight negative correlation with survival, suggesting younger passengers had a marginally higher survival rate.
   - SibSp and Parch (family size) have weak correlations with survival, indicating family size had limited impact.
""".format(
    df.shape[0], df.shape[1],
    df['Age'].median(), df['Embarked'].mode()[0],
    len(outliers), round(lower_bound, 2), round(upper_bound, 2),
    round(df['Survived'].mean() * 100, 2),
    df['Age'].median(), df['Fare'].median(),
    corr_matrix.loc['Pclass', 'Survived']
)

# Save insights to a markdown file
with open('titanic_eda_insights.md', 'w') as f:
    f.write(insights)

print("\nEDA completed. Visualizations saved as 'titanic_visualizations.png' and 'titanic_correlation_heatmap.png'.")
print("Insights saved to 'titanic_eda_insights.md'.")

# Titanic Dataset EDA Insights

1. **Dataset Overview**:
   - The dataset contains 891 rows and 11 columns after cleaning.
   - Key features include Survived (target), Pclass, Sex, Age, Fare, and Embarked.

2. **Missing Values**:
   - Age: ~19% missing, imputed with median (28.0).
   - Embarked: Few missing, imputed with mode ('S').
   - Cabin: Dropped due to ~77% missing values.

3. **Outliers**:
   - Fare had 116 outliers, capped using IQR method (lower: -26.72, upper: 65.63).

4. **Visualizations**:
   - **Survival Count**: 38.38% of passengers survived.
   - **Survival by Class**: Higher survival rates in 1st class vs. 3rd class.
   - **Age Distribution**: Most passengers were aged 20-40, with a median of 28.0.
   - **Fare Distribution**: Right-skewed, most fares below 14.4542 after capping.
   - **Correlation Heatmap**: Strong negative correlation between Pclass and Survived (-0.34), indicating lower classes had lower survival rates.

5. **Key Observations**:
   - Passenger class and fare are strongly linked to survival, with wealthier passengers (1st class, higher fares) more likely to survive.
   - Age shows a slight negative correlation with survival, suggesting younger passengers had a marginally higher survival rate.
   - SibSp and Parch (family size) have weak correlations with survival, indicating family size had limited impact.

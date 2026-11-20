import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


data = {
    'experience': [1, 2, 3, None, 5, 6, 7, 8, 9, 10, 3, 5],
    'education': ['BS', 'BS', 'MS', 'BS', 'MS', 'PhD', 'MS', 'PhD', 'BS', 'PhD', 'BS', 'MS'],
    'salary_k': [45, 50, 70, 58, 85, 120, 95, 140, 90, 160, 55, 88]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)


df['experience'] = df['experience'].fillna(df['experience'].median())

print("\nDataset after handling missing values:")
print(df)

df = pd.get_dummies(df, columns=['education'], dtype=int)

print("\nDataset after One Hot Encoding:")
print(df)


X = df.drop('salary_k', axis=1)
y = df['salary_k']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMAE:", mae)
print("R2 Score:", r2)


new_data = pd.DataFrame({
    'experience': [4],
    'education_BS': [0],
    'education_MS': [1],
    'education_PhD': [0]
})

new_data = scaler.transform(new_data)

salary = model.predict(new_data)

print("\nPredicted Salary:", salary[0], "K")
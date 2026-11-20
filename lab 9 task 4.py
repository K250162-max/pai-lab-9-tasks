import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = load_breast_cancer()

X = data.data
y = data.target

print("Number of Samples:", X.shape[0])
print("Number of Features:", X.shape[1])

print("\nClass Counts:")
print("Class 0:", np.sum(y == 0))
print("Class 1:", np.sum(y == 1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LogisticRegression(max_iter=10000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", accuracy)

probabilities = model.predict_proba(X_test)[:, 1]


thresholds = [0.3, 0.5, 0.7]

for threshold in thresholds:

    y_threshold = (probabilities >= threshold).astype(int)

    count = np.sum(y_threshold == 1)

    accuracy = accuracy_score(y_test, y_threshold)

    print("\nThreshold:", threshold)
    print("Predicted Class 1:", count)
    print("Accuracy:", accuracy)

def sigmoid(z):

    return 1 / (1 + np.exp(-z))

decision_values = model.decision_function(X_test)


sigmoid_values = sigmoid(decision_values)

actual_probabilities = model.predict_proba(X_test)[:, 1]


print("\nFirst 5 Sigmoid Values:")
print(sigmoid_values[:5])

print("\nFirst 5 predict_proba Values:")
print(actual_probabilities[:5])


print("\nAre Both Equal?")

print(np.allclose(sigmoid_values, actual_probabilities))
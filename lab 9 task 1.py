import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

hours = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0]
marks = [32, 40, 41, 50, 54, 60, 62, 70, 78, 88]

X = np.array(hours).reshape(-1, 1)
y = np.array(marks)

# Create and train the model
model = LinearRegression()
model.fit(X, y)


print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
predictions = model.predict([[5.5], [8.0]])

print("Marks after 5.5 hours:", predictions[0])
print("Marks after 8 hours:", predictions[1])

y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)


plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')

plt.xlabel("Hours Studied")
plt.ylabel("Marks Obtained")
plt.title("Linear Regression: Hours Studied vs Marks")
plt.legend()
plt.show()
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0])
y = np.array([32, 40, 41, 50, 54, 60, 62, 70, 78, 88])

def gradient_descent(x, y, lr, epochs):

    m = 0
    c = 0

    n = len(x)
    mse_history = []

    for i in range(1, epochs + 1):

        y_pred = m * x + c

        dm = (-2/n) * np.sum(x * (y - y_pred))
        dc = (-2/n) * np.sum(y - y_pred)

        m = m - lr * dm
        c = c - lr * dc

        if i % 100 == 0:

            y_pred = m * x + c

            mse = np.mean((y - y_pred) ** 2)

            mse_history.append(mse)

            print("Epoch:", i, "MSE:", mse)

    return m, c, mse_history

learning_rates = [0.001, 0.01, 0.05]

epochs = 30000

results = {}

for lr in learning_rates:

    print("\nLearning Rate:", lr)

    m, c, mse_history = gradient_descent(x, y, lr, epochs)

    results[lr] = (m, c, mse_history)

    print("Final Slope:", m)
    print("Final Intercept:", c)

for lr in learning_rates:

    mse_history = results[lr][2]

    epoch_values = range(100, epochs + 1, 100)

    plt.plot(epoch_values, mse_history, label=f"LR = {lr}")

plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("Gradient Descent Loss Curve")
plt.legend()
plt.show()
# linear regression --> finding a function that minimises error (difference bw prediction and real value)
# y = m . x + b (minimise)
# E = (1 / n) * sum [i = 0 to n] (yi - y)^2 --> mean squared error
# finding partial derivative wrt m and b --> direction of steepest ascent wrt m and b  --> go the opposite direction
# der ( E / m) = (1/n) . sum [i = 0 to n] 2 . (yi - (m.xi + b)) . (-xi) = (-2/n) . sum [i = 0 to n] xi (yi - (m.xi + b))
# der (E / b) = (-2 / n) . sum [i = 0 to n] (yi - (m.xi + b))
# m = m - L . (der(E/m))
# b = b - L . (der(E/b))
# L --> learning rate ( size of steps) ~ 0.001 


# dataset contains x_max = 100, x_min = 0

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('linear_regression_dataset.csv')

# print(data)

# plt.scatter(data.x, data.y)
# plt.show()

def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].x
        y = points.iloc[i].y
        total_error += (y - (m * x + b)) ** 2
    total_error /= float(len(points))
    return total_error

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)
    for i in range(n):
        x = points.iloc[i].x
        y = points.iloc[i].y
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b

def train(points, learning_rate, epochs):
    m = 0
    b = 0

    losses = []
    m_values = []
    b_values = []

    for epoch in range(epochs):
        loss = loss_function(m , b, points)
        losses.append(loss)
        m_values.append(m)
        b_values.append(b)

        m, b = gradient_descent(m, b, points, learning_rate)
    final_loss = loss_function(m, b, points)
    return m, b, final_loss, losses, m_values, b_values

# problems faced: initially the learning rate L was too large, tried at 0.001, 0.0005, 0.0003.
# what converged properly: 0.0001, 0.00005, 0.00001

# comparing different successful learning rates
learning_rates = [0.0001, 0.00005, 0.00001]
epochs = 300

results = {}

for L in learning_rates:
    m, b, final_loss, losses, m_values, b_values = train(data, L, epochs)
    results[L] = {"m": m, "b": b, "final_loss": final_loss, "losses": losses}
    print(
        f"L = {L} | "
        f"final loss = {final_loss:.6f} | "
        f"m = {m:.6f} | "
        f"b = {b:.6f}"
    )

# plotting loss curves

plt.figure()

for L in learning_rates:
    plt.plot(
        results[L]["losses"],
        label=f"L = {L}"
    )

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Learning Rate Comparison")
plt.legend()
plt.grid()

plt.savefig("learning_rate_comparison.png", dpi=300)

plt.show()

# plotting fitted lines

plt.figure()

plt.scatter(
    data.x,
    data.y,
    color="black",
    label="Data"
)

for L in learning_rates:

    m = results[L]["m"]
    b = results[L]["b"]

    plt.plot(
        data.x,
        m * data.x + b,
        label=f"L = {L}"
    )

plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression — Learning Rate Comparison")
plt.legend()
plt.grid()

plt.savefig("fitted_lines_comparison.png", dpi=300)

plt.show()

# all three rates converge to same solution(figure 2)
# however they have different loss curves --> different learning rates can converge to the same solution, but at different speeds 

# final parameters output:
# L = 0.0001 | final loss = 7.893242 | m = 0.998783 | b = 0.012953
# L = 5e-05 | final loss = 7.893309 | m = 0.998768 | b = 0.013958
# L = 1e-05 | final loss = 7.893364 | m = 0.998756 | b = 0.014767
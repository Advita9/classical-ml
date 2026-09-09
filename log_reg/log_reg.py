# logistic regression --> why is it regression when it is actually used to predict classes? : it outputs probabilities of how likely an output is going to belong to a binary classification
# input of m samples : Xi (vector containing n features: [X1i, X2i ... Xmi])
# multiply with a vector thetaT (vector of paramaters) --> linear combination
# thetaT . Xi = zi (logits / output of linear combination)
# there is also an intercept / bias added
# sigmoid function --> returns a value between 0 and 1 (non inclusive) --> 0.5 in the middle
# sigmoid(z) = 1 / (1+ e^(-z))
# hi(theta) . Xi = sigmoid(zi) --> estimated probability of the instance being positive (where h is sigmoid of z)
# we only want to maximise to 1 this if the instance is actually positive, else minimise to 0 --> basis of the optimisation function for this
# likelihood of paramters being ootimal --> theta = product [i = 1 to m] [[h(theta) . Xi] ^ (yi)] . [1 - [h(theta) . Xi] ^ (1 - yi)]. where yi is ground truth for instance i
# eg. if sigmoid gives 0.999 --> 0.999^1 * (1-0.999)^0 = 0.999  (we get strong output for ground truth being 1)
# --> 0.999^0 * (1-0.999)^1 = 0.001 (we get close to 0 for ground truth being 0 and sigmoid predicting 0.999 for 1)
# limitations of current function :
# 1. need to negate function --> so we can perform gradient descent for minimising
# 2. numerically unstable --> need to take logarithm (natural)
# 3. need to scale it --> divide by m
# log likelihood of theta --> l(theta) = sum [i = 1 to m] yi . ln(h(theta) . Xi) + (1-yi) . ln(1 - (h(theta) . Xi))
# cross entropy loss function (which we need to minimise) --> j(theta) = (-1/m) [l(theta) = sum [i = 1 to m] yi . ln(h(theta) . Xi) + (1-yi) . ln(1 - (h(theta) . Xi))]

# dl / d theta j = (-1/m) dl / dh . dh / dz . dz / d theta (partial derivative calculation)
# dl / d theta j = (-1/m) [(y/h) - ((1-y)/(1-h))] . [h (1-h))] . [Xji] = (-1/m) [(y - h) . (Xji)]
# gradient of J(theta) - (-1/m) XT (h(thetaX) - y) = (1/m) XT (y-h(thetaX)) --> gives us the direction
# theta --> theta - L . gradient of J theta where L = learning rate 

import numpy as np

def sigmoid(z):
    # handling overflow when z is a large negative number
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def calculate_gradient(theta, X, y):
    m = y.size # number of instances
    return (X.T @ (sigmoid(X @ theta) - y)) / m

def gradient_descent(X, y, alpha=0.1, num_iter=100, tol=1e-7):

    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.zeros(X_b.shape[1])

    losses = []

    for i in range(num_iter):

        # predictions
        probabilities = sigmoid(X_b @ theta)

        # calculate gradient
        grad = calculate_gradient(theta, X_b, y)

        # update parameters
        theta -= alpha * grad

        # calculate and store loss
        loss = binary_cross_entropy(theta, X_b, y)
        losses.append(loss)

        # convergence check
        if np.linalg.norm(grad) < tol:
            break

    return theta, losses



def predict_proba(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return sigmoid(X_b @ theta)

def predict(X, theta, threshold = 0.5):
    return (predict_proba(X, theta) >= threshold).astype(int)

def binary_cross_entropy(theta, X, y):
    probabilities = sigmoid(X @ theta)
    epsilon = 1e-15
    probabilities = np.clip(probabilities, epsilon, 1-epsilon)
    loss = -np.mean(y * np.log(probabilities) + (1-y) * np.log(1-probabilities))
    return loss

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



learning_rates = [0.001, 0.1, 0.5]

results = {}

for alpha in learning_rates:

    theta, losses = gradient_descent(
        X_train_scaled,
        y_train,
        alpha=alpha,
        num_iter=100
    )

    y_pred_train = predict(X_train_scaled, theta)
    y_pred_test = predict(X_test_scaled, theta)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    results[alpha] = {
        "theta": theta,
        "losses": losses,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc
    }

    print(
        f"alpha={alpha} | "
        f"train accuracy={train_acc:.4f} | "
        f"test accuracy={test_acc:.4f}"
    )


alphas = [str(alpha) for alpha in learning_rates]

train_accuracies = [
    results[alpha]["train_accuracy"]
    for alpha in learning_rates
]

test_accuracies = [
    results[alpha]["test_accuracy"]
    for alpha in learning_rates
]

x = np.arange(len(learning_rates))
width = 0.35

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))

plt.bar(
    x - width / 2,
    train_accuracies,
    width,
    label="Train"
)

plt.bar(
    x + width / 2,
    test_accuracies,
    width,
    label="Test"
)

plt.xticks(x, alphas)
plt.xlabel("Learning Rate (α)")
plt.ylabel("Accuracy")
plt.title("Logistic Regression — Accuracy by Learning Rate")
plt.ylim(0.9, 1.0)
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.savefig(
    "logistic_learning_rate_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


plt.figure(figsize=(10, 6))

for alpha in learning_rates:
    plt.plot(
        results[alpha]["losses"],
        label=f"α = {alpha}"
    )

plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy")
plt.title("Logistic Regression — Learning Rate Comparison")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig(
    "logistic_learning_rate_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# results:
# alpha=0.001 | train accuracy=0.9385 | test accuracy=0.9211
# alpha=0.1 | train accuracy=0.9802 | test accuracy=0.9649
# alpha=0.5 | train accuracy=0.9890 | test accuracy=0.9649
# (venv) advitas@Advitas-MacBook-Pro log_reg %           
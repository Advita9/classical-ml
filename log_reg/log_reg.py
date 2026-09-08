# logistic regression --> why is it regression when it is actually used to predict classes? : it outputs probabilities of how likely an output is going to belong to a binary classification
# input of m samples : Xi (vector containing n features: [X1i, X2i ... Xmi])
# multiply with a vector thetaT (vector of paramaters) --> linear combination
# thetaT . Xi = zi (logits / output of linear combination)
# there is also an intercept / bias added
# sigmoid function --> returns a value between 0 and 1 (non inclusive) --> 0.5 in the middle
# sigmoid(z) = 1 / (1- e^(-z))
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
    return 1.0 / (1.0 + np.exp(-z))

def calculate_gradient(theta, X, y):
    m = y.size # number of instances
    return (X.T @ (sigmoid(X @ theta) - y)) / m

def gradient_descent(X, y, alpha = 0.1, num_iter= 100, tol=1e-7):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.zeros(X_b.shape[1])

    for i in range(num_iter):
        grad = calculate_gradient(theta, X_b, y)
        theta -= alpha * grad

        if np.linalg.norm(grad) < tol:
            break
    return theta

def predict_proba(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return sigmoid(X_b @ theta)

def predict(X, theta, threshold = 0.5):
    return (predict_proba(X, theta) >= threshold).astype(int)

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

theta_hat = gradient_descent(X_train_scaled, y_train, alpha = 0.1)

y_pred_train = predict(X_train_scaled, theta_hat)
y_pred_test = predict(X_test_scaled, theta_hat)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print(train_acc)
print(test_acc)

# results:
# 0.9758241758241758
# 1.0
                             
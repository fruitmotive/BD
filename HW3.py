import numpy as np
import matplotlib.pyplot as plt

def f(x, i):
    return x ** i 

def F(x, output='full'):
    a = [1, 1, -1, 0.2]

    if output == 'summarized':
        return sum([a[i] * f(x, i) for i in range(4)])

    return np.array([[f(x, i) for i in range(4)]])


class CanonLinearRegression:

    def __init__(self, sigma=0, m=0):
        self.n = 0
        self.V = 0
        self.upsilon = np.zeros((m, 1))
        self.T = np.zeros((m, m))
        self.sigma = sigma
        self.m = m

    def addition(self, x, y):
        self.n += 1
        self.V += y ** 2

        aux = F(x)
        auxT = np.transpose(aux)

        self.upsilon += auxT * y
        self.T += auxT @ aux

    def predict(self, x):
        T_inv = np.linalg.inv(self.T)
        F_ = F(x)
        F_T = np.transpose(F_)
        ups = self.upsilon
        ups_T = np.transpose(self.upsilon)

        est_f = F_ @ T_inv @ ups
        d_est_f = self.sigma ** 2 * F_ @ T_inv @ F_T
        est_d_est_f = (self.V - ups_T @ T_inv @ ups) / (self.n - self.m) @ F_ @ T_inv @ F_T

        return est_f[0][0], d_est_f[0][0], est_d_est_f[0][0]


model = CanonLinearRegression(1, 4)
X_, Y = [], []

number_of_dots = 20

for _ in range(number_of_dots):
    x = 4 * np.random.rand()
    y = F(x, 'summarized') + np.random.normal(scale=1)

    X_.append(x)
    Y.append(y)

    model.addition(x, y)

predictions, real = [], []
with_sigma, without_sigma = [], []

X = np.linspace(0, 4, num=1000)
for x in X:
    real.append(F(x, 'summarized'))
    prediction = model.predict(x)
    predictions.append(prediction[0])
    with_sigma.append(prediction[1])
    without_sigma.append(prediction[2])

plt.plot(X, predictions, color='black', label='Estimated Function')

plt.plot(X, np.array(predictions) - np.array(with_sigma), color='green', label='Error')
plt.plot(X, np.array(predictions) + np.array(with_sigma), color='green')

plt.plot(X, np.array(predictions) - np.array(without_sigma), color='green', ls='--', label='Estimated Error')
plt.plot(X, np.array(predictions) + np.array(without_sigma), color='green', ls='--')

plt.plot(X, real, color='red', label='True Function')

plt.scatter(X_, Y)

plt.title("Prediction with {0} training dots".format(number_of_dots))

plt.legend()

plt.show()

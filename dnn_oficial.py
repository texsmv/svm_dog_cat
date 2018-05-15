import numpy as np
from sklearn.externals import joblib
from funciones import shuffle


def tanh(x, deriv=False):
    if(deriv):
        return 1 / (pow(np.cosh(x), 2))
    return np.tanh(x)


def sigmoidea(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def linear(x, deriv=False):
    if deriv:
        shape = np.shape(x)
        temp = x.flatten()
        temp = np.array([1 for e in temp])
        temp = temp.reshape(shape)

    return x


class dnn:
    def __init__(self):

        self.layers = []
        self.activation = []
        self.weights = []
        self.bias = []
        self.alfa = 1
        self.umbral = 0.0001
        self.funcs = {"sigmoidea": sigmoidea, "linear": linear, "tanh": tanh}

    def add_layer(self, size, activacion="linear"):
        self.layers = self.layers + [np.array([np.empty([size, ])])]
        self.bias = self.bias + [np.array([np.random.rand(size)])]
        np.random.seed(1)

        self.weights = self.weights + [np.random.rand(len(self.layers[len(self.layers) - 2][0]), size)]
        self.activation = self.activation + [activacion]

    def add_input_layer(self, size):
        self.layers = self.layers + [np.array([np.empty([size, ])])]

    def imprimir(self):
        print(" ")
        print("..............................Estado.................................. ")
        print(" ")
        print("Layer 0: ")
        print(np.shape(self.layers[0]))
        print(self.layers[0])
        for i in range(1, len(self.layers)):
            print("Pesos: ")
            print(np.shape(self.weights[i - 1]))
            print(self.weights[i - 1])
            print(" ")

            print("Layer ", i, " :")
            print(np.shape(self.layers[i]))
            print(self.layers[i])

    def calcular_netas(self, input):
        self.layers[0][0] = input
        for i in range(1, len(self.layers)):
            self.layers[i] = np.dot(self.layers[i - 1], self.weights[i - 1]) + self.bias[i - 1]
            self.layers[i] = self.funcs[self.activation[i - 1]](self.layers[i])

    def forward(self, input):
        self.calcular_netas(np.array(input))

    def train(self, inputs, outputs, epocas, lr):
        self.alfa = lr
        pos = 0
        for i in range(0, epocas):
            self.forward(inputs[pos])
            self.calcular_deltas(outputs[pos])
            self.update_weights()
            error = self.error(outputs[pos])
            pos = (pos + 1) % len(inputs)
            print("error: ", error)

    def error(self, Sd):
        error = sum(((Sd - self.layers[-1][0]) ** 2) / 2)
        return error

    def calcular_deltas(self, Sd):
        Sd = np.array([np.array(Sd)])
        self.deltas = []
        delta_o = (Sd - self.layers[-1]) * self.funcs[self.activation[-1]](self.layers[-1], deriv=True)
        self.deltas = self.deltas + [delta_o]
        for k in range(len(self.layers) - 2, 0, -1):
            delta = np.dot(self.deltas[0], self.weights[k].T)
            delta = delta * self.funcs[self.activation[k - 1]](self.layers[k], deriv=True)
            self.deltas = [delta] + self.deltas

    def update_weights(self):
        for k in range(1, len(self.layers)):
            errores = self.layers[k - 1].T.dot(self.deltas[k - 1])
            self.weights[k - 1] += self.alfa * errores
            self.bias[k - 1] += self.alfa * self.deltas[k - 1]

    def test(self, inputs):
        for e in inputs:
            self.forward(e)
            print(self.layers[-1][0])


rnn = dnn()
rnn.add_input_layer(36)
rnn.add_layer(15, activacion="sigmoidea")
rnn.add_layer(1, activacion="sigmoidea")

X = joblib.load("X.pkl")
y = joblib.load("Y.pkl")

X, y = shuffle(X, y)
rnn.train(X, y, 1000000, 0.001)

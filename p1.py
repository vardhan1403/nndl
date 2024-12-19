import numpy as np
class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, x):
        linear_output = np.dot(x, self.weights) + self.bias
        return self.sigmoid(linear_output)

    def fit(self, x, y):
        num_samples, num_features = x.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for _ in range(self.epochs):
            for idx, sample in enumerate(x):
                linear_output = np.dot(sample, self.weights) + self.bias
                y_predicted = self.sigmoid(linear_output)
                error = y[idx] - y_predicted
                self.weights += self.learning_rate * error * y_predicted * (1 - y_predicted) * sample
                self.bias += self.learning_rate * error * y_predicted * (1 - y_predicted)

x_and = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
y_and = np.array([0, 0, 0, 1])

x_or = np.array([[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]])
y_or = np.array([0, 1, 1, 1])

perceptron_and = Perceptron(learning_rate=0.1, epochs=1000)
perceptron_and.fit(x_and, y_and)

print("AND gate predictions:")
for i in range(len(x_and)):
    prediction = perceptron_and.predict(x_and[i])
    print(f"Input: {x_and[i]} - Prediction: {1 if prediction >= 0.5 else 0}")

perceptron_or = Perceptron(learning_rate=0.1, epochs=1000)
perceptron_or.fit(x_or, y_or)

print("OR gate predictions:")
for i in range(len(x_or)):
    prediction = perceptron_or.predict(x_or[i])
    print(f"Input: {x_or[i]} - Prediction: {1 if prediction >= 0.5 else 0}")
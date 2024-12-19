import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=10000):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand(1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return sigmoid(weighted_sum)

    def train(self, training_inputs, labels):
        for epoch in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)

                error = label - prediction

                adjustments = error * sigmoid_derivative(prediction)

                self.weights += self.learning_rate * inputs * adjustments
                self.bias += self.learning_rate * adjustments

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

and_labels = np.array([0, 0, 0, 1])

or_labels = np.array([0, 1, 1, 1])

and_perceptron = Perceptron(input_size=2)

print("Training for AND gate...")
and_perceptron.train(inputs, and_labels)

print("\nAND Gate Results:")
for i in inputs:
    print(f"Input: {i}, Output: {and_perceptron.predict(i)}")

or_perceptron = Perceptron(input_size=2)

print("\nTraining for OR gate...")
or_perceptron.train(inputs, or_labels)

print("\nOR Gate Results:")
for i in inputs:
    print(f"Input: {i}, Output: {or_perceptron.predict(i)}")

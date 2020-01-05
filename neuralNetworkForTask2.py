from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        random.seed(1)
        self.synaptic_weights = 2 * random.random((6, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def sigmoidDerivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            output = self.decide(training_set_inputs)
            error = training_set_outputs - output

            adjustment = dot(training_set_inputs.T, error * self.sigmoidDerivative(output))

            self.synaptic_weights += adjustment

    def decide(self, inputs):
        return self.sigmoid(dot(inputs, self.synaptic_weights))
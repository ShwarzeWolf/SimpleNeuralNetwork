from numpy import exp, array, random, dot
import neuralNetworkForTask2

neural_network = neuralNetworkForTask2.NeuralNetwork()

training_set_inputs = array([[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 1, 0,  0, 0, 0], [0, 1, 1,  0, 0, 0], [1, 0, 0, 0, 0, 0], [1, 0, 1  0, 0, 0]])
training_set_outputs = array([[0, 0, 1, 0, 1, 1, 0, 0, 0]]).T

neural_network.train(training_set_inputs, training_set_outputs, 10000)

print("Considering new situation [1, 0, 0] -> ?: ")
print(neural_network.decide(array([0, 1, 0, 0, 0, 0])))


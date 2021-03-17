import numpy as np
np.random.seed(0)

inputs =[[1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]]

class layer_dense:
  def __init__(self, n_inputs, n_neurons):
    self.weights = 0.10*np.random.randn(n_inputs, n_neurons)
    self.biases = np.zeros((1, n_neurons))
  def forward(self, inputs):
    self.outputs = np.dot(inputs, self.weights)+self.biases

layer1 = layer_dense(4,5)
layer2 = layer_dense(5,2)

layer1.forward(inputs)
layer2.forward(layer1.outputs)

# print(layer1.outputs)
print(layer2.outputs)

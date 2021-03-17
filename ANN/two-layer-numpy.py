inputs =[[1, 2, 3, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.0]]
weights = [[0.2, 0.8, -0.5, 1.0],
          [0.5, -0.91, 0.26, -0.5],
          [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],
          [-0.5, 0.12, -0.33],
          [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

layer_outputs = []

# let's try with numpy
import numpy as np
# calculating matrix multiplication via np.dot(x,y)
layer1_outputs = np.dot(inputs,np.array(weights).T)+biases 
layer2_outputs = np.dot(layer1_outputs,np.array(weights2).T)+biases2
# print(outputs) # [4.8   1.21  2.385]
print(layer1_outputs)
print(layer2_outputs)



# following code is for calculating mannually
'''
for neuron_weight, neuron_bias in zip(weights, biases):
  neuron_output = 0
  for input_val, weight_val in zip(inputs, neuron_weight):
    neuron_output += input_val * weight_val
  neuron_output += neuron_bias
  layer_outputs.append(neuron_output)
print(layer_outputs) # [4.8   1.21  2.385]
'''
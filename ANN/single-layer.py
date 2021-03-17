inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
          [0.5, -0.91, 0.26, -0.5],
          [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]



# following code is for calculating mannually
layer_outputs = []
for neuron_weight, neuron_bias in zip(weights, biases):
  neuron_output = 0
  for input_val, weight_val in zip(inputs, neuron_weight):
    neuron_output += input_val * weight_val
  neuron_output += neuron_bias
  layer_outputs.append(neuron_output)
print(layer_outputs) # [4.8   1.21  2.385]
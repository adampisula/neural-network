from math import e as e_const
from random import random
from typing import Type

# HELPERS
def sigmoid(x):
  if isinstance(x, list):
    return [sigmoid(a) for a in x]
  
  return 1 / (1 + e_const**(-1*x))

def relu(x):
  return max([0, x])

def dot_product(a, b):
  if len(a) != len(b):
    raise ValueError('Lengths of vectors do not match.')

  return sum([a[i] * b[i] for i in range(len(a))])

def mse(x, y): # x IS OUTPUT, y IS DESIRED
  return sum([(x[i] - y[i])**2 for i in range(len(x))]) / len(x)

class Neuron:
  weights = []
  bias = 0
  activation_function = None

  def __init__(self, n_previous, weights, bias, activation_function=sigmoid):
    if len(weights) != n_previous:
      raise ValueError('Incorrect size of weights vector.')
      
    self.weights = weights
    self.bias = bias
    self.activation_function = activation_function

class ANN:
  layers = []
  input_layer_length = 0

  def __init__(self, structure, weights=None, biases=None):
    if weights == None:
      weights = []

      # FOR EVERY LAYER APART FROM THE FIRST
      for i in range(1, len(structure)):
        layer_weights = []

        # FOR EVERY NEURON IN THAT LAYER
        for j in range(structure[i]):
          # FOR EVERY NEURON IN THE PREVIOUS LAYER
          layer_weights.append([random() for x in range(structure[i - 1])])

        weights.append(layer_weights)

    if biases == None:
      biases = []

      # FOR EVERY LAYER APART FROM THE FIRST
      for i in range(1, len(structure)):
        # FOR EVERY NEURON IN THAT LAYER
        biases.append([0 for x in range(structure[i])])

    if isinstance(structure, (list, tuple)):
      self.input_layer_length = structure[0]

      # FOR EVERY LAYER APART FORM THE FIRST
      for i in range(1, len(structure)):
        layer = []

        # FOR EVERY NEURON IN THAT LAYER
        for j in range(structure[i]):
          layer.append(Neuron(structure[i - 1], weights[i - 1][j], biases[i - 1][j]))

        self.layers.append(layer)

    else:
      raise TypeError('Network structure not a list or a tuple.')

  def propagate(self, input):
    if isinstance(input, (list, tuple)):
      if len(input) != self.input_layer_length:
        raise ValueError('Input\'s length does not match the first layer\'s length.')

      layer_values = input

      # FOR EVERY LAYER APART FROM THE FIRST ONE (IT'S VIRTUAL SO TO SPEAK)
      for layer in self.layers:
        next_layer_values = []

        print(layer_values)

        for neuron in layer:
          next_layer_values.append(neuron.activation_function(dot_product(layer_values, neuron.weights) + neuron.bias))

        layer_values = next_layer_values

      return layer_values

    else:
      raise TypeError('Input should be a list or a tuple.')

w = [
  [
    [1, 2],
    [3, 4],
    [5, 6]
  ],
  [
    [1, 2, 3],
    [4, 5, 6]
  ]
]

b = [
  [0.5, 1.5, 2.5],
  [2, 4]
]

#a = ANN([2, 3, 2], w, b)

#a = ANN([2, 6, 2])

#print(a.propagate([0.1, 0.5]))

print(mse([1, 2, 3], [0, -1, 5]))
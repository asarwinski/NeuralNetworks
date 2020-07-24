from dense import Dense
from sigmoid import Sigmoid
from tanh import Tanh
from relu import Relu
import numpy as np

class MLP():
	def __init__(self, input_dim, sizes):
		self.layers = []
		sizes.insert(0, input_dim)
		
		for i in range(1, len(sizes)-1):
			layer = Dense(sizes[i], sizes[i-1], Relu())
			self.layers.append(layer)

		l = len(sizes)
		layer = Dense(sizes[l-1], sizes[l-2], Sigmoid())
		self.layers.append(layer)

	def get_cost(self, Y_hat, Y):
		m = Y_hat.shape[1]
		return -np.sum(Y*np.log(Y_hat) + (1-Y)*np.log(1-Y_hat)) / m

	def predict(self, X):
		output = X

		for layer in self.layers:
			output = layer.forward(output)

		return output

	def train(self, X, Y, epochs, learning_rate):
		costs = []

		for i in range(epochs):
			Y_hat = self.predict(X)
			cost = self.get_cost(Y_hat, Y)
			

			prev_deriv = -(Y/Y_hat) + (1-Y)/(1-Y_hat)

			for layer in reversed(self.layers):
				prev_deriv = layer.backward(prev_deriv)

			for layer in self.layers:
				layer.update_params(learning_rate)

			if i%10 == 0:
				print('Epoch {0}: {1}'.format(i, cost))
				costs.append(cost)

		return costs
		

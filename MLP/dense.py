import numpy as np
from layer import ILayer

class Dense(ILayer):
	def __init__(self, units, input_size, activation):
		self.W = np.random.randn(units, input_size) / np.sqrt(input_size)
		self.b = np.zeros([units, 1])
		self.activation = activation
		
	def forward(self, X):
		Z = np.dot(self.W, X) + self.b
		A = self.activation.get_output(Z)
		
		self.last_X = X
		self.last_Z = Z

		return A

	def backward(self, prev_deriv):
		m = self.last_X.shape[1]
		
		dZ = self.activation.get_derivative(self.last_Z)
		dZ = dZ * prev_deriv
		self.dW = np.dot(dZ, self.last_X.T) / m
		
		self.db = np.sum(dZ, axis=1, keepdims=True) / m	

		x = np.dot(self.W.T, dZ)

		# print('dW{0}: {1}'.format(self.index, self.dW))
		# print('db{0}: {1}'.format(self.index, self.db))
		# print('dA{0}: {1}'.format(self.index-1, x))
		return x

	def update_params(self, learning_rate):
		self.W = self.W - learning_rate*self.dW
		self.b = self.b - learning_rate*self.db

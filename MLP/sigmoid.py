import numpy as np
from activation import IActivation

class Sigmoid(IActivation):
	def get_output(self, x):
		return 1 / (1 + np.exp(-x))

	def get_derivative(self, x):
		activation = self.get_output(x)
		return activation * (1 - activation)
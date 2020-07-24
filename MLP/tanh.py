import numpy as np
from activation import IActivation

class Tanh(IActivation):
	def get_output(self, x):
		return np.tanh(x)

	def get_derivative(self, x):
		activation = self.get_output(x)
		return 1 - activation**2
from activation import IActivation
import numpy as np

class Relu(IActivation):
	def get_output(self, x):
		return np.maximum(0, x)

	def get_derivative(self, x):
		return (x > 0).astype(float)
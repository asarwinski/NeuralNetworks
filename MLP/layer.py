from abc import ABC, abstractmethod

class ILayer(ABC):
	@abstractmethod
	def forward(self, X):
		pass
	
	@abstractmethod
	def backward(self, prev_deriv):
		pass

	@abstractmethod
	def update_params(self, learning_rate):
		pass
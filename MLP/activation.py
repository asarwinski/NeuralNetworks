from abc import ABC, abstractmethod

class IActivation(ABC):
	@abstractmethod
	def get_output(self, x):
		pass
	
	@abstractmethod
	def get_derivative(self, x):
		pass
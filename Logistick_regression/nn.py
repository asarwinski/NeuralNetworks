import numpy as np

class LogisticRegression:
	def __init__(self, input_dim):
		self.W = np.zeros([input_dim, 1])
		self.b = 0

	def _sigm_(self, x):
		return 1 / (1 + np.exp(-x))

	def predict(self, X):
		z = np.dot(self.W.T, X) + self.b	
		prediction = self._sigm_(z)
		return prediction

	def cost(self, Y_hat, Y):
		# -(Y * log(Y_hat) + (1 - Y)(log(1-Y_hat)))	
		# TODO - change cost function or activation to simg
		cost = -1*(Y * np.log(Y_hat) + (1 - Y)*np.log(1-Y_hat))
		cost = cost.sum() / Y.shape[1]
		return cost

	def getGradients(self, X, Y_hat, Y):
		# nom = Y * (Y_hat + Y_hat**2 - Y_hat**3 - 1) + Y_hat - Y_hat**3
		# denom = Y_hat * (1 - Y_hat)
		dz = Y_hat - Y

		dw = np.dot(X, dz.T) / Y.shape[1]
		db = np.sum(dz) / Y.shape[1]

		return {'dw': dw, 'db': db}
		
	def updateWeights(self, grads, learning_rate):
		dw = grads['dw']
		db = grads['db']
		
		self.W = self.W - learning_rate * dw
		self.b = self.b - learning_rate * db

	def train(self, X, Y, epochs, learning_rate = 0.002, callbacks = None):
		for i in range(epochs):
			y_hat = self.predict(X)
			cost = self.cost(y_hat, Y)
			grads = self.getGradients(X, y_hat, Y)
			self.updateWeights(grads, learning_rate)
			for call in callbacks:
				call(i, cost)
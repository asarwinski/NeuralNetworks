from mlp import MLP
import numpy as np
import matplotlib.pyplot as plt
from utils import getDatasets
np.random.seed(1)

train_X, train_Y, test_X, test_Y, labels = getDatasets()

input_dim = 12288
hidden_sizes = [20, 7, 5, 1]

mlp = MLP(input_dim, hidden_sizes)
costs = mlp.train(train_X, train_Y, 2500, 0.002)

predictions = mlp.predict(train_X)

predictions = (predictions > 0.5).astype(float)

accuracy = (predictions == train_Y).astype(float)
accuracy = np.mean(accuracy)

print('Accuracy on training set: {0}'.format(accuracy))


predictions = mlp.predict(test_X)

predictions = (predictions > 0.5).astype(float)

accuracy = (predictions == test_Y).astype(float)
accuracy = np.mean(accuracy)

print('Accuracy on test set: {0}'.format(accuracy))

plt.plot(costs)
plt.show()
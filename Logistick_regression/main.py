from nn import LogisticRegression
import numpy as np
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from utils import getDatasets

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

history = []

def callback(epoch, cost):
	history.append(cost)
	print("{0}: {1}".format(epoch, cost))


train_X, train_Y, test_X, test_Y, labels = getDatasets()

network = LogisticRegression(train_X.shape[0])
network.train(train_X, train_Y, 2000, learning_rate=0.002, callbacks=[callback])

predictions = network.predict(train_X)

predictions = (predictions > 0.5).astype(float)

accuracy = (predictions == train_Y).astype(float)
accuracy = np.mean(accuracy)

print('Accuracy on training set: {0}'.format(accuracy))


predictions = network.predict(test_X)

predictions = (predictions > 0.5).astype(float)

accuracy = (predictions == test_Y).astype(float)
accuracy = np.mean(accuracy)

print('Accuracy on test set: {0}'.format(accuracy))

predictions = network.predict(train_X)
predictions = (predictions > 0.5).astype(int)

plt.figure(figsize=[10,10])
for i in range(25):
	plt.subplot(5,5, i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(train_X[:,i].reshape([64,64,3]))
	plt.xlabel(labels[predictions[0,i]])
plt.show()

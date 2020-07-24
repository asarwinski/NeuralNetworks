import tensorflow as tf
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


def processImg(path):
	img = tf.io.read_file(path)
	img = tf.io.decode_image(img)

	h = img.shape[0]
	w = img.shape[1]

	if h > w:
		img = tf.image.crop_to_bounding_box(img, (h - w) // 2, 0, w, w)
	else:
		img = tf.image.crop_to_bounding_box(img, 0, (w - h) // 2, h, h)

	img = tf.image.resize(img, (64, 64))
	img /= 255

	return img.numpy()

def getDatasets():
	paths_cats = glob('data/cats/*.jpg')[:100]
	paths_dogs = glob('data/dogs/*.jpg')[:100]


	train_X = []
	print('Processing images...')
	cats = (list(map(processImg, paths_cats)))
	cats = np.asarray(cats)

	dogs = (list(map(processImg, paths_dogs)))
	dogs = np.asarray(dogs)


	train_X = np.concatenate([cats, dogs])
	train_X = train_X.reshape([-1, 12288])

	train_Y = np.concatenate([np.zeros([len(cats),1]),np.ones([len(dogs),1])])

	c = list(zip(list(train_X),list(train_Y)))
	random.shuffle(c)
	train_X, train_Y = zip(*c)
	train_X = np.asarray(train_X)
	train_Y = np.asarray(train_Y)

	train_X = train_X.T
	train_Y = train_Y.T

	test_size = int(train_X.shape[1] * 0.1)

	test_X = train_X[:,0:test_size]
	train_X = train_X[:,test_size:]

	test_Y = train_Y[:,0:test_size]
	train_Y = train_Y[:,test_size:]

	labels = {0: 'cat', 1:'dog'}

	print('Finished!!!')

	return train_X, train_Y, test_X, test_Y, labels
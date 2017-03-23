import numpy as np
import pandas as pd

from sklearn.utils import shuffle


def relu(x):
	return x * (x > 0)


def error_rate(p, t):
	return np.mean(p != t)


def getKaggleMNIST():
	# MNIST data:
	# column 0 is labels
	# columns 1-785 is data with values 0...255
	# total size of csv: (42000, 1, 28, 28)
	train = pd.read_csv('Data/MNIST/train.csv').as_matrix().astype(np.float32)
	train = shuffle(train)

	Xtrain, Ytrain = train[:-1000, 1:] / 255, train[:-1000, 0].astype(np.int32)
	Xtest, Ytest = train[-1000:, 1:] / 255, train[-1000:, 0].astype(np.int32)

	return Xtrain, Ytrain, Xtest, Ytest


def init_weights(shape):
	return np.random.randn(*shape) / np.sqrt(sum(shape))

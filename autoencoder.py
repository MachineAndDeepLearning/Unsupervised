import numpy as np
import matplotlib.pyplot as plt

import theano
import theano.tensor as T

from sklearn.utils import shuffle

from util import relu, error_rate, getKaggleMNIST, init_weights


class AutoEncoder(object):
	def __init__(self, M, an_id):
		self.M = M
		self.id = an_id

	def fit(self, X, learning_rate=0.5, mu=0.99, epochs=10, batch_size=100, show_fig=False):
		N, D = X.shape
		n_batches = N // batch_size

		# initialize the layer weights
		W0 = init_weights((D, self.M))
		self.W = theano.shared(W0, 'W_%s' % self.id)
		self.bh = theano.shared(np.zeros(self.M), 'bh_%s' % self.id)
		self.bo = theano.shared(np.zeros(D), 'bo_%s' % self.id)

		# keep track of params
		self.params = [self.W, self.bh, self.bo]
		self.forward_params = [self.W, self.bh]

		# for momentum
		self.dW = theano.shared(np.zeros(W0.shape), 'dW_%s' % self.id)
		self.dbh = theano.shared(np.zeros(self.M), 'dbh_%s' % self.id)
		self.dbo = theano.shared(np.zeros(D), 'dbo_%s' % self.id)

		# keep track of momentum params
		self.dparams = [self.dW, self.dbh, self.dbo]
		self.forward_dparams = [self.dW, self.dbh]

		# define input varialbes
		X_in = T.matrix('X_%s' % self.id)
		X_hat = self.forward_output(X_in)

		H = T.nnet.sigmoid(X_in.dot(self.W) + self.bh)
		self.hidden_op = theano.function(
			inputs=[X_in],
			outputs=H,
		)

		# define cost
		# cost = ((X_in - X_hat) * (X_in - X_hat)).mean()
		cost = -(X_in * T.log(X_hat) + (1 - X_in) * T.log(1 - X_hat)).mean()

		cost_op = theano.function(
			inputs=[X_in],
			outputs=[cost]
		)

		# gradient descent
		updates = [
					(p, p + mu * dp - learning_rate * T.grad(cost, p)) for p, dp in zip(self.params, self.dparams)
					] + [
					(dp, mu * dp - learning_rate * T.grad(cost, p)) for p, dp in zip(self.params, self.dparams)
					]

		train_op = theano.function(
			inputs=[X_in],
			updates=updates,
		)

		costs = []
		print("Training autoencoder: %s" % self.id)
		for i in range(epochs):
			print("Epoch:", i)
			X = shuffle(X)

			for j in range(n_batches):
				batch = X[j * batch_size:(j * batch_size + batch_size)]
				train_op(batch)
				the_cost = cost_op(X)
				print("j / n_batches:", j, "/", n_batches, "cost:", the_cost)
				costs.append(the_cost)

		if show_fig:
			plt.plot(costs)
			plt.show()

	def forward_hidden(self, X):
		Z = T.nnet.sigmoid(X.dot(self.W) + self.bh)
		return Z

	def forward_output(self, X):
		Z = self.forward_hidden(X)
		Y = T.nnet.sigmoid(Z.dot(self.W.T) + self.bo)
		return Y


class DNN(object):
	def __init__(self, hidden_layer_sizes, UnsupervisedModel=AutoEncoder):
		self.hidden_layers = []
		count = 0;
		for M in hidden_layer_sizes:
			ae = UnsupervisedModel(M, count)
			self.hidden_layers.append(ae)
			count += 1

	def fit(self, X, Y, Xtest, Ytest, pretrain=True, learning_rate=0.01, mu=0.99, reg=0.1, epochs=1, batch_sz=100,
			show_fig=False):
		pretrain_epochs = 1
		if not pretrain:
			pretrain_epochs = 0

		current_input = X
		for ae in self.hidden_layers:
			ae.fit(current_input, epochs=pretrain_epochs)
			current_input = ae.hidden_op(current_input)

		# log reg weights
		N = len(Y)
		K = len(set(Y))
		W0 = init_weights((self.hidden_layers[-1].M, K))
		self.W = theano.shared(W0, 'W_logreg')
		self.b = theano.shared(np.zeros(K), 'b_logreg')

		self.params = [self.W, self.b]
		for ae in self.hidden_layers:
			self.params += ae.forward_params

		# momentum
		self.dW = theano.shared(np.zeros(W0.shape), 'dW_logreg')
		self.db = theano.shared(np.zeros(K), 'db_logreg')
		self.dparams = [self.dW, self.db]
		for ae in self.hidden_layers:
			self.dparams += ae.forward_dparams

		# input and outputs
		X_in = T.matrix('X_in')
		targets = T.ivector('Targets')
		pY = self.forward(X_in)

		# squared_magnitude = [(p*p).sum() for p in self.params]
		# reg_cost = T.sum(squared_magnitude)
		cost = -T.mean(T.log(pY[T.arange(pY.shape[0]), targets]))  # + reg*reg_cost
		prediction = self.predict(X_in)
		cost_predict_op = theano.function(
			inputs=[X_in, targets],
			outputs=[cost, prediction],
		)

		updates = [
					(p, p + mu * dp - learning_rate * T.grad(cost, p)) for p, dp in zip(self.params, self.dparams)
					] + [
					(dp, mu * dp - learning_rate * T.grad(cost, p)) for p, dp in zip(self.params, self.dparams)
					]

		train_op = theano.function(
			inputs=[X_in, targets],
			updates=updates,
		)

		n_batches = N // batch_sz
		costs = []

		print("Supervised trainng")
		for i in range(epochs):
			print("Epoch:", i)
			X, Y = shuffle(X, Y)

			for j in range(n_batches):
				Xbatch = X[j * batch_sz:(j * batch_sz + batch_sz)]
				Ybatch = Y[j * batch_sz:(j * batch_sz + batch_sz)]
				train_op(Xbatch, Ybatch)

				the_cost, the_prediction = cost_predict_op(Xtest, Ytest)
				error = error_rate(the_prediction, Ytest)
				print("j / n_batches:", j, "/", n_batches, "cost:", the_cost, "error rate:", error)
				costs.append(the_cost)

		if show_fig:
			plt.plot(costs)
			plt.show()

	def predict(self, X):
		return T.argmax(self.forward(X), axis=1)

	def forward(self, X):
		current_input = X
		for ae in self.hidden_layers:
			Z = ae.forward_hidden(current_input)
			current_input = Z
		Y = T.nnet.softmax(T.dot(current_input, self.W) + self.b)
		return Y


def main():
	Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()
	dnn = DNN([1000, 750, 500])
	dnn.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=3)
	# vs
	# dnn = DNN([1000, 750, 500])
	# dnn.fit(Xtrain, Ytrain, Xtest, Ytest, pretrain=False, epochs=10)


if __name__ == '__main__':
	main()

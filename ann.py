import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict


class NeuralNetwork(MLPClassifier):
    """
    Implementation of a fully connected neural network
    """
    # sizes of all layers
    layer_sizes = []
    # sizes of hidden layers
    hidden_layer_sizes = []
    # sizes of weights on each layer
    weight_sizes = []
    # regularization rate
    alpha = 0.0
    # vector of weights
    coefs_ = []
    # number of all layers
    n_layers_ = 0
    # number of hidden layers
    n_hidden_layers_ = 0
    # number of neurons on last layer
    n_outputs_ = 0
    # list of activations of every neuron
    activations = []
    # number of neurons on last layer
    n_classes = 0
    # matrix, consisting of samples and their attributes
    X = []
    # vector, consisting of classes to which the samples belong
    y = []
    # y in one hot encoding
    y_hot = []
    # regularization mask
    regularization_mask = []

    def __init__(self, hidden_layer_sizes, alpha):
        """
        Initializes parameters
        :param hidden_layer_sizes: list of sizes that represent the number of neurons in each hidden layer
        :param alpha: regularization rate
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha

    def set_data_(self, X, y):
        """
        Calculates and saves parameters and sizes, used in later methods
        :param X: matrix, consisting of samples and their attributes
        :param y: vector, consisting of classes to which the samples belong
        """
        self.X = X
        self.y = y
        # size of first layer of neurons
        start_layer_size = X.shape[1]
        self.n_classes = max(y) + 1
        self.layer_sizes = [start_layer_size] + self.hidden_layer_sizes + [self.n_classes]
        self.n_layers_ = len(self.layer_sizes)
        self.n_hidden_layers_ = len(self.hidden_layer_sizes)
        self.y_hot = self.one_hot_encode(y)


    def init_weights_(self):
        """
        Initializes the weights and calculates the regularization mask.
        The returned vector od weights also includes the weights originating from the bias neurons.
        :return: vector filled with weights
        """
        n_weights = 0
        self.weight_sizes = []
        self.regularization_mask = np.empty(0)
        np.random.seed(5)

        # iterates through the layer_sizes list and caluclates the number of weights an each level and the regularization mask
        for i in range(len(self.layer_sizes) - 1):
            # adds zeros to the regularization mask, where the weights originate from the bias neuron
            self.regularization_mask = np.append(self.regularization_mask, np.zeros(self.layer_sizes[i+1]))
            # adds ones to other weights on current level
            self.regularization_mask = np.append(self.regularization_mask, np.ones(self.layer_sizes[i] * self.layer_sizes[i+1]))

            # updates the weight_sizes list for the current layer
            self.weight_sizes += [(self.layer_sizes[i] + 1) * self.layer_sizes[i+1]]
            # adds the number of weights on current layer to the counter
            n_weights += self.weight_sizes[i]
        # returns vector of random weights ranging from 0 to 0.1
        return np.random.randn(n_weights) * 0.03

    def unflatten_coefs(self, coefs):
        """
        Transforms the weights vector to a list of matrices.
        Each matrix represents a different layer of weights.
        :param coefs: weights vector
        :return: list of matrices, consisting of weights on each layer
        """
        start_index = 0
        unflat = []
        # iterates through the layers and reshapes each vector of weights to a matrix and then appends it to the list
        for ind, val in enumerate(self.weight_sizes):
            unflat.append(coefs[start_index:(start_index + val)].reshape(self.layer_sizes[ind] + 1, self.layer_sizes[ind + 1]))
            start_index += val
        return unflat

    def fit(self, X, y):
        """
        Calculates the weights, that minimize the cost function.
        In other words, fits the model to training data (matrix X) and targets (vector y).
        :param X: matrix, consisting of samples and their attributes
        :param y: vector, consisting of classes to which the samples belong
        :return: self
        """
        self.set_data_(X, y)
        self.coefs_ = self.init_weights_()
        # performs optimized gradient descent
        x, _, _ = fmin_l_bfgs_b(self.cost, self.coefs_, self.grad)
        self.coefs_ = x

        return self

    def one_hot_encode(self, vector):
        """
        Transforms each value in list with one hot encoding
        :param vector: list of values
        :return: matrix, consisting of encoded input values
        """
        one_hot = np.zeros((len(vector), self.n_classes))
        # iterates through list and encodes each value
        for i in range(len(vector)):
            one_hot[i, vector[i]] = 1
        return one_hot

    def predict(self, X):
        """
        Performs a forward pass and returns the indices of a classes with the highest neuron output activation.
        In other words, predicts the classes, to which the input samples belong.
        :param X: matrix, consisting of samples and their attributes
        :return: vector of predicted classes
        """
        last_A = self.feed_forward(X, self.coefs_)
        return np.argmax(last_A, axis=1)

    def feed_forward(self, X, coefs):
        """
        Performs a forward pass and saves the activations of neurons on each layer.
        Iteratively calculates the activations on each layer based on the weights and activations of the previous layer.
        :param X: matrix, consisting of samples and their attributes
        :param coefs: weights vector
        :return: list of activation of neurons on the last layer
        """
        uf_coefs = self.unflatten_coefs(coefs)
        A = np.array(X)
        self.activations = []
        # iterates through the layers and calculates the activations of the next layer
        for layer in uf_coefs:
            # inserts a column of ones, which represents the biases
            A = np.insert(A, 0, 1, axis=1)
            self.activations.append(A)
            Z = np.dot(A, layer)
            A = logistic_function(Z)
        self.activations.append(A)
        return A

    def predict_proba(self, X):
        """
        Performs a forward pass and returns the probabilities of samples belonging to each class.
        :param X: matrix, consisting of samples and their attributes
        :return: matrix, where the first dimension represents samples and the second represents the probabilities, that
        the sample belongs to the class
        """
        last_A = self.feed_forward(X, self.coefs_)
        return last_A / np.sum(last_A, axis=1)[:, np.newaxis]

    def cost(self, coefs):
        """
        Calculates the cost of the weights.
        :param coefs:
        :return:
        """
        A = self.feed_forward(self.X, coefs)
        J = (1 / (2 * len(self.y))) * np.sum((A - self.y_hot) ** 2)
        regularization = np.dot(coefs ** 2, self.regularization_mask)
        return J + (self.alpha / 2) * regularization

    def grad(self, coefs):
        """
        Performs backpropagation and calculates the gradient of the weights.
        :param coefs: weights vector
        :return: vector, containing the gradients for each weight
        """
        uf_coefs = self.unflatten_coefs(coefs)
        self.feed_forward(self.X, coefs)

        # calculates gradients for the last layer
        d = np.multiply(np.multiply((self.activations[-1] - self.y_hot), self.activations[-1]), (1 - self.activations[-1]))
        D = np.dot((1 / len(self.y)) * np.transpose(self.activations[-2]), d)
        # adds gradients from the last layer to the final gradients vector
        Ds = D.flatten()

        # iterates from the last to the first layer and calculates weight gratients for each layer respectively
        for i in range(len(uf_coefs) - 1, 0, -1):
            d = np.multiply(np.multiply(np.dot(d, np.transpose(uf_coefs[i])), self.activations[i]), (1 - self.activations[i]))
            # removes the column belonging to bias weights from the matrix
            d = d[:, 1:]
            D = np.dot((1 / len(self.y)) * np.transpose(self.activations[i-1]), d)
            # adds gradients from the current layer to the final gradients vector
            Ds = np.append(D, Ds)
        reg_grad = self.alpha * coefs * self.regularization_mask
        return Ds + reg_grad

    def grad_approx(self, coefs, e):
        """
        Calculates the approximation of the gradient vector of the weights.
        :param coefs: weights vector
        :param e: small numeric value
        :return: numeric approximation of the gradient vector
        """
        grad = np.zeros(len(coefs))
        for i in range(len(coefs)):
            orig_coef = coefs[i]
            coefs[i] += e
            cost_plus_e = self.cost(coefs)
            coefs[i] -= 2 * e
            cost_minus_e = self.cost(coefs)
            grad[i] = (cost_plus_e - cost_minus_e) / (2 * e)
            coefs[i] = orig_coef
        return grad


def logistic_function(x):
    """
    Calculates the logistic (sigmoid) function on each element of the matrix
    :param x: matrix of elements, on which to perform the logistic function
    :return: matrix with values ranging from 0 to 1
    """
    return 1.0 / (1.0 + np.exp(-x))


def cross_validation():
    """
    Compares the implemented neural network to logistic regression and gradient boosting classifier.
    :return: list of F1 estimates gathered with cross validation on Iris data
    """
    k = 5
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    ann = cross_val_predict(NeuralNetwork([10, 2], alpha=1e-5), X, y, cv=k)
    reg = cross_val_predict(LogisticRegression(), X, y, cv=k)
    gra = cross_val_predict(GradientBoostingClassifier(), X, y, cv=k)
    ann_f1 = f1_score(y, ann, average='weighted')
    reg_f1 = f1_score(y, reg, average='weighted')
    gra_f1 = f1_score(y, gra, average='weighted')
    return ann_f1, reg_f1, gra_f1

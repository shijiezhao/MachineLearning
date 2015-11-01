"""
$Simple SVM implementation from this web: http://tullo.ch/articles/svm-py/
@Modified from Andrew Tulloch
@Author: Shijie Zhao, Yichen Shen, @Cambridge, @Nov,01,2015
@Mathematics background: /Users/shijie/Dropbox (MIT)/Learning/2015.03.Machine Learning/Exercise/6867midtermreview.pdf
"""
import numpy as np
import cvxopt.solvers
import logging
import numpy as np
import numpy.linalg as la


MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5
# determine whether a vector is support vector


class SVMTrainer(object):
    def __init__(self, kernel, c):
        self._kernel = kernel
        self._c = c

    def train(self, X, y):
        """
        Given the training features X with labels y, returns a SVM predictor representing the trained SVM. 
        """
        lagrange_multipliers = self._compute_multipliers(X, y)
        print lagrange_multipliers
        return self._construct_predictor(X, y, lagrange_multipliers)

    def _kernel_matrix(self, X):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self._kernel(x_i, x_j)
        return K

    def _construct_predictor(self, X, y, lagrange_multipliers):
    	"""
    	Math background:https://en.wikipedia.org/wiki/Support_vector_machine
    	"""
        support_vector_indices = lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER
        margin_vector_indices = self._c - lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER
        margin_support_vectors = X[margin_vector_indices]
        # e.g. [True False]

        support_multipliers = lagrange_multipliers[support_vector_indices]
        support_vectors = X[support_vector_indices]
        support_vector_labels = y[support_vector_indices]
        
        # M: S.V. on the margin
        M = []
        for i in range(len(support_multipliers)):
        	if (self._c - support_multipliers[i]) > MIN_SUPPORT_VECTOR_MULTIPLIER:
        		M.append(i)
        # Calc. beta0
        BB = 0
        for j in M:
        	BB = BB + support_vector_labels[j] - sum(alpha*y*self._kernel(support_vectors[j],x) for alpha,y,x in zip(support_multipliers,support_vector_labels,support_vectors))
        beta0 = BB/len(M)
        
        return SVMPredictor(
            kernel=self._kernel,
            beta0=beta0,
            weights=support_multipliers,
            support_vectors=support_vectors,
            support_vector_labels=support_vector_labels,
            margin_support_vectors=margin_support_vectors)

    def _compute_multipliers(self, X, y):   # Compute the Lagrange multipliers
    	# Quadratic programming to solve alpha's
    	# Web: http://cvxopt.org/userguide/coneprog.html
    	
        n_samples, n_features = X.shape
        
        K = self._kernel_matrix(X)
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))
        
        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))
        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)
        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)
        
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Lagrange multipliers
        return np.ravel(solution['x'])

class SVMPredictor(object):
    def __init__(self,
                 kernel,
                 beta0,
                 weights,
                 support_vectors,
                 support_vector_labels,
                 margin_support_vectors):
        self._kernel = kernel
        self._beta0 = beta0
        self._weights = weights
        self._support_vectors = support_vectors
        self._support_vector_labels = support_vector_labels
        self._margin_support_vectors=margin_support_vectors

    def predict(self, x):
        """
        Computes the SVM prediction on the given features x.
        """
        result = self._beta0
        for z_i, x_i, y_i in zip(self._weights,
                                 self._support_vectors,
                                 self._support_vector_labels):
            result += z_i * y_i * self._kernel(x_i, x)
        return np.sign(result).item()
    
    def project(self, x):
        """
        Computes the SVM prediction on the given features x.
        """
        result = self._beta0
        for z_i, x_i, y_i in zip(self._weights,
                                 self._support_vectors,
                                 self._support_vector_labels):
            result += z_i * y_i * self._kernel(x_i, x)
        return result


class Kernel(object):
    """Implements list of kernels from
    http://en.wikipedia.org/wiki/Support_vector_machine
    """
    @staticmethod
    def linear():
        def f(x, y):
            return np.inner(x, y)
        return f

    @staticmethod
    def gaussian(sigma):
        def f(x, y):
            exponent = -np.sqrt(la.norm(x-y) ** 2 / (2 * sigma ** 2))
            return np.exp(exponent)
        return f

    @staticmethod
    def _polykernel(dimension, offset):
        def f(x, y):
            return (offset + np.dot(x, y)) ** dimension
        return f

    @staticmethod
    def inhomogenous_polynomial(dimension):
        return Kernel._polykernel(dimension=dimension, offset=1.0)

    @staticmethod
    def homogenous_polynomial(dimension):
        return Kernel._polykernel(dimension=dimension, offset=0.0)

    @staticmethod
    def hyperbolic_tangent(kappa, c):
        def f(x, y):
            return np.tanh(kappa * np.dot(x, y) + c)
        return f
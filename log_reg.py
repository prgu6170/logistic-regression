#!/usr/bin/env python
# coding: utf-8
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pickle, gzip       
import numpy as np
from collections import defaultdict
from sklearn.utils import shuffle
import math

class Numbers:
    """
    Class to store MNIST data for images of 9 and 8 only
    """ 
    def __init__(self, location):
        # Load the dataset
        with gzip.open(location, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)
 
        self.train_x, self.train_y = train_set
        train_indices = np.where(self.train_y > 7)
        self.train_x, self.train_y = self.train_x[train_indices], self.train_y[train_indices]
        self.train_y = self.train_y - 8
 
        self.valid_x, self.valid_y = valid_set
        valid_indices = np.where(self.valid_y > 7)
        self.valid_x, self.valid_y = self.valid_x[valid_indices], self.valid_y[valid_indices]
        self.valid_y = self.valid_y - 8


class LogReg:
    
    def __init__(self, X, y, eta = 0.1):
        """
        Creates a logistic regression classifier
        :param num_features: The number of features (including bias)
        :param eta: Learning rate (the default is a constant value)
        :method: This should be the name of the method (sgd_update or mini_batch_descent)
        :batch_size: optional argument that is needed only in the case of mini_batch_descent
        """
        self.X = X
        self.y = y
        self.w = np.zeros(X.shape[1])
        self.eta = eta
        
    def calculate_score(self, x):
        """
        :param x: This can be a single training example or it could be n training examples
        :return score: Calculates the score that will plug into the logistic function
        """
        if len(np.shape(x)) != 1:
            self.score = []
            for each_x in x:
                self.each_score = np.dot(self.w, each_x.T)
                self.score.append(self.each_score)
        else:
            self.score = np.dot(self.w, x.T)
        return self.score
    
    def sigmoid(self, score):
        """
        :param score: Either a real valued number or a vector to convert into a number between 0 and 1
        :return sigmoid: Calcuate the output of applying the sigmoid function to the score. This could be a single
        value or a vector depending on the input
        """
        if isinstance(score, list):
            self.output = []
            for each_score in score:
                answer = 1 / (1 + np.exp(-each_score))
                self.output.append(answer)
        else:
            self.output = 1 / (1 + np.exp(-score))
        return self.output
    
    def compute_gradient(self, x, h, y):
        """
        :param x: Feature vector
        :param h: predicted class label
        :param y: real class label
        :return gradient: Return the derivate of the cost w.r.t to the weights
        """
        gradient = []
        for k in range(len(self.w)):
            gradient.append(0)
            gradient[k] = (h - y)*x[k]
        return gradient
        
     
    def sgd_update(self):
        """
        Computes a stochastic gradient update over the entire dataset to improve the log likelihood.
        :param x_i: The features of the example to take the gradient with respect to
        :param y: The target output of the example to take the gradient with respect to
        :return: Return the new value of the regression coefficients
        """
        for i in range(len(self.X)):
            self.h = self.sigmoid(self.calculate_score(self.X[i]))
            for k in range(len(self.X[0])):
                self.w[k] = self.w[k] - (self.eta*self.compute_gradient(self.X[i], self.h, self.y[i])[k])
        return self.w
        
    
    def mini_batch_update(self, batch_size):
        """
        One iteration of the mini-batch update over the entire dataset (one sweep of the dataset).
        :param X: NumPy array of features (size : no of examples X features)
        :param y: Numpy array of class labels (size : no of examples X 1)
        :param batch_size: size of the batch for gradient update
        :returns w: Coefficients of the classifier (after updating)
        """
        i = 0
        j = 1
        self.h = self.sigmoid(self.calculate_score(self.X))
        self.gradient = [0] * len(self.X[0])
        for x in self.X:
            if i < (batch_size*j):
                self.temp_gradient = self.compute_gradient(self.X[i], self.h[i], self.y[i])
                for k in range(len(self.X[0])):
                    self.gradient[k] += self.temp_gradient[k]
                i += 1
            else:
                j += 1
                for k in range(len(self.X[0])):
                    self.w[k] = self.w[k] - self.eta*self.gradient[k]
                self.h = self.sigmoid(self.calculate_score(self.X))
        else:
            for k in range(len(self.X[0])):
                self.w[k] = self.w[k] - self.eta*self.gradient[k]
        return self.w
    
    def progress(self, test_x, test_y, update_method, *batch_size):
        """
        Given a set of examples, computes the probability and accuracy
        :param test_x: The features of the test dataset to score
        :param test_y: The features of the test 
        :param update_method: The update method to be used, either 'sgd_update' or 'mini_batch_update'
        :param batch_size: Optional arguement to be given only in case of mini_batch_update
        :return: A tuple of (log probability, accuracy)
        """
        self.predictions = []
        if update_method == "sgd_update":
            self.weights = self.sgd_update()
        if update_method == "mini_batch_update":
            self.weights = self.mini_batch_update(batch_size[0])
        for x in test_x:
            self.h = self.sigmoid(self.calculate_score(x))
            if self.h >= 0.5:
                self.predictions.append(1)
            else:
                self.predictions.append(0)
        self.correct_predictions = 0
        self.log_probability = 0
        for i in range(len(test_y)):
            if test_y[i] == self.predictions[i]:
                correct_predictions += 1
            if test_y[i] == 1:
                self.log_probability += math.log(self.sigmoid(self.calculate_score(test_x[i])))
            else:
                self.log_probability += math.log(1-self.sigmoid(self.calculate_score(test_x[i])))
        self.accuracy = self.correct_predictions/len(test_y)
        
        return self.log_probability, self.accuracy


class LogReg1:
    
    def __init__(self, X, y, eta = 0.1):
        """
        Creates a logistic regression classifier
        :param num_features: The number of features (including bias)
        :param eta: Learning rate (the default is a constant value)
        :method: This should be the name of the method (sgd_update or mini_batch_descent)
        :batch_size: optional argument that is needed only in the case of mini_batch_descent
        """
        self.X = X
        self.y = y
        self.w = np.zeros(X.shape[1])
        self.eta = eta
        
    def calculate_score(self, x):
        """
        :param x: This can be a single training example or it could be n training examples
        :return score: Calculate the score that will plug into the logistic function
        """
        if len(np.shape(x)) != 1:
            self.score = []
            for each_x in x:
                self.each_score = np.dot(self.w, each_x.T)
                self.score.append(self.each_score)
        else:
            self.score = np.dot(self.w, x.T)
        return self.score
    
    def sigmoid(self, score):
        """
        :param score: Either a real valued number or a vector to convert into a number between 0 and 1
        :return sigmoid: Calcuate the output of applying the sigmoid function to the score. This could be a single
        value or a vector depending on the input
        """
        if isinstance(score, list):
            self.output = []
            for each_score in score:
                answer = 1 / (1 + np.exp(-each_score))
                self.output.append(answer)
        else:
            self.output = 1 / (1 + np.exp(-score))
        return self.output
    
    def compute_gradient(self, x, h, y):
        """
        :param x: Feature vector
        :param h: predicted class label
        :param y: real class label
        :return gradient: Returns the derivative of the cost w.r.t to the weights
        """
        gradient = []
        for k in range(len(self.w)):
            gradient.append(0)
            gradient[k] = (h - y)*x[k]
        return gradient
        
     
    def sgd_update(self):
        """
        Compute a stochastic gradient update over the entire dataset to improve the log likelihood.
        :param x_i: The features of the example to take the gradient with respect to
        :param y: The target output of the example to take the gradient with respect to
        :return: Return the new value of the regression coefficients
        """
        for i in range(len(self.X)):
            self.h = self.sigmoid(self.calculate_score(self.X[i]))
            self.gradient = self.compute_gradient(self.X[i], self.h, self.y[i])
            for k in range(len(self.X[0])):
                self.w[k] = self.w[k] - (self.eta*self.gradient[k])
        return self.w
    
    def mini_batch_update(self, batch_size):
        """
        One iteration of the mini-batch update over the entire dataset (one sweep of the dataset).
        :param X: NumPy array of features (size : no of examples X features)
        :param y: Numpy array of class labels (size : no of examples X 1)
        :param batch_size: size of the batch for gradient update
        :returns w: Coefficients of the classifier (after updating)
        """
        i = 0
        j = 1
        self.gradient = [0] * len(self.X[0])
        for x in self.X:
            if i < (batch_size*j):
                self.h = self.sigmoid(self.calculate_score(self.X[i]))
                self.temp_gradient = self.compute_gradient(self.X[i], self.h, self.y[i])
                for k in range(len(self.X[0])):
                    self.gradient[k] += self.temp_gradient[k]
                i += 1
            else:
                j += 1
                for k in range(len(self.X[0])):
                    self.w[k] = self.w[k] - self.eta*self.gradient[k]
        else:
            for k in range(len(self.X[0])):
                self.w[k] = self.w[k] - self.eta*self.gradient[k]
        return self.w
    
    def progress(self, test_x, test_y, update_method, epochs, *batch_size):
        """
        Given a set of examples, computes the probability and accuracy
        :param test_x: The features of the test dataset to score
        :param test_y: The features of the test
        :param update_method: The update method to be used, either 'sgd_update' or 'mini_batch_update'
        :param batch_size: Optional arguement to be given only in case of mini_batch_update
        :return: A tuple of (log probability, accuracy)
        """
        self.accuracy_list = []
        if update_method == "sgd_update":
            for epoch in range(1, epochs + 1):
                self.correct_predictions = 0
                self.weights = self.sgd_update()
                for i in range(len(test_x)):
                    self.h = self.sigmoid(self.calculate_score(test_x[i]))
                    if self.h >= 0.5:
                        if test_y[i] == 1:
                            self.correct_predictions += 1
                    else:
                        if test_y[i] == 0:
                            self.correct_predictions += 1
                self.accuracy = self.correct_predictions / len(test_y)
                self.accuracy_list.append(self.accuracy)
                self.X, self.y = shuffle(self.X, self.y)
        if update_method == "mini_batch_update":
            for epoch in range(1, epochs + 1):
                self.correct_predictions = 0
                self.weights = self.mini_batch_update(batch_size[0])
                for i in range(len(test_x)):
                    self.h = self.sigmoid(self.calculate_score(test_x[i]))
                    if self.h >= 0.5:
                        if test_y[i] == 1:
                            self.correct_predictions += 1
                    else:
                        if test_y[i] == 0:
                            self.correct_predictions += 1

                self.accuracy = self.correct_predictions / len(test_y)
                self.accuracy_list.append(self.accuracy)
                self.X, self.y = shuffle(self.X, self.y)
        return self.accuracy_list
    
    def calculate_h(self, test_x, test_y):
        self.accuracy_list = []
        self.correct_predictions = 0
        for i in range(len(test_x)):
            self.h = self.sigmoid(self.calculate_score(test_x[i]))
        return self.h


class PlotGraph:
    def __init__(self, epochs):
        self.epochs = int(epochs)
    """    
    def setup(self):
        num = Numbers('data/mnist.pklz')
        self.train_x = num.train_x
        self.train_y = num.train_y
        self.valid_x = num.valid_x
        self.valid_y = num.valid_y
    """
    def sgd_graph(self):
        self.num = Numbers('data/mnist.pklz')
        self.eta_range = [.0001, .01, .1, .5, 1]
        #self.eta_range = [.5, 1]
        for eta in self.eta_range:
            classifier = LogReg1(self.num.train_x, self.num.train_y, eta)
            self.accuracy_list_test = classifier.progress(self.num.valid_x, self.num.valid_y, 'sgd_update', self.epochs)
            plt.plot(range(1, self.epochs+1), self.accuracy_list_test, label="ETA_"+str(eta)+"_test")
            self.accuracy_list_train = classifier.progress(self.num.train_x, self.num.train_y, 'sgd_update', self.epochs)
            plt.plot(range(1, self.epochs+1), self.accuracy_list_train, label="ETA_"+str(eta)+"_train")
        plt.legend(loc='lower left')
        plt.title("SGD accuracy plots")
        plt.show()
    def mini_batch_graph(self):
        self.num = Numbers('data/mnist.pklz')
        self.eta_range = [.0001, .01, .1, .5, 1]
        for eta in self.eta_range:
            classifier = LogReg1(self.num.train_x, self.num.train_y, eta)
            self.accuracy_list_test = classifier.progress(self.num.valid_x, self.num.valid_y, 'mini_batch_update', self.epochs, 500)
            plt.plot(range(1, self.epochs+1), self.accuracy_list_test, label="ETA_"+str(eta)+"_test")
            self.accuracy_list_train = classifier.progress(self.num.train_x, self.num.train_y, 'mini_batch_update', self.epochs, 500)
            plt.plot(range(1, self.epochs+1), self.accuracy_list_train, label="ETA_"+str(eta)+"_train")
        plt.legend(loc='lower left')
        plt.title("Mini-batch-update accuracy plots")
        plt.show()

graph = PlotGraph(10)
graph.sgd_graph()
graph.mini_batch_graph()


class Numbers2:
    """
    Class to store MNIST data for images of 0-9
    """ 
    def __init__(self, location):
        # Load the dataset
        with gzip.open(location, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)
 
        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set


data2 = Numbers2('data/mnist.pklz')
print(data2.train_y[:10])
def view_digit(example, label=None):
    if label is not None: print("true label: {:d}".format(label))
    plt.imshow(example.reshape(28,28), cmap='gray');
view_digit(data2.train_x[18],data2.train_y[18])


class MultiLogReg:
    """
    Class to store MNIST data for images of 0-9
    """ 
    
    def __init__(self, X, y, eta = 0.1):
        #self.X = self.normalize_data(X)
        self.X = X
        self.y = self.one_hot_encoding(y)
        self.eta = eta
        self.reg_model = self.get_optimal_parameters() 
    def one_hot_encoding(self, y):
        self.encod = np.zeros((len(y), 10), dtype=int)
        for i in range(len(y)):
            self.encod[i][y[i]] = 1
    """
    def normalize_data(self, X):
        # TO DO: Normalize the dataset X using the mean and standard deviation of all the training examples 
        mean = np.mean(X, axis=0)
        std = std = np.std(X, axis=0)
        for i in range(len(X[0])):
            #print(len(X[0]))
            #print(i)
            if std[i] != 0.0:
                for j in range(len(X)):
                    #print(std)
                    X[j][i] = (X[j][i] - mean[i])/std[i]
                    #print(X[j][i])
    """
    def get_optimal_parameters(self):
        self.reg = []
        for i in range(10):
            self.reg.append(LogReg1(self.X, self.encod.T[i], self.eta))
            self.reg[i].mini_batch_update(200)
        return self.reg
    
    def predict(self, test_image, test_label):
        self.test_image = np.reshape(test_image, (-1, len(test_image)))
        self.test_label = np.array([test_label])
        self.probabilities = []
        
        for i in range(10):
            prblty = self.reg_model[i].calculate_h(self.test_image, self.test_label)
            self.probabilities.append([i, prblty])
        return self.probabilities
    

multi_classifier = MultiLogReg(data2.train_x, data2.train_y, 0.5)
print("\nPredict output:")
print(multi_classifier.predict(data2.test_x[208], data2.test_y[208]))
print("\nTest label is:")
print(data2.test_y[208])


class Accuracy:
    def __init__(self, train_x, train_y, eta = 0.01):
        self.train_x = train_x
        self.train_y = train_y
        self.eta = eta
        self.multi_classifier = MultiLogReg(self.train_x, self.train_y, self.eta)
    
    def get_predictions(self, test_x, test_y):
        self.train_x, self.train_y = shuffle(self.train_x, self.train_y)
        self.predictions = []
        for i in range(len(test_x)):
            self.predict = self.multi_classifier.predict(test_x[i], test_y[i])
            self.mx = 0
            self.high = 0.0
            for element in self.predict:
                if element[1] > self.high:
                    self.high = element[1]
                    self.mx = element[0]
            self.predictions.append(self.mx)
        return self.predictions
    
    def get_accuracy(self, test_x, test_y):
        self.predictions = self.get_predictions(test_x, test_y)
        self.correct = 0
        for i in range(len(self.predictions)):
            if self.predictions[i] == test_y[i]:
                self.correct += 1
        self.accuracy = self.correct/len(test_y)
        return self.accuracy
    
    def confusionMatrix(self, testX, testY):
        """
        Generates a confusion matrix for the given test set
        """
        C = np.zeros((10, 10), dtype=int)
        i = 0
        prediction_array = self.get_predictions(testX, testY)
        for i in range(len(testY)):
            C[testY[i]][prediction_array[i]] += 1
        return C

    
evaluate = Accuracy(data2.train_x, data2.train_y, 0.5)
test_accuracy = evaluate.get_accuracy(data2.test_x, data2.test_y)
train_accuracy = evaluate.get_accuracy(data2.train_x, data2.train_y)
print("\nAccuracy for test data:")
print(test_accuracy)
print("\nAccuracy for train data:")
print(train_accuracy)
print("\nConfusion matrix for test data:")
print(evaluate.confusionMatrix(data2.test_x, data2.test_y))
print("\nConfusion matrix for train data")
print(evaluate.confusionMatrix(data2.train_x, data2.train_y))

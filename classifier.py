import numpy as np
import math
import pickle
from mlxtend.data import loadlocal_mnist

class Classifier:
    w_classifier = None

    def init_weights(self, shape, a=-0.5, b=0.5):
        return a + (b - a) * np.random.random(shape)

    def dump(self, filename):
        if self.w_classifier is None:
            raise Exception("you should fit data before predict")

        with open(filename, 'wb') as f:
            pickle.dump(self.w_classifier, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.w_classifier = pickle.load(f)

    def cross_entropy_point(self, X, Y, weights):
        wt = np.array(weights).T
        predicted = np.matmul(X, wt)

        sum_ce = 0.0
        for i, p in enumerate(predicted):
            sum_exp = sum([math.exp(pi) for pi in p])
            soft_max_p = [math.exp(pi) / sum_exp for pi in p]

            sum_ce += sum(Y[i] * np.log(soft_max_p))
        return -1.0 * sum_ce / len(Y)

    def get_gradient(self, X, Y, weights, w_delta=0.1):
        wt = np.array(weights).T
        predicted = np.matmul(X, wt)

        soft_max_predicted = np.zeros(predicted.shape)
        for i in range(len(soft_max_predicted)):
            sum_exp = sum([math.exp(v) for v in predicted[i]])
            soft_max_predicted[i] = np.array([math.exp(v) / sum_exp for v in predicted[i]])

        deriv = soft_max_predicted - Y
        grad = np.matmul(deriv.T, X)
        grad /= len(X)

        return grad

    def gradient_descent(self, X, Y, lr=1.0, iter=300):
        _, features_num = X.shape
        w = self.init_weights((10, features_num))
        lr_decr = False
        current_ce = self.cross_entropy_point(X, Y, w)
        for i in range(iter):
            print('iter =', i)
            g = self.get_gradient(X, Y, w)
            nw = w - lr * g
            new_ce = self.cross_entropy_point(X, Y, nw)
            if new_ce < current_ce:
                w = nw
                current_ce = new_ce
                print("ce =", current_ce)
                if not lr_decr:
                    lr *= 2
                yield w
            else:
                lr *= 0.8
                lr_decr = True

    def fit(self, X, Y, X_validate, Y_validate, lr = 0.1, iter=300):
        best_ce_validate = None
        best_w = None
        for w in self.gradient_descent(X, Y, lr, iter):
            ce_validate = self.cross_entropy_point(X_validate, Y_validate, w)
            print("ce_validate =", ce_validate)
            if best_ce_validate is None or ce_validate < best_ce_validate:
                best_ce_validate = ce_validate
                best_w = w.copy()
        self.w_classifier = best_w.copy()

    def predict(self, X):
        if self.w_classifier is None:
            raise Exception("you should fit data before predict")

        wt = np.array(self.w_classifier).T
        predicted = np.matmul(X, wt)

        Y = [0] * len(X)
        for i, p in enumerate(predicted):
            cls = np.argmax(p)
            Y[i] = cls
        return Y

    @staticmethod
    def prepare_y_ohe(y):
        Y = []
        for yv in y:
            row_y = [0] * 10
            row_y[yv] = 1
            Y.append(row_y)
        return Y

    @staticmethod
    def norm_and_bias(x):
        x_norm = np.array(x) / 255
        ones = np.ones((len(x_norm), 1))
        return np.hstack([x_norm, ones])

    @staticmethod
    def loadData(images_path, labels_path):
        images, labels = loadlocal_mnist(
            images_path=images_path,
            labels_path=labels_path)
        return images, labels

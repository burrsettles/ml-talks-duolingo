"""
Burr Settles
Duolingo ML Dev Talk #1: Supervised Learning

Simple L2-regularized logistic regression, using stochastic gradient descent.
"""

import math
from collections import defaultdict, namedtuple
from random import shuffle

DEFAULT_SIGMA = 20.0
DEFAULT_ETA = 0.001

class LogisticRegression(object):

    def __init__(self, sigma=DEFAULT_SIGMA, eta=DEFAULT_ETA, init_weights=None, eval_func=eval_linear):
        super(LinearRegression, self).__init__()
        self.sigma = sigma          # L2 prior variance
        self.eta = eta              # initial learning rate
        self.weights = defaultdict(float)
        if init_weights is not None:
            for k in init_weights:
                self.weights[k] = init_weights[k]
        self.fcounts = None
        self.eval_func = eval_func    # evaluation method called after each iteration

    def predict(self, inst):
        a = sum([float(self.weights[k]) * inst.data[k] for k in inst.data])
        return min(1-1e-7, max(1e-7, 1./(1.+math.exp(-a)) ))

    def error(self, inst):
        return inst.target - self.predict(inst)

    def training_update(self, inst):
        err = self.error(inst)
        for k in inst.data:
            rate = inst.weight * self.eta / math.sqrt(1 + self.fcounts[k])
            # L2 regularization update
            self.weights[k] -= rate * self.weights[k] / self.sigma**2
            # error update
            self.weights[k] += rate * err * inst.data[k]
            # increment feature count for learning rate
            self.fcounts[k] += 1

    def train(self, trainset, iterations=500):
        for it in range(iterations):
            shuffle(trainset)
            for inst in trainset:
                self.training_update(inst)

class Instance(namedtuple('Instance', ['target', 'data', 'name', 'meta', 'weight'])):
    def __new__(cls, target, data, name=None, meta=None, weight=1.0):
        if not isinstance(target, (int, float)):
            raise Exception("Instance target must be a number.")
        if type(data) is not dict:
            raise Exception("Instance data must be a dict.")
        return super(Instance, cls).__new__(cls, float(target), data, name, meta, weight)
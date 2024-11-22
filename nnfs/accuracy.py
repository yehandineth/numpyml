import numpy as np

from Base import Accuracy

class RegressionAccuracy(Accuracy):

    def __init__(self):

        self.precision = None
    
    def fit(self, y, refit=False, tightness = 250):

        if self.precision is None or refit:

            self.precision = np.std(y)/tightness
    
    def compare(self, preds, truth):

        return (np.absolute(preds - truth) < self.precision)
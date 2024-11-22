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
    

class Classification_Accuracy(Accuracy):

    def __init__(self):

        pass

    def fit(self, y):

        pass

    def compare(self, preds, truth):

        # if len(truth.shape) == 2:
        #     truth = np.argmax(truth, axis=-1)
        return preds==truth
from Base import Loss
import numpy as np

class MSE(Loss):

    def __init__(self):

        pass

    def forward(self, preds, truth):

        mse = np.square(truth - preds) 

        return np.mean(mse, axis = -1)

    def backward(self, preds, truth):

        self.dinputs = -2 * ((truth - preds)/preds.shape[-1])  /preds.shape[0]  


class MAE(Loss):

    def __init__(self):

        pass

    def forward(self, preds, truth):

        mae = np.abs(truth - preds)

        return np.mean(mae, axis = -1)
    
    def backward(self, preds, truth):

        temp = np.sign(truth-preds)

        self.dinputs = (temp/preds.shape[-1]) /preds.shape[0]
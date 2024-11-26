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


class BinaryCrossentropy(Loss):

    def __init__(self):

        pass

    def forward(self, preds, truth):

        preds_clipped = np.clip(preds, 1e-7,1 - 1e-7)

        return np.mean(-(truth * np.log(preds_clipped) + (1-truth) * np.log(1 - preds_clipped)), axis=-1)

    def backward(self, preds ,truth):

        preds_clipped = np.clip(preds, 1e-7,1 - 1e-7)

        self.dinputs = (-(truth/preds_clipped - (1 - truth)/ (1 - preds_clipped))/preds_clipped.shape[-1])/preds_clipped.shape[0]


class CategoricalCrossentropy(Loss):

    def forward(self, preds, truth):

        n_samples = len(truth)

        clipped_preds = np.clip(preds, 1e-7, 1-1e-7)
     
        #if sparsely encoded
        if len(truth.shape) == 1:
            confidences = clipped_preds[range(n_samples), truth]
            truth = np.eye(len(preds[0]))[truth]
        
        #if one_hot encoded
        elif len(truth.shape) == 2:
            confidences = np.sum(clipped_preds * truth, axis=1)

        return -np.log(confidences)
    
    #Adding backward propagation
    def backward(self, truth, preds):

        n_samples = len(truth)
        
        if len(truth.shape) == 1:
            truth = np.eye(len(preds[0]))[truth]
        self.dinputs = (-truth/preds)/n_samples
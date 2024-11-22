import numpy as np

class Input():

    def __init__(self):

        pass

    def forward(self, inputs, training):
        
        self.inputs = inputs
        self.output = inputs

class Dense_Layer():

    def __init__(self, n_inputs, n_neurons, lambdal1w=0, lambdal1b=0, lambdal2w=0, lambdal2b=0, weight_multiplier = 0.01):

        self.weights = weight_multiplier * np.random.randn(n_inputs, n_neurons)#Already transposed
        self.biases = np.zeros(shape=(1, n_neurons))
        self.lambdal1w = lambdal1w
        self.lambdal1b = lambdal1b
        self.lambdal2w = lambdal2w
        self.lambdal2b = lambdal2b


    def forward(self, inputs : np.ndarray, training):
        self.output = np.dot(inputs, self.weights) + self.biases
        # Saving the inputs for backpropagation using
        self.inputs = inputs

    # Backward propagation method
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        self.dinputs = np.dot(dvalues, self.weights.T)

        if self.lambdal1w > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.lambdal1w * dL1

        if self.lambdal1b > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.lambdal1b * dL1
        
        if self.lambdal2w > 0:
            self.dweights +=  2 * self.lambdal2w * self.weights

        if self.lambdal2b > 0:
            self.dbiases += 2 * self.lambdal2b * self.biases


class Dropout():

    def __init__(self, rate):

        self.forward_rate = 1 - rate

    def forward(self, inputs, training=True):

        self.inputs = inputs
        
        if not training:

            self.output = inputs.copy()

            return

        self.binary_mask = np.random.binomial(1, self.forward_rate, size=inputs.shape)/self.forward_rate

        self.output = inputs * self.binary_mask
    
    def backward(self, dvalues):

        self.dinputs = dvalues * self.binary_mask
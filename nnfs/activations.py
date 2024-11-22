import numpy as np

class LinearActivation():

    def __init__(self):

        pass

    def forward(self, inputs):

        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):

        self.dinputs = dvalues.copy()

    def predict(self, outputs):

        return outputs
    
class ReLU_Activation():

    def __init__(self):

        pass

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        #Save the inputs for backpropagation
        self.inputs = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predict(self, outputs):

        return outputs
    
class Softmax_Activation():

    def __init__(self):

        pass

    def forward(self, inputs):

        self.inputs = inputs

        exps = np.exp(inputs-np.max(inputs, axis=-1, keepdims=True))
        self.output = exps/ np.sum(exps, axis=-1, keepdims=True)
    
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues) 
        for i , (output, dvalue) in enumerate(zip(self.output, dvalues)):
            output = np.expand_dims(output, axis=-1)
            jac = np.diagflat(output) - np.dot(output, output.T)

            self.dinputs[i] = np.dot(jac,dvalue)
    
    def predict(self, outputs):

        return np.argmax(outputs, axis=-1)
    

class Sigmoid():

    def __init__(self):

        pass

    def forward(self, inputs):

        self.inputs = inputs

        self.output = 1/(1 + np.exp(-inputs))

    def backward(self, dvalues):

        self.dinputs = dvalues * self.output * ( 1 - self.output)

    def predict(self, outputs):

        return (outputs > 0.5) *1
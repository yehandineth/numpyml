import numpy as np 

class Optimizer():

    def __init__(self, learning_rate = 1.0, decay = 0.):

        self.learning_rate = learning_rate
        self.decay = decay

        self.current_rate = learning_rate

        self.steps = 0

    def pre_update(self):

        if self.decay:

            self.current_rate = self.learning_rate / (1 + self.decay * self.steps)

    def post_update(self):

        self.steps += 1

class SGD(Optimizer):

    def __init__(self, learning_rate = 1.0, decay = 0., momentum=0.):

        super().__init__(learning_rate=learning_rate, decay=decay)
        
        self.momentum = momentum

        
    def update_params(self, layer):

        if not hasattr(layer, 'weight_momentum'):
            layer.weight_momentum = 0
            layer.bias_momentum = 0

        layer.weight_momentum = -(self.current_rate * layer.dweights) + (layer.weight_momentum * self.momentum)
        layer.bias_momentum = -(self.current_rate * layer.dbiases) + (layer.bias_momentum * self.momentum)

        layer.weights += layer.weight_momentum
        layer.biases +=  layer.bias_momentum

class AdaGrad(Optimizer):

    def __init__(self, learning_rate = 1.0, decay = 0., epsilon=1e-7):

        super().__init__(learning_rate=learning_rate, decay=decay)
        self.epsilon = epsilon
    
    def update_params(self, layer):

        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        w_update = -(self.current_rate * layer.dweights)/(np.sqrt(layer.weight_cache) + self.epsilon)
        b_update = -(self.current_rate * layer.dbiases)/(np.sqrt(layer.bias_cache) + self.epsilon)

        layer.weights += w_update
        layer.biases += b_update


class RMSProp(Optimizer):

    def __init__(self, learning_rate = 0.001, decay = 0., epsilon=1e-7, rho=0.9):

        super().__init__(learning_rate=learning_rate, decay=decay)
        self.epsilon = epsilon
        self.rho = rho

    def update_params(self, layer):

        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + (1-self.rho)*layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1-self.rho)*layer.dbiases**2

        w_update = -(self.current_rate * layer.dweights)/(np.sqrt(layer.weight_cache) + self.epsilon)
        b_update = -(self.current_rate * layer.dbiases)/(np.sqrt(layer.bias_cache) + self.epsilon)

        layer.weights += w_update
        layer.biases += b_update

class Adam(Optimizer):

    def __init__(self, learning_rate = 0.001, decay = 0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):

        super().__init__(learning_rate=learning_rate, decay=decay)
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def update_params(self, layer):

        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.bias_momentum = np.zeros_like(layer.biases)

        layer.weight_momentum = (1 - self.beta_1) * layer.dweights + (layer.weight_momentum * self.beta_1)
        layer.bias_momentum = (1 - self.beta_1) * layer.dbiases + (layer.bias_momentum * self.beta_1)

        weight_momentum_corrected = layer.weight_momentum/(1 - self.beta_1 ** (self.steps + 1))
        bias_momentum_corrected = layer.bias_momentum/(1 - self.beta_1 ** (self.steps + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1-self.beta_2)*layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1-self.beta_2)*layer.dbiases**2

        weight_cache_corrected = layer.weight_cache/(1 - self.beta_2 ** (self.steps + 1))
        bias_cache_corrected = layer.bias_cache/(1 - self.beta_2 ** (self.steps + 1))

        w_update = -(self.current_rate * weight_momentum_corrected)/(np.sqrt(weight_cache_corrected) + self.epsilon)
        b_update = -(self.current_rate * bias_momentum_corrected)/(np.sqrt(bias_cache_corrected) + self.epsilon)

        layer.weights += w_update
        layer.biases += b_update

import numpy as np
from layers import Input

class Accuracy():

    def calculate(self, preds, truth):

        return np.mean(self.compare(preds, truth))
    
class Loss():

    def calculate(self, preds, y):

        losses = self.forward(preds, y)

        return np.mean(losses), self.regularization_loss()
    
    def regularization_loss(self):

        regularization_loss = 0

        for layer in self.trainable_layers:

            if layer.lambdal1w > 0:
                regularization_loss += layer.lambdal1w * np.sum(np.abs(layer.weights))

            if layer.lambdal1b > 0:
                regularization_loss += layer.lambdal1b * np.sum(np.abs(layer.biases))
            
            if layer.lambdal2w > 0:
                regularization_loss += layer.lambdal2w * np.sum(np.square(layer.weights))

            if layer.lambdal2b > 0:
                regularization_loss += layer.lambdal2b * np.sum(np.square(layer.biases))

        return regularization_loss
    
    def get_trainable_layers(self, trainable_layers):

        self.trainable_layers = trainable_layers

class Model():

    def __init__(self):

        self.layers = []

    def add(self, layer):

        self.layers.append(layer)

    def compile(self,*, optimizer, loss : Loss, accuracy : Accuracy):

        self.optimizer = optimizer

        self.loss = loss
    
        self.accuracy = accuracy
        self.finalize()

    def fit(self, X, y,*, epochs=10000, print_frequency=100):

        self.accuracy.fit(y)
        self.current_history = {
            'losses' : [],
            'accuracies' : [],
            'learning_rates' : [],
        }
        
        for epoch in range(1,epochs+1):
            
            output = self.forward(X)

            data_loss, reg_loss = self.loss.calculate(output, y)

            loss = data_loss + reg_loss

            predictions = self.predictor.predict(output)
            
            accuracy = self.accuracy.calculate(predictions, y)

            self.backward(output, y)

            self.optimize()

            self.current_history['losses'].append(loss)
            self.current_history['accuracies'].append(accuracy)
            self.current_history['learning_rates'].append(self.optimizer.current_rate)
    
            if epoch%print_frequency == 0 or epoch == epochs:
                print('--------------------------------------------------------------------------------------------------------------------------')
                print('Epoch :', epoch, 'Learning_rate :', self.optimizer.current_rate)
                print('Training Accuracy :', accuracy,  'Training Loss', loss, 'Training Data Loss :',
                    data_loss, 'Regularization Loss:', reg_loss)
        
        return self.current_history

    def finalize(self):

        self.input_layer = Input()

        self.trainable_layers = []

        for i,layer in enumerate(self.layers):
            if hasattr(layer, 'weights'):
                self.trainable_layers.append(layer)

            if i == 0:
                layer.prev = self.input_layer
                layer.next = self.layers[i+1]
            elif i==len(self.layers) - 1:
                layer.prev = self.layers[i-1]
                layer.next = self.loss
                self.predictor = layer
            else:
                layer.prev = self.layers[i-1]
                layer.next = self.layers[i+1]

        self.loss.get_trainable_layers(self.trainable_layers)

    def forward(self, X):
        
        self.input_layer.forward(X)

        for layer in self.layers:
            layer.forward(layer.prev.output)

        return layer.output
    
    def backward(self, output ,y):

        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
    
    def optimize(self):
        
        self.optimizer.pre_update()
        for layer in self.trainable_layers:
            self.optimizer.update_params(layer)
        self.optimizer.post_update()
        

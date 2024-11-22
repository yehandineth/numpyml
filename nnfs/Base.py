import numpy as np
from layers import Input
from activations import Softmax_Activation
from losses import CategoricalCrossentropy

class Accuracy():

    def calculate(self, preds, truth):

        return np.mean(self.compare(preds, truth))
    

class Loss():

    def calculate(self, preds, y, *, regularize=False):

        data_loss = np.mean(self.forward(preds, y))

        if not regularize:

            return data_loss

        return data_loss, self.regularization_loss()
    
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


class CategoricalCrossentropyWithSoftmax():

    def __init__(self):
        
        pass
    
    def backward(self, preds, truth):

        samples = preds.shape[0]

        if len(truth.shape) == 2:
            truth = np.argmax(truth, axis = -1)

        self.dinputs = preds.copy()

        self.dinputs[range(samples), truth] -= 1

        self.dinputs /= samples



class Model():

    def __init__(self):

        self.layers = []
        self.softmax_classifier = None

    def add(self, layer):

        self.layers.append(layer)

    def compile(self,*, optimizer, loss : Loss, accuracy : Accuracy):

        self.optimizer = optimizer

        self.loss = loss
    
        self.accuracy = accuracy
        self.finalize()

    def fit(self, X, y,*,validation_data=None, epochs=10000, print_frequency=100):
        
        X_val,y_val = None,None
        if validation_data is not None:
            X_val,y_val =  validation_data
        self.accuracy.fit(y)
        self.current_history = {
            'losses' : [],
            'accuracies' : [],
            'learning_rates' : [],
        }
        if validation_data is not None:
            self.current_history['val_accuracies'] = []
            self.current_history['val_losses'] = []
        
        for epoch in range(1,epochs+1):
            
            output = self.forward(X, training=True)

            data_loss, reg_loss = self.loss.calculate(output, y, regularize=True)

            loss = data_loss + reg_loss

            predictions = self.predictor.predict(output)
            
            accuracy = self.accuracy.calculate(predictions, y)

            self.backward(output, y)

            self.optimize()

            self.current_history['losses'].append(loss)
            self.current_history['accuracies'].append(accuracy)
            self.current_history['learning_rates'].append(self.optimizer.current_rate)

            if validation_data is not None:
                val_output = self.forward(X_val, training=False)

                val_loss = self.loss.calculate(val_output, y_val)

                val_predictions = self.predictor.predict(val_output)
            
                val_accuracy = self.accuracy.calculate(val_predictions, y_val)

                self.current_history['val_accuracies'].append(val_accuracy)
                self.current_history['val_losses'].append(val_loss)

    
            if epoch%print_frequency == 0 or epoch == epochs:
                print('--------------------------------------------------------------------------------------------------------------------------')
                print('Epoch :', epoch, 'Learning_rate :', self.optimizer.current_rate)
                print('Training Accuracy :', accuracy,  'Training Loss', loss, 'Training Data Loss :',
                    data_loss, 'Regularization Loss:', reg_loss)
                if validation_data is not None:
                    print('Validation Accuracy :', val_accuracy,  'Validation Loss :', val_loss)

                
        
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

        if isinstance(self.layers[-1], Softmax_Activation) and isinstance(self.loss, CategoricalCrossentropy):
            
            self.softmax_classifier = CategoricalCrossentropyWithSoftmax()

        self.loss.get_trainable_layers(self.trainable_layers)

    def forward(self, X, training=True):
        
        self.input_layer.forward(X, training=training)

        for layer in self.layers:
            layer.forward(layer.prev.output, training=training)

        return layer.output
    
    def backward(self, output ,y):

        if self.softmax_classifier is not None:
            self.softmax_classifier.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier.dinputs
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
        else:
            self.loss.backward(output, y)
            for layer in reversed(self.layers):
                layer.backward(layer.next.dinputs)
    
    def optimize(self):
        
        self.optimizer.pre_update()
        for layer in self.trainable_layers:
            self.optimizer.update_params(layer)
        self.optimizer.post_update()
        

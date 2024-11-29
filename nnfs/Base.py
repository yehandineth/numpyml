import numpy as np
from layers import Input
from tqdm import tqdm

class Accuracy():

    def calculate(self, preds, truth):

        batch_accuracy = self.compare(preds, truth)

        self.accumulated_accuracy += np.sum(batch_accuracy)
        self.samples_passed += len(batch_accuracy)

        return np.mean(batch_accuracy)
    
    def get_accumulated_accuracy(self):

        return self.accumulated_accuracy/self.samples_passed
    
    def reset_accuracy(self):

        self.accumulated_accuracy = 0
        self.samples_passed = 0

class Loss():

    def calculate(self, preds, y, *, regularize=False):

        batch_loss =self.forward(preds, y)
        
        data_loss = np.mean(batch_loss)

        self.accumulated_loss += np.sum(batch_loss)
        self.samples_passed += len(batch_loss)

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

    def get_accumulated_loss(self, regularize=False):

        data_loss = self.accumulated_loss / self.samples_passed

        if not regularize:

            return data_loss

        return data_loss, self.regularization_loss()

    def reset_loss(self):

        self.accumulated_loss = 0
        self.samples_passed = 0


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

    @staticmethod
    def load(path : str):

        import pickle

        with open(path,'rb') as file:
            model = pickle.load(file)
        
        return model

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

    def fit(self, X, y,*,validation_data=None, epochs=1, batch_size=None):
        
        X_val,y_val = None,None
        val_batch_size = batch_size
        
        self.accuracy.fit(y)
        self.current_history = {
            'losses' : [],
            'accuracies' : [],
            'learning_rates' : [],
        }
        if validation_data is not None:
            self.current_history['val_accuracies'] = []
            self.current_history['val_losses'] = []
            X_val,y_val =  validation_data

        if batch_size is None:
            batch_size = X.shape[0]
            if validation_data is not None:
                val_batch_size = X_val.shape[0]
        
        num_steps = int(np.ceil(X.shape[0]/batch_size))

        for epoch in range(1,epochs+1):
            
            print('--------------------------------------------------------------------------------------------------------------------------')
            print('Epoch :', epoch, 'Learning_rate :', self.optimizer.current_rate)
            self.accuracy.reset_accuracy()
            self.loss.reset_loss()

            bar = tqdm(range(num_steps))
            for step in bar:
            
                start = step*batch_size
                num_samples = batch_size if step!=num_steps-1 else (X.shape[0] - step*batch_size)

                batch_X = X[start:start + num_samples]
                batch_y = y[start:start + num_samples]


                output = self.forward(batch_X, training=True)

                self.loss.calculate(output, batch_y, regularize=True)

                data_loss, reg_loss = self.loss.get_accumulated_loss(regularize=True)

                batch_loss = data_loss + reg_loss

                predictions = self.predictor.predict(output)

                self.accuracy.calculate(predictions, batch_y)
            
                batch_accuracy = self.accuracy.get_accumulated_accuracy()

                self.backward(output, batch_y)

                self.optimize()
                bar.set_description(f'Accuracy : {batch_accuracy} Loss : {batch_loss}' )

            loss = batch_loss
            accuracy = batch_accuracy

            self.current_history['losses'].append(loss)
            self.current_history['accuracies'].append(accuracy)
            self.current_history['learning_rates'].append(self.optimizer.current_rate)

            if validation_data is not None:

                val_loss,val_accuracy = self.evaluate(X_val,y_val,batch_size=val_batch_size)

            self.current_history['val_accuracies'].append(val_accuracy)
            self.current_history['val_losses'].append(val_loss)

            print('Training Accuracy :', accuracy,  'Training Loss', loss, 'Training Data Loss :',
                data_loss, 'Regularization Loss:', reg_loss)
            if validation_data is not None:
                print('Validation Accuracy :', val_accuracy,  'Validation Loss :', val_loss)

        
        return self.current_history
    
    def evaluate(self, X_val, y_val,*, batch_size=None):

        if batch_size is not None:
            num_val_steps = int(np.ceil(X_val.shape[0]/batch_size)) 
        else:
            num_val_steps = 1
            batch_size = X_val.shape[0]
        val_bar = tqdm(range(num_val_steps))
        self.accuracy.reset_accuracy()
        self.loss.reset_loss()

        for step in val_bar:

            start = step*batch_size
            num_samples = batch_size if step!=num_val_steps-1 else (X_val.shape[0] - step*batch_size)

            batch_X = X_val[start:start + num_samples]
            batch_y = y_val[start:start + num_samples]

            val_output = self.forward(batch_X, training=False)

            self.loss.calculate(val_output, batch_y)

            val_loss = self.loss.get_accumulated_loss()

            val_predictions = self.predictor.predict(val_output)

            self.accuracy.calculate(val_predictions, batch_y)
        
            val_accuracy = self.accuracy.get_accumulated_accuracy()

            val_bar.set_description(f'Validation Accuracy : {val_accuracy} Validation Loss : {val_loss}' )
        
        return val_loss,val_accuracy


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
        from losses import CategoricalCrossentropy
        from activations import Softmax_Activation

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

    def predict(self, X, *,batch_size=None):

        if batch_size is not None:
            num_steps = int(np.ceil(X.shape[0]/batch_size)) 
        else:
            num_steps = 1
            batch_size = X.shape[0]
        bar = tqdm(range(num_steps))
   
        output = []

        for step in bar:

            start = step*batch_size
            num_samples = batch_size if step!=num_steps-1 else (X.shape[0] - step*batch_size)

            batch_X = X[start:start + num_samples]

            batch_output = self.forward(batch_X, training=False)

            output.append(batch_output)
        
        return np.vstack(output)
    
    def get_weights(self):

        weights = []
        for layer in self.trainable_layers:
            weights.append(layer.get_weights())
        
        return weights
    
    def set_weights(self, weights):

        for weight,layer in zip(weights,self.trainable_layers):

            layer.set_weights(*weight)

    def clone_model(self, reset=True):
        
        import copy
        
        model = copy.deepcopy(self)

        self.accuracy.reset_accuracy()
        self.loss.reset_loss()
        model.input_layer.__dict__.pop( 'output' , None )
        model.loss.__dict__.pop( 'dinputs' , None )
        for layer in model.layers:
            for property in [ 'inputs' , 'output' , 'dinputs' ,'dweights' , 'dbiases' ]:
                layer.__dict__.pop( property , None )

        return model

    def save_weights(self, path : str = 'model.weights'):

        import pickle

        with open(path,'wb') as file:
            
            pickle.dump(self.get_weights(),file)
        
        print('Successfully saved weights to', path)

    def save_model(self, path : str = 'model.model'):
        
        import pickle

        with open(path,'wb') as file:
            
            pickle.dump(self.clone_model(),file)
        
        print('Successfully saved model to', path)
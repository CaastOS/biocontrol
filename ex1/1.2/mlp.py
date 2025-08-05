import numpy as np
import matplotlib.pyplot as plt

from perceptron import Perceptron, Sigmoid, LinearActivation

class Layer:
    def __init__(self, num_inputs, num_units, act_f):
        """ 
        Initialize the layer, creating `num_units` perceptrons with `num_inputs` each. 
        """
        self.num_inputs = num_inputs
        self.num_units = num_units
        
        self.ps = []
        for i in range(num_units):
            self.ps.append(Perceptron(num_inputs, act_f, weight_init_std=0.5))

    def activation(self, x):
        """ Returns the activation `a` of all perceptrons in the layer, given the input vector`x`. """
        return np.array([p.activation(x) for p in self.ps])

    def output(self, a):
        """ Returns the output `o` of all perceptrons in the layer, given the activation vector `a`. """
        return np.array([p.output(ai) for p, ai in zip(self.ps, a)])

    def predict(self, x):
        """ Returns the output `o` of all perceptrons in the layer, given the input vector `x`. """
        return np.array([p.predict(x) for p in self.ps])

    def gradient(self, a):
        """ Returns the gradient of the activation function for all perceptrons in the layer, given the activation vector `a`. """
        return np.array([p.gradient(ai) for p, ai in zip(self.ps, a)])

    def update_weights(self, dw):
        """ 
        Update the weights of all of the perceptrons in the layer, given the weight change of each.
        Input size: (n_inputs+1, n_units)
        """
        for i in range(self.num_units):
            self.ps[i].w += dw[:,i]

    @property
    def w(self):
        """
            Returns the weights of the neurons in the layer.
            Size: (n_inputs+1, n_units)
        """
        return np.array([p.w for p in self.ps]).T

    def import_weights(self, w):
        """ 
            Import the weights of all of the perceptrons in the layer.
            Input size: (n_inputs+1, n_units)
        """
        for i in range(self.num_units):
            self.ps[i].w = w[:,i]

class MLP:
    """ 
        Multi-layer perceptron class

    Parameters
    ----------
    n_inputs : int
        Number of inputs
    n_hidden_units : int
        Number of units in the hidden layer
    n_outputs : int
        Number of outputs
    alpha : float
        Learning rate used for gradient descent
    """
    def __init__(self, num_inputs, n_hidden_units, n_outputs, alpha=1e-3):
        self.num_inputs = num_inputs
        self.n_hidden_units = n_hidden_units
        self.n_outputs = n_outputs
        self.alpha = alpha

        # Define layers with proper weight initialization
        self.l1 = Layer(num_inputs, n_hidden_units, Sigmoid)
        self.l_out = Layer(n_hidden_units, n_outputs, LinearActivation)

    def predict(self, x):
        # Forward pass through hidden layer
        a1 = self.l1.activation(x)
        o1 = self.l1.output(a1)
        
        # Forward pass through output layer
        a_out = self.l_out.activation(o1)
        y = self.l_out.output(a_out)
        
        return y

    def train(self, inputs, outputs):
        """
         Train the network

        Parameters
        ----------
        `x` : numpy array
            Inputs (size: n_examples, n_inputs)
        `t` : numpy array
            Targets (size: n_examples, n_outputs)
        """

        n_examples = inputs.shape[0]
        
        # Initialize weight change accumulators
        dw1 = np.zeros_like(self.l1.w)
        dw_out = np.zeros_like(self.l_out.w)
        
        for i in range(n_examples):
            inp = inputs[i]
            target = outputs[i]
            
            # Forward pass
            a1 = self.l1.activation(inp)
            o1 = self.l1.output(a1)
            
            a_out = self.l_out.activation(o1)
            y = self.l_out.output(a_out)
            
            # Backpropagation
            grad_out = self.l_out.gradient(a_out)
            delta_out = grad_out * (target - y)
            
            # Hidden layer error
            grad1 = self.l1.gradient(a1)
            weighted_sum = np.dot(self.l_out.w[1:, :], delta_out)
            delta1 = grad1 * weighted_sum.flatten()
            
            # Weight updates
            o1_with_bias = np.insert(o1, 0, 1)
            dw_out += np.outer(o1_with_bias, delta_out)
            
            inp_with_bias = np.insert(inp, 0, 1)
            dw1 += np.outer(inp_with_bias, delta1)
        
        # Apply weight updates
        self.l1.update_weights(self.alpha * dw1 / n_examples)
        self.l_out.update_weights(self.alpha * dw_out / n_examples)

    def export_weights(self):
        return [self.l1.w, self.l_out.w]
    
    def import_weights(self, ws):
        if ws[0].shape == self.l1.w.shape and ws[1].shape == self.l_out.w.shape:
            print("Importing weights..")
            self.l1.import_weights(ws[0])
            self.l_out.import_weights(ws[1])
        else:
            print("Sizes do not match")

def calc_prediction_error(model, x, t):
    """ Calculate the average prediction error """
    n_examples = x.shape[0]
    total_error = 0
    
    for i in range(n_examples):
        prediction = model.predict(x[i])
        error = (prediction - t[i]) ** 2
        total_error += np.sum(error)
    
    return total_error / n_examples

if __name__ == "__main__":

    # Question 2 ############################################################
    # Create a layer with 5 neurons, each taking 2 inputs (plus bias)
    layer = Layer(2, 5, Sigmoid)

    # Test input [π, 1]^T
    test_input = np.array([np.pi, 1])

    # Print output
    output = layer.predict(test_input)
    print("Output of layer with 5 neurons:")
    print(output)

    # Print weights
    print("\nWeights of the whole layer:")
    print(layer.w)
    print(f"Weight matrix dimensions: {layer.w.shape}")
    ########################################################################
    
    # Training data for XOR gate
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Try different learning rates
    learning_rates = [1.0, 0.75, 0.5, 0.1, 0.01, 0.001]
    
    for lr in learning_rates:
        print(f"\n=== Testing Learning Rate: {lr} ===")
        
        # Initialize network for XOR with current learning rate
        mlp_xor = MLP(2, 2, 1, alpha=lr)
        
        # Test initial error
        initial_error = calc_prediction_error(mlp_xor, X, y)
        print(f"Initial MSE: {initial_error:.6f}")
        
        # Train the network
        epochs = 2000
        errors = []
        
        for epoch in range(epochs):
            mlp_xor.train(X, y)
            if epoch % 200 == 0:
                error = calc_prediction_error(mlp_xor, X, y)
                errors.append(error)
                print(f"Epoch {epoch}, MSE: {error:.6f}")
        
        # Final test
        final_error = calc_prediction_error(mlp_xor, X, y)
        print(f"Final MSE: {final_error:.6f}")
        print("Final XOR predictions:")
        for i in range(len(X)):
            pred = mlp_xor.predict(X[i])
            print(f"Input: {X[i]}, Target: {y[i][0]}, Prediction: {pred[0]:.4f}")
        
        # If we get good results, break
        if final_error < 1e-5:
            print(f"SUCCESS with learning rate {lr}!")
            
            # Plot MSE vs epochs
            plt.figure(figsize=(10, 6))
            epochs_plot = list(range(0, epochs, 200)) + [epochs-1]
            errors_plot = errors + [final_error]
            plt.plot(epochs_plot, errors_plot, 'b-o')
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.title(f'Training Error vs Epochs (XOR Gate, LR={lr})')
            plt.yscale('log')
            plt.grid(True)
            plt.show()
            
            # Plot decision boundary
            plt.figure(figsize=(8, 6))
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                                np.linspace(y_min, y_max, 500))
            grid = np.c_[xx.ravel(), yy.ravel()]
            Z = np.array([mlp_xor.predict(pt)[0] for pt in grid])
            Z = Z.reshape(xx.shape)

            # Decision boundary and soft regions
            plt.pcolormesh(xx, yy, Z, shading='auto', cmap='coolwarm', alpha=0.8)
            contour = plt.contour(xx, yy, Z, levels=[0.5], colors='k', linewidths=2)

            # Plot input points
            for cls in [0, 1]:
                idx = np.where(y.flatten() == cls)
                plt.scatter(X[idx, 0], X[idx, 1], label=f'Class {cls}', edgecolors='k', marker='o')

            plt.title(f'Decision Boundary (XOR Gate, LR={lr})')
            plt.xlabel('Input 1')
            plt.ylabel('Input 2')
            plt.colorbar(label='Output')
            plt.grid(True)
            plt.legend()
            plt.show()
            
            break


# Considerations
# 2. As expected, the weight matrix has shape (3, 5). 3 rows (2 inputs + 1 bias) and 5 columns (5 neurons).
# 3. The output layer has n_hidden_units inputs.
# 7a. To get perfect results (only 1.0 and 0.0 predicted), around 1200 epochs are needed.
# 7b. For this configuration higher learning rates seem to work better, with 0.5 being the first optimal.

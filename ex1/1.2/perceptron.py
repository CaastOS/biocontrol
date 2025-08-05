import numpy as np
from activation import ActivationFunction
import matplotlib.pyplot as plt

class SignActivation(ActivationFunction):
    """ 
    Sign activation: `f(x) = 1 if x > 0, 0 if x <= 0`
    """
    def forward(self, x):
        """
        Return 1 for inputs greater than 0, and 0 otherwise.
        """
        return np.where(x > 0, 1, 0)
    
    def gradient(self, x):
        """
        Function derivative.
        The gradient of the sign function is not well-defined at x=0 and is 0 elsewhere.
        """
        return 0 if x != 0 else None

class Sigmoid(ActivationFunction):
    def forward(self, x):
        """Computes the sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        """Computes the gradient of the sigmoid function."""
        fx = self.forward(x)
        return fx * (1 - fx)


class LinearActivation(ActivationFunction):
    def forward(self, x):
        """The output is the same as the input."""
        return x

    def gradient(self, x):
        """The gradient is always 1."""
        return np.ones_like(x)

class Perceptron:
    """ 
    Perceptron neuron model
    
    Parameters
    ----------
    n_inputs : int
        Number of inputs
    act_f : Subclass of `ActivationFunction`
        Activation function
    """
    def __init__(self, n_inputs, act_f, weight_init_std=0.01):
        """
        Perceptron class initialization.
        Initializes weights randomly and sets the activation function.
        """

        # Initialize weights with a small random value.
        self.n_inputs = n_inputs
        self.reset_weights(weight_init_std)
        
        # Instantiate the activation function
        self.f = act_f()

        if self.f is not None and not isinstance(self.f, ActivationFunction):
            raise TypeError("self.f should be a class instance.")

    def reset_weights(self, std_dev):
        """
        Resets the weights using a normal distribution with the given standard deviation.
        This is useful for the grid search to ensure a fresh start for each trial.
        """
        mean = 0
        self.w = np.random.normal(mean, std_dev, size=self.n_inputs + 1)

    def activation(self, x):
        """
        Computes the activation `a` given an input `x`.
        a = w0*1 + w1*x1 + w2*x2 + ...
        """
        x_with_bias = np.insert(x, 0, 1)
        a = np.dot(self.w, x_with_bias)
        return a

    def output(self, a):
        """
        Computes the neuron output `y`, given the activation `a`.
        """
        y = self.f.forward(a)
        return y

    def predict(self, x):
        """
        Computes the neuron output `y`, given the input `x`.
        """
        activation = self.activation(x)
        return self.output(activation)

    def gradient(self, a):
        """
        It computes the gradient of the activation function, given the activation `a`.
        """
        return self.f.gradient(a)

if __name__ == '__main__':
    data = np.array([
        [0.5, 0.5, 0], [1.0, 0, 0], [2.0, 3.0, 0], 
        [0, 1.0, 1], [0, 2.0, 1], [1.0, 2.2, 1]
    ])
    xdata = data[:, :2]
    ydata = data[:, 2]

    # Grid Search Parameters
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    weight_init_stds = [0.001, 0.01, 0.1, 1.0]
    epochs = 1000
    convergence_threshold = 1e-15

    results = []

    # Initialize a single perceptron instance for the grid search
    perceptron = Perceptron(n_inputs=2, act_f=SignActivation)

    for lr in learning_rates:
        for std in weight_init_stds:
            # Reset weights for each new combination of hyperparameters
            perceptron.reset_weights(std)
            epochs_to_converge = epochs
            best_weights = None
            
            print(f"\nTesting: Learning Rate = {lr}, Weight Init Std = {std}")
            
            for epoch in range(epochs):
                total_error = 0
                for x_input, y_target in zip(xdata, ydata):
                    prediction = perceptron.predict(x_input)
                    error = y_target - prediction
                    
                    if error != 0:
                        x_with_bias = np.insert(x_input, 0, 1)
                        perceptron.w += lr * error * x_with_bias
                    
                    total_error += abs(error)
                
                if total_error < convergence_threshold:
                    epochs_to_converge = epoch + 1
                    best_weights = perceptron.w.copy()
                    break
            
            if best_weights is not None:
                print(f"Converged in {epochs_to_converge} epochs.")
                print(f"Weights: {best_weights}")
            else:
                print(f"Did not converge within {epochs} epochs. Final total error: {total_error:.4f}")
                
            results.append({
                'lr': lr, 
                'std': std, 
                'epochs': epochs_to_converge, 
                'weights': best_weights
            })

    # Find the best result
    converged_results = [r for r in results if r['weights'] is not None]

    if not converged_results:
        print("No combinations converged. Try a larger search space or more epochs.")
    else:
        best_result = min(converged_results, key=lambda x: x['epochs'])
        print("--" * 40)
        print(f"Best Learning Rate: {best_result['lr']}")
        print(f"Best Weight Initialization Std Dev: {best_result['std']}")
        print(f"Epochs to Converge: {best_result['epochs']}")
        print(f"Best Weights: {best_result['weights']}")

        # Plotting the best result
        plt.figure(figsize=(8, 6))
        
        plt.scatter(xdata[:, 0], xdata[:, 1], c=ydata, cmap='viridis', edgecolors='k', label='Data Points')
        xp = np.linspace(np.min(xdata[:,0]) - 1, np.max(xdata[:,0]) + 1, 100)
        
        best_w = best_result['weights']
        if best_w[2] != 0:
            yp = -(best_w[1] * xp + best_w[0]) / best_w[2]
            plt.plot(xp, yp, 'k--', label='Decision Boundary')

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(f"Perceptron Decision Boundary (Best Model: LR={best_result['lr']}, Std={best_result['std']})")
        plt.legend()
        plt.grid(True)
        plt.xlim(np.min(xdata[:,0]) - 1, np.max(xdata[:,0]) + 1)
        plt.ylim(np.min(xdata[:,1]) - 1, np.max(xdata[:,1]) + 1)
        plt.show()

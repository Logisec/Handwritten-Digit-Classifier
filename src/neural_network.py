import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize a simple two-layer neural network.

        Args:
            input_size (int): Size of the input layer (784 for MNIST)
            hidden_size (int): Size of the hidden layer
            output_size (int): Size of the output layer (10 for digit classification)
            learning_rate (float): Learning rate for gradient descent
        """
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU function."""
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        """Softmax activation function with numerical stability."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """
        Forward pass through the neural network.

        Args:
            X (numpy.ndarray): Input data of shape (batch_size, input_size)

        Returns:
            tuple: (output, cache) where cache contains intermediate values
        """
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self.relu(z1)

        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self.softmax(z2)

        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}

        return a2, cache

    def backward(self, y, cache):
        """
        Backward pass to compute gradients.

        Args:
            y (numpy.ndarray): True labels as one-hot encodings
            cache (dict): Cached values from forward pass

        Returns:
            dict: Gradients for each parameter
        """
        batch_size = y.shape[0]

        X = cache["X"]
        a1 = cache["a1"]
        a2 = cache["a2"]
        z1 = cache["z1"]

        dz2 = a2 - y
        dW2 = (1 / batch_size) * np.dot(a1.T, dz2)
        db2 = (1 / batch_size) * np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(z1)
        dW1 = (1 / batch_size) * np.dot(X.T, dz1)
        db1 = (1 / batch_size) * np.sum(dz1, axis=0, keepdims=True)

        gradients = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

        return gradients

    def update_parameters(self, gradients):
        """
        Update network parameters using calculated gradients.

        Args:
            gradients (dict): Gradients for each parameter
        """
        self.W1 -= self.learning_rate * gradients["W1"]
        self.b1 -= self.learning_rate * gradients["b1"]
        self.W2 -= self.learning_rate * gradients["W2"]
        self.b2 -= self.learning_rate * gradients["b2"]

    def train_step(self, X, y):
        """
        Perform one training step (forward pass, backward pass, update parameters).

        Args:
            X (numpy.ndarray): Batch of training examples
            y (numpy.ndarray): Batch of training labels (one-hot encoded)

        Returns:
            float: Loss for this batch
        """
        output, cache = self.forward(X)
        gradients = self.backward(y, cache)
        self.update_parameters(gradients)
        loss = -np.mean(np.sum(y * np.log(np.clip(output, 1e-10, 1.0)), axis=1))

        return loss

    def predict(self, X):
        """
        Make predictions for input X.

        Args:
            X (numpy.ndarray): Input data

        Returns:
            numpy.ndarray: Predicted classes (0-9)
        """
        output, _ = self.forward(X)
        return np.argmax(output, axis=1)

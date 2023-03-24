# Imports
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike

class NeuralNetwork:
    """
    This is a class that generates a fully-connected neural network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            A list of dictionaries describing the layers of the neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}]
            will generate a two-layer deep network with an input dimension of 64, a 32 dimension hidden layer, and an 8 dimensional output.
        lr: float
            Learning rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            (see nn_arch above)
    """

    def __init__(
        self,
        nn_arch: List[Dict[str, Union[int, str]]],
        lr: float,
        seed: int,
        batch_size: int,
        epochs: int,
        loss_function: str
    ):

        # Save architecture
        self.arch = nn_arch

        # Save hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size

        # Initialize the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD! IT IS ALREADY COMPLETE!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """

        # Seed NumPy
        np.random.seed(self._seed)

        # Define parameter dictionary
        param_dict = {}

        # Initialize each layer's weight matrices (W) and bias matrices (b)
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1

        return param_dict

    def _single_forward(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        A_prev: ArrayLike,
        activation: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        # Basic test that compatible activation functin was chosen.
        assert activation in ['sigmoid', 'relu'], 'The activation function must be either sigmoid, or relu.'

        # Calculate layer linear transformed matrix (z).
        Z_curr = A_prev.dot(W_curr.T) + b_curr.T

        # Use new z matrix to calculate activation matrix for new current layer.
        if activation == 'sigmoid':
            A_curr = self._sigmoid(Z_curr)
        else:
            A_curr = self._relu(Z_curr)
        
        return A_curr, Z_curr

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        # Initialize cache with input matrix.
        cache = {'A0' : X}
        # Initialize first activation matrix.
        A_prev = X

        # For each layer within the network, calculate forward weights, biases, and activations.
        for L in range(1, len(self.arch) + 1): # Dictionary references begin from index 1 - start here and add 1.'
            A_curr, Z_curr = self._single_forward(self._param_dict[f"W{L}"],
                                                  self._param_dict[f"b{L}"],
                                                  A_prev,
                                                  self.arch[L-1]["activation"]) # Update values for one layer.
            
            cache[f"A{L}"] = A_curr # Save current activation matrix in cache.
            cache[f"Z{L}"] = Z_curr # Save current layer linear transformed matrix in cache.

            A_prev = A_curr # Update activation matrix to compute next step.

        y_hat = A_prev

        return y_hat, cache

    def _single_backprop(
        self,
        W_curr: ArrayLike,
        b_curr: ArrayLike,
        Z_curr: ArrayLike,
        A_prev: ArrayLike,
        dA_curr: ArrayLike,
        activation_curr: str
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        # Basic test that compatible activation functin was chosen.
        assert activation_curr in ['sigmoid', 'relu'], 'The current activation function must be either sigmoid, or relu.'

        # Backprop depends on current activation function.
        if activation_curr == 'sigmoid':
            dZ_curr = self._sigmoid_backprop(dA_curr, Z_curr)
        else:
            dZ_curr = self._relu_backprop(dA_curr, Z_curr)
        
        # Calculate all relevant derivatives.
        dA_prev = (dZ_curr).dot(W_curr)
        dW_curr = (A_prev.T).dot(dZ_curr).T # Needs to align with dimensions of self._param_dict[Wx]
        db_curr = np.sum(dZ_curr, axis = 0).reshape(b_curr.shape) # Ensure that has same dimensions as bias.

        return dA_prev, dW_curr, db_curr

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        # Initialize gradient dictionary.
        grad_dict = {}

        # Calculate dA based on the chosen loss function.
        if self._loss_func == 'bce':
            dA_curr = self._binary_cross_entropy_backprop(y, y_hat)
        else:
            dA_curr = self._mean_squared_error_backprop(y, y_hat)

        for L in range(1, len(self.arch) + 1)[::-1]:
            # For each layer, get weights, bias, layer linear transformation matrix, partial derivative
            # of loss function with respect to activation matrix, and activation.
            W_curr, b_curr, Z_curr, A_prev, dA_curr, activation_curr = (self._param_dict[f"W{L}"],
                                                                        self._param_dict[f"b{L}"],
                                                                        cache[f"Z{L}"],
                                                                        cache[f"A{L - 1}"],
                                                                        dA_curr,
                                                                        self.arch[L - 1]['activation']
            )

            # Run single instance of backpropagation.
            dA_prev, dW_curr, db_curr = self._single_backprop(W_curr,
                                                              b_curr,
                                                              Z_curr,
                                                              A_prev,
                                                              dA_curr,
                                                              activation_curr
            )

            # Store gradient information into dictionary.
            grad_dict[f"dA_prev{L}"] = dA_prev
            grad_dict[f"dW_curr{L}"] = dW_curr
            grad_dict[f"db_curr{L}"] = db_curr
            
            # Update activation for following layers.
            dA_curr = dA_prev

        return grad_dict
    
    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.
        """
        # Update internal parameters of network, this is how model learns.
        for L in range(1, len(self.arch) + 1): # Layers start with 1, make sure to update all layers by adding 1 to length of architecture.
            # Subtract weights and bias gradient taken from the backpropagation, from the forward values, multiplied by the learning rate.
            self._param_dict[f"W{L}"] = self._param_dict[f"W{L}"] - self._lr * grad_dict[f"dW_curr{L}"]
            self._param_dict[f"b{L}"] = self._param_dict[f"b{L}"] - self._lr * grad_dict[f"db_curr{L}"]

    def fit(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike
    ) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network by backpropagation for the number of epochs defined at
        the initialization of this class instance.

        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        # Set model_fit attribute to True.
        self._model_fit = True

        # Initialize lists for storing epoch losses.
        per_epoch_loss_train, per_epoch_loss_val = [], []

        # Separate training data into mini batches by calculating from size of data.
        num_batch = np.ceil( len(y_train) / self._batch_size)

        # Initialize first epoch.
        epoch = 1

        # Iterate across each epoch.
        while epoch < self._epochs:
            # Shuffle training data.
            shuffle_index = np.random.permutation( len(y_train) )
            shuffled_X = X_train[shuffle_index]
            shuffled_y = y_train[shuffle_index]

            # Split shuffled training data into mini batches.
            X_batch = np.array_split(shuffled_X, num_batch)
            y_batch = np.array_split(shuffled_y, num_batch)

            # Initialize and track epoch loss within each entry.
            entry_epoch_loss_train, entry_epoch_loss_val = [], []

            # For one epoch iterate across.
            for X, y in zip(X_batch, y_batch):
                # Perform forward pass.
                y_hat, cache = self.forward(X)

                # Get y hat for validation set.
                y_val_hat = self.predict(X_val)

                # Calculate losses based on loss functions.
                if self._loss_func == 'mse':
                    loss_train = self._mean_squared_error(y, y_hat)
                    loss_val = self._mean_squared_error(y_val, y_val_hat)
                else:
                    loss_train = self._binary_cross_entropy(y, y_hat)
                    loss_val = self._binary_cross_entropy(y_val, y_val_hat)

                # Append losses across batches.
                entry_epoch_loss_train.append(loss_train)
                entry_epoch_loss_val.append(loss_val)

                # Perform backpropagation and update parameters.
                grad_dict = self.backprop(y, y_hat, cache)
                self._update_params(grad_dict)
            
            # Append mean epoch loss from training and validation.
            per_epoch_loss_train.append( np.mean(entry_epoch_loss_train) )
            per_epoch_loss_val.append( np.mean(entry_epoch_loss_val) )
            
            # Increment epoch.
            epoch += 1

        return per_epoch_loss_train, per_epoch_loss_val

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        # Make sure that a model has been fit before returning any prediction.
        assert self._model_fit == True, 'Please run .fit prior to running predict.'

        # Return y_hat from the forward run.
        return self.forward(X)[0]

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        # Converting Z to float array should help with exponential function.
        Z = Z.astype(float)

        return 1 / (1 + np.exp(-Z) )

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
         # Sigmoid derivative is f′(z)=f(z)(1−f(z)) where f(z) is sigmoid function.
        dz = self._sigmoid(Z) * ( 1 - self._sigmoid(Z) )
        return dz * dA

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        # Returns element in Z, if element is bigger than 0, otherwise returns 0.
        return np.maximum(0, Z)  

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        # Rectified linear function has gradient 0 when z is less than or equal to 0 and 1 otherwise.
        dz = np.where(Z <= 0, 0, 1)
        return dz * dA

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        # Adjust 0 and 1 values in y_hat to prevent NaN values.
        y_hat[y_hat == 0] = 0.000001
        y_hat[y_hat == 1] = 0.999999

        loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat) )
        return loss

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        # Adjust 0 and 1 values in y_hat to prevent NaN values.
        y_hat[y_hat == 0] = 0.000001
        y_hat[y_hat == 1] = 0.999999

        # Partial derivative of loss with respect to A.
        dA = ( (1 - y)/(1 - y_hat) - (y / y_hat) ) / len(y)
        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        return np.mean( (y - y_hat) ** 2 )

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative for backprop.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        return (-2 * (y - y_hat) ) / len(y)
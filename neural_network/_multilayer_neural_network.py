import numpy as np
from neural_network.utils import sigmoid, relu, tanh


class _BaseMultilayerNeuralNetwork:
    def __init__(
        self,
        random_state,
        layer_dims,
        activation,
        learning_rate,
        num_iterations,
    ):
        self.random_state = random_state
        self.layer_dims = layer_dims
        self.activation = activation
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def _initialize_parameters(self):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network

        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                        bl -- bias vector of shape (layer_dims[l], 1)
        """

        np.random.seed(self.random_state)

        parameters = {}
        L = len(self.layer_dims)

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(self.layer_dims[l],
                                                       self.layer_dims[l-1]) / np.sqrt(self.layer_dims[l-1])  # *0.01
            parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))

            assert (parameters['W' + str(l)].shape ==
                    (self.layer_dims[l], self.layer_dims[l-1]))
            assert (parameters['b' + str(l)].shape == (self.layer_dims[l], 1))

        self.parameters = parameters

        return parameters

    def _forward_pass(self, X, parameters):
        """
        Implement forward propagation for the [LINEAR->RELU or TANH]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()

        Returns:
        AL -- last post-activation value
        caches -- list of caches containing:
                    every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                    the cache of linear_sigmoid_forward() (there is one, indexed L-1)
        """

        caches = []
        A = X
        L = len(parameters) // 2   # number of layers in the neural network

        # Implement [LINEAR -> RELU or TANH]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A
            A, cache = self._linear_activation_forward(
                A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation=self.activation)
            caches.append(cache)

        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        AL, cache = self._linear_activation_forward(
            A, parameters['W' + str(L)], parameters['b' + str(L)], activation="sigmoid")
        caches.append(cache)

        assert (AL.shape == (1, X.shape[1]))

        return AL, caches

    def _linear_activation_forward(self, A_prev, W, b, activation):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        A -- the output of the activation function, also called the post-activation value
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
                  stored for computing the backward pass efficiently
        """

        if activation == "sigmoid":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = self._linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)

        elif activation == "relu":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = self._linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)

        elif activation == "tanh":
            # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
            Z, linear_cache = self._linear_forward(A_prev, W, b)
            A, activation_cache = tanh(Z)

        else:
            print(
                "\033[91mError! Please make sure you have passed the value correctly in the \"activation\" parameter")

        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

    def _linear_forward(self, A, W, b):
        """
        Implement the linear part of a layer's forward propagation.

        Arguments:
        A -- activations from previous layer (or input data): (size of previous layer, number of examples)
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (size of the current layer, 1)

        Returns:
        Z -- the input of the activation function, also called pre-activation parameter
        cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
        """

        Z = W.dot(A) + b

        assert (Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)

        return Z, cache

    def _compute_cost(self, AL, Y):
        """
        Implement the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

        Returns:
        cost -- cross-entropy cost
        """

        m = Y.shape[1]

        # Compute loss from aL and y.
        cost = (1./m) * (-np.dot(Y, np.log(AL).T) -
                         np.dot(1-Y, np.log(1-AL).T))

        # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        cost = np.squeeze(cost)
        assert (cost.shape == ())

        return cost

    def _backprop(self, AL, Y, caches):
        """
        Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" or "tanh" (there are (L-1) or them, indexes from 0 to L-2)
                    the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)

        Returns:
        grads -- A dictionary with the gradients
                grads["dA" + str(l)] = ...
                grads["dW" + str(l)] = ...
                grads["db" + str(l)] = ...
        """
        grads = {}
        L = len(caches)  # the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

        # Initializing the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
        current_cache = caches[L-1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)
                                                            ] = self._linear_activation_backward(dAL, current_cache, activation="sigmoid")

        for l in reversed(range(L-1)):
            # lth layer: (RELU -> LINEAR) gradients.
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self._linear_activation_backward(
                grads["dA" + str(l + 1)], current_cache, activation=self.activation)
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads

    def _linear_activation_backward(self, dA, cache, activation):
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.

        Arguments:
        dA -- post-activation gradient for current layer l
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = self._relu_backward(dA, activation_cache)
            dA_prev, dW, db = self._linear_backward(dZ, linear_cache)

        elif activation == "sigmoid":
            dZ = self._sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self._linear_backward(dZ, linear_cache)

        elif activation == "tanh":
            dZ = self._tanh_backward(dA, activation_cache)
            dA_prev, dW, db = self._linear_backward(dZ, linear_cache)

        else:
            print(
                "\033[91mError! Please make sure you have passed the value correctly in the \"activation\" parameter")

        return dA_prev, dW, db

    def _relu_backward(self, dA, cache):
        """
        Implement the backward propagation for a single RELU unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """

        Z = cache
        dZ = np.array(dA, copy=True)  # just converting dz to a correct object.

        # When z <= 0, you should set dz to 0 as well.
        dZ[Z <= 0] = 0

        assert (dZ.shape == Z.shape)

        return dZ

    def _sigmoid_backward(self, dA, cache):
        """
        Implement the backward propagation for a single SIGMOID unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """

        Z = cache

        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)

        assert (dZ.shape == Z.shape)

        return dZ

    def _tanh_backward(self, dA, cache):
        """
        Implement the backward propagation for a single TANH unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """

        Z = cache
        dZ = dA * (1.0 - np.tanh(Z) ** 2)

        assert (dZ.shape == Z.shape)

        return dZ

    def _linear_backward(self, dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1./m * np.dot(dZ, A_prev.T)
        db = 1./m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)

        return dA_prev, dW, db

    def _update_parameters(self, parameters, grads, learning_rate):
        """
        Update parameters using gradient descent

        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients, output of _backprop

        Returns:
        parameters -- python dictionary containing your updated parameters
                    parameters["W" + str(l)] = ...
                    parameters["b" + str(l)] = ...
        """

        L = len(parameters) // 2  # number of layers in the neural network

        # Update rule for each parameter. Use a for loop.
        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - \
                learning_rate * grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - \
                learning_rate * grads["db" + str(l+1)]

        return parameters


class MNNClassifier(_BaseMultilayerNeuralNetwork):

    def __init__(
        self,
        random_state=None,
        layer_dims=[],
        activation="relu",
        learning_rate=0.0075,
        num_iterations=3000
    ):
        super().__init__(
            random_state=random_state,
            layer_dims=layer_dims,
            activation=activation,
            learning_rate=learning_rate,
            num_iterations=num_iterations,

        )

    def fit(self, X, Y):
        """
        Implements a multilayer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

        Arguments:
        X -- input data, of shape (n_x, number of examples)
        Y -- true "label" vector (containing 1 if cat, 0 if non-cat), of shape (1, number of examples)
        layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
        learning_rate -- learning rate of the gradient descent update rule
        num_iterations -- number of iterations of the optimization loop

        Returns:
        parameters -- parameters learnt by the model. They can then be used to predict.
        """

        np.random.seed(self.random_state)
        costs = []

        # Parameters initialization.
        parameters = self._initialize_parameters()

        # Loop (gradient descent)
        for i in range(0, self.num_iterations):

            # Forward propagation: [LINEAR -> RELU or TANH]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = self._forward_pass(X, parameters)

            # Compute cost.
            cost = self._compute_cost(AL, Y)

            # Backward propagation.
            grads = self._backprop(AL, Y, caches)

            # Update parameters.
            parameters = self._update_parameters(
                parameters, grads, self.learning_rate)

            # Print the cost every 100 iterations
            if i % 100 == 0 or i == self.num_iterations - 1:
                print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
            if i % 100 == 0 or i == self.num_iterations:
                costs.append(cost)

        return parameters, costs

    def predict(self, X, y):
        """
        This function is used to predict the results of a multilayer neural network.

        Arguments:
        X -- data set of examples you would like to label
        parameters -- parameters of the trained model

        Returns:
        p -- predictions for the given dataset X
        """

        m = X.shape[1]
        n = len(self.parameters) // 2  # number of layers in the neural network
        p = np.zeros((1, m))

        # Forward propagation
        probas, caches = self._forward_pass(X, self.parameters)

        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0

        print("Accuracy: " + str(np.sum((p == y)/m)))

        return p


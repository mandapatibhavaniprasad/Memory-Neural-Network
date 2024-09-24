import copy
import numpy as np

class MemoryNeuralNetwork:
    def __init__(self, number_of_input_neurons=2, number_of_hidden_neurons=6, number_of_output_neurons=2, neeta=4e-5, neeta_dash=4e-5, lipschitz_norm=1.0, spectral_norm=False, seed_value=16981):
        #nn means network neuron.
        #mn means memory neuron.

        np.random.seed(seed_value)

        self.number_of_input_neurons = number_of_input_neurons
        self.number_of_hidden_neurons = number_of_hidden_neurons
        self.number_of_output_neurons = number_of_output_neurons
        self.squared_error = 0.0

        self.spectral_norm = spectral_norm
        self.lipschitz = lipschitz_norm
        
        #Set the learning rates for the weight updates and memory coefficients updates:
        self.neeta = neeta
        self.neeta_dash = neeta_dash
        
        #Initialise memory coefficients randomly
        self.alpha_input_layer = np.random.rand(self.number_of_input_neurons)
        self.alpha_hidden_layer = np.random.rand(self.number_of_hidden_neurons)
        self.alpha_last_layer = np.random.rand(self.number_of_output_neurons)
        
        #Initialise weights of the network randomly
        self.beta = np.random.rand(self.number_of_output_neurons)
        self.weights_input_to_hidden_nn = np.random.rand(self.number_of_input_neurons, self.number_of_hidden_neurons)
        self.weights_hidden_to_output_nn = np.random.rand(self.number_of_hidden_neurons, self.number_of_output_neurons)
        self.weights_input_to_hidden_mn = np.random.rand(self.number_of_input_neurons, self.number_of_hidden_neurons)
        self.weights_hidden_to_output_mn = np.random.rand(self.number_of_hidden_neurons, self.number_of_output_neurons)
        #self.bias_input_to_hidden_layer = 0.01 * (np.random.rand(self.number_of_hidden_neurons) - 0.5) #1x6
        #self.bias_hidden_to_output_layer = 0.01 * (np.random.rand() - 0.5) #1x1
        
        #Initialise past values as zeros
        self.prev_output_of_input_layer_nn = np.zeros(self.number_of_input_neurons)
        self.prev_output_of_input_layer_mn = np.zeros(self.number_of_input_neurons)
        self.prev_output_of_hidden_layer_nn = np.zeros(self.number_of_hidden_neurons)
        self.prev_output_of_hidden_layer_mn = np.zeros(self.number_of_hidden_neurons)
        self.prev_output_of_nn = np.zeros(self.number_of_output_neurons)
        self.prev_output_of_mn = np.zeros(self.number_of_output_neurons)
        
    def feedforward(self, input_array):
        self.input_nn = np.array(input_array, dtype="float64")
        self.output_of_input_layer_nn = self.input_nn
        self.output_of_input_layer_mn = (self.alpha_input_layer * self.prev_output_of_input_layer_nn) + ((1.0 - self.alpha_input_layer) * self.prev_output_of_input_layer_mn)
        
        self.input_to_hidden_layer_nn = np.matmul(self.weights_input_to_hidden_nn.transpose(), self.output_of_input_layer_nn) + np.matmul(self.weights_input_to_hidden_mn.transpose(), self.output_of_input_layer_mn)
        self.output_of_hidden_layer_nn = self.activation_function(self.input_to_hidden_layer_nn)
        self.output_of_hidden_layer_mn = (self.alpha_hidden_layer * self.prev_output_of_hidden_layer_nn) + ((1.0 - self.alpha_hidden_layer) * self.prev_output_of_hidden_layer_mn)
        
        self.output_of_last_layer_mn = (self.alpha_last_layer * self.prev_output_of_nn) + ((1.0 - self.alpha_last_layer) * self.prev_output_of_mn)
        self.input_to_last_layer_nn = np.matmul(self.weights_hidden_to_output_nn.transpose(), self.output_of_hidden_layer_nn) + np.matmul(self.weights_hidden_to_output_mn.transpose(), self.output_of_hidden_layer_mn) + (self.beta * self.output_of_last_layer_mn)
        self.output_nn = self.output_layer_activation_function(self.input_to_last_layer_nn)
        
        self.prev_output_of_input_layer_nn = copy.deepcopy(self.output_of_input_layer_nn)
        self.prev_output_of_input_layer_mn = copy.deepcopy(self.output_of_input_layer_mn)
        self.prev_output_of_hidden_layer_nn = copy.deepcopy(self.output_of_hidden_layer_nn)
        self.prev_output_of_hidden_layer_mn = copy.deepcopy(self.output_of_hidden_layer_mn)
        self.prev_output_of_nn = copy.deepcopy(self.output_nn)
        self.prev_output_of_mn = copy.deepcopy(self.output_of_last_layer_mn)

    def backprop(self, y_des):
        self.y_des = copy.deepcopy(y_des)
        
        #Calculate squared error loss
        self.squared_error = np.sum((self.output_nn - self.y_des) ** 2)
        self.error_last_layer = (self.output_nn - self.y_des) * self.output_layer_activation_function_derivative(self.input_to_last_layer_nn)
        self.error_hidden_layer = self.activation_function_derivative(self.input_to_hidden_layer_nn) * np.matmul(self.weights_hidden_to_output_nn, self.error_last_layer)
        
        #Update Weights of network
        self.weights_hidden_to_output_nn -= self.neeta * np.array([self.error_last_layer,] * self.number_of_hidden_neurons) * np.array([self.output_of_hidden_layer_nn,] * self.number_of_output_neurons).transpose()
        self.weights_input_to_hidden_nn -= self.neeta * np.array([self.error_hidden_layer,]*self.number_of_input_neurons) * (np.array([self.output_of_input_layer_nn,]*self.number_of_hidden_neurons).transpose())
        self.weights_hidden_to_output_mn -= self.neeta * np.array([self.error_last_layer,] * self.number_of_hidden_neurons) * np.array([self.output_of_hidden_layer_mn,] * self.number_of_output_neurons).transpose()
        self.weights_input_to_hidden_mn -= self.neeta * np.array([self.error_hidden_layer,]*self.number_of_input_neurons) * (np.array([self.output_of_input_layer_mn,]*self.number_of_hidden_neurons).transpose())
        self.beta -= self.neeta_dash * self.error_last_layer * self.output_of_last_layer_mn
        
        #pd means partial derivative
        self.pd_e_wrt_v_hidden_layer = np.matmul(self.weights_hidden_to_output_mn, self.error_last_layer)
        self.pd_e_wrt_v_input_layer = np.matmul(self.weights_input_to_hidden_mn, self.error_hidden_layer)
        self.pd_e_wrt_v_last_layer = self.beta * self.error_last_layer
        self.pd_v_wrt_alpha_hidden_layer = self.prev_output_of_hidden_layer_nn - self.prev_output_of_hidden_layer_mn
        self.pd_v_wrt_alpha_input_layer = self.prev_output_of_input_layer_nn - self.prev_output_of_input_layer_mn
        self.pd_v_wrt_alpha_last_layer = self.prev_output_of_nn - self.prev_output_of_mn
        
        #Update memory coefficients
        self.alpha_hidden_layer -= self.neeta_dash * self.pd_e_wrt_v_hidden_layer * self.pd_v_wrt_alpha_hidden_layer
        self.alpha_input_layer -= self.neeta_dash * self.pd_e_wrt_v_input_layer * self.pd_v_wrt_alpha_input_layer
        self.alpha_last_layer -= self.neeta_dash * self.pd_e_wrt_v_last_layer * self.pd_v_wrt_alpha_last_layer

        #Constrain coefficients between 0 and 1 to ensure stability of network
        self.alpha_hidden_layer = np.clip(self.alpha_hidden_layer, 0.0, 1.0)
        self.alpha_input_layer = np.clip(self.alpha_input_layer, 0.0, 1.0)
        self.alpha_last_layer = np.clip(self.alpha_last_layer, 0.0, 1.0)
        self.beta = np.clip(self.beta, 0.0, 1.0)

        if(self.spectral_norm):
        
            # print("Spectral Norm:")
            # print(np.linalg.norm(self.weights_hidden_to_output_mn, ord=2))
            # print("\n")
            self.weights_input_to_hidden_mn = (self.weights_input_to_hidden_mn / np.linalg.norm(self.weights_input_to_hidden_mn, ord=2)) * (self.lipschitz ** (1/2) )
            self.weights_input_to_hidden_nn = (self.weights_input_to_hidden_nn / np.linalg.norm(self.weights_input_to_hidden_nn, ord=2)) * (self.lipschitz ** (1/2) )
            self.weights_hidden_to_output_mn = (self.weights_hidden_to_output_mn / np.linalg.norm(self.weights_hidden_to_output_mn, ord=2)) * (self.lipschitz ** (1/2) )
            self.weights_hidden_to_output_nn = (self.weights_hidden_to_output_nn / np.linalg.norm(self.weights_hidden_to_output_nn, ord=2)) * (self.lipschitz ** (1/2) )
            self.beta = (self.beta / np.linalg.norm(self.beta, ord=2)) * (self.lipschitz ** (1/2) )

    def activation_function(self, x):
        #g1_x = 1.0/(1.0 + np.exp(-1.0 * x))
        g1_x = 15 * np.tanh(x / 15)
        return g1_x

    def output_layer_activation_function(self, x):
        #g2_x = (1.0 - np.exp(-1.0 * x))/(1.0 + np.exp(-1.0 * x))
        return x
        #return g2_x

    def activation_function_derivative(self, x):
        #g1_dash_x = np.exp(-1.0 *x)/((1.0 + np.exp(-1.0 * x))**2.0)
        return 1.0 - np.square(np.tanh(x / 15))
        #return g1_dash_x
        
    def output_layer_activation_function_derivative(self, x):
        #g2_dash_x = (2.0 * np.exp(-1.0 * x))/((1.0 + np.exp(-1.0 * x))**2.0)
        return 1.0
        #return g2_dash_x

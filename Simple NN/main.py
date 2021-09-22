import numpy as np
import scipy.special

# neural network class definition
class neuralNetwork:

    # initialize the NN
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        
        # set number of nodes in each layer
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        
        # set learning rate
        self.lr = learning_rate
        
        # NN weights
        # weights matrices are w_i_j, from node i to node j 
        self.wih = np.random.normal(0.0, pow(self.hnodes,-0.5), (self.hnodes,self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes,-0.5), (self.onodes,self.hnodes))
        
        # activation function (sigmoid)
        self.activation_func = lambda x: scipy.special.expit(x)
        
        pass
        
    # train the NN
    def train(self, inputs_list, targets_list):
    
        # convert input list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        # calcs signals associated with hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)
        
        # calcs signals associated with final layer
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)
        
        # finding error with back prop
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        
        # update weights
        self.who += self.lr*np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lr*np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
    
        pass
        
    # query the NN
    def query(self, inputs_list):
    
        # convert input list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        
        # calcs signals associated with hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)
        
        # calcs signals associated with final layer
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)
        
        return final_outputs

# parameters for NN
input_nodes = 3
hidden_nodes = 3
output_nodes = 3
learning_rate = 0.3

# create NN
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


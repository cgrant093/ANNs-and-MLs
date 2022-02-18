import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        
        # NN weights matrices are w_i_j, from node i to node j 
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
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3

# create NN
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)


# load training and testing data
training_data_file = open("mnist_dataset/mnist_train.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()

test_data_file = open("mnist_dataset/mnist_test.csv", "r")
test_data_list = test_data_file.readlines()
test_data_file.close()

# plot one example of the training data
#all_values = data_list[0].split(',')
#image_array = np.asfarray(all_values[1:]).reshape((28,28))
#plt.imshow(image_array, cmap='Greys', interpolation='None')
#plt.show()


print("\nNeural Network training progress:")

# training the NN 
for record in tqdm(training_data_list):
    # split the record by the ',' commas
    all_values = record.split(',')
    
    # scale and shift the inputs
    inputs = (np.asfarray(all_values[1:])/255.0 * 0.99) + 0.01 
    
    #creates the target output values (also scaled)
    targets = np.zeros(output_nodes) + 0.01
    
    #all_vaules[0] is the target label for this record
    targets[int(all_values[0])] = 0.99
    
    #train the network
    n.train(inputs, targets)
    
    pass
    
    
# testing the NN

# get the first test record, print and plot it
all_values = test_data_list[0].split(',')
print("\nTest value is: ", all_values[0])

#image_array = np.asfarray(all_values[1:]).reshape((28,28))
#plt.imshow(image_array, cmap='Greys', interpolation='None')
#plt.show()

#query the NN
test = n.query((np.asfarray(all_values[1:])/255.0 * 0.99) + 0.01)
print("Trained Neural Network output is:\n", np.round(test, 3))    





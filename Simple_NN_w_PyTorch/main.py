import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

data_folder = 'C:/Users/cgran/Documents/Github/ANNs-and-ML/Simple_NN_w_PyTorch/'

"""
Neural network class

Sequential: a simple stack of layers
    Each layer has only one input tensor and one output tensor
Linear: applies a linear transformation to the incoming data
ReLU: applies the rectified linear unit function element-wise
"""
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

"""
Neural network class

Conv2d: applies 2D convolution over an input signal
Dropout2d/dropout: Randomly zeros entire channels. Default p=0.5
Linear: applies a linear transformation to the incoming data

relu/ReLU: applies the rectified linear unit function element-wise
max_pool2D: applies a 2D max pooling over an input signal 
view: returns new tensor with same data but of different shape
log_softmax: applies a softmax followed by a logarithm: log(softmax(x))
"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # 10 input channels, 20 output channels, 5x5 square convolution
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        # affine operation: y = W.x + b
            # first input should usually be 20*5*5 = 500
            # but with dropout we can reduce that number?
                # default p=0.5, but 302/500 = 0.64
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)



# training parameters
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# load in input data
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(data_folder, train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ])
                            ),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(data_folder, train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                ])
                            ),
    batch_size=batch_size_test, shuffle=True)
  
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)] 



# create network
net = Net()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)


# training algorithm    
def train(epoch):
    net.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = net(data)
        
        loss = F.nll_loss(output, target)
        loss.backward()
        
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
                
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
                
            torch.save(net.state_dict(), 'results/model.pth')
            torch.save(optimizer.state_dict(), 'results/optimizer.pth')

# testing algorithm
def test():
    net.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
          output = net(data)
          test_loss += F.nll_loss(output, target, size_average=False).item()
          pred = output.data.max(1, keepdim=True)[1]
          correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



# train the network
test()
for epoch in range(1, n_epochs+1):
    train(epoch)
    test()

# print training curve
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()






import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class BidirectionalLSTM(nn.Module):
    """bidirectional LSTM RNN class"""
    
    def __init__(self, nInput, nHidden, nOutput):
        # nX = number of "X" nodes
        super(BidirectionalLSTM, self).__init__()
        
        self.rnn = nn.LSTM(nInput, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden*2, nOutput)

    def forward(self, input):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size() 
            # T = sequence length? why is it called T? time?
            # b = batch size
            # h = 2 * hOutput     # 2 b/c bidirectional
        t_rec = recurrent.view(T*b, h)
        output = self.embedding(t_rec)  # [T*b, nOutput]
        return output.view(T, b, -1)
  


  
class CRNN(nn.Module):
    """The convolution layers + the recurrent layers"""
    
    def __init__(self, args, leakyRelu=False):
        super(CRNN, self).__init__()
        
        assert args['imgH'] % 16 == 0, 'imgH has to be a multiple of 16'
        
        kernel  = [3, 3, 3, 3, 3, 3, 2]                 # kernel sizes
        padding = [1, 1, 1, 1, 1, 1, 0]                 # pool sizes
        stride  = [1, 1, 1, 1, 1, 1, 1]                 # stride sizes
        feature = [64, 128, 256, 256, 512, 512, 512]    # feature sizes
        
        # setup 7 layer CNN
        cnn = nn.Sequential()
        
        def convRelu(i, batchNormalization=False):
            nInput = args['nChannels'] if i ==0 else feature[i-1]
            nOutput = feature[i]
            cnn.add_module('conv{0}'.format(i),
                            nn.Conv2d(nInput, nOutput, kernel[i], stride[i], padding[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOutput))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
        
        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)   # 128x4x32?
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2), 
                        nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)   # 256x2x16?
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3), 
                        nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(2, True)   # 512x1x16
        
        self.cnn = cnn
        
        # setup 2 layer bidirectional LSTM RNN
        self.rnn = nn.Sequential()
        self.rnn = nn.Sequential(
            BidirectionalLSTM(args['nHidden']*2, args['nHidden'], args['nHidden']),
            BidirectionalLSTM(args['nHidden'], args['nHidden'], args['nClasses']))
    
    def forward(self, input):
        # CNN features
        conv = self.cnn(input)
        b, c, h, w = conv.size() 
            # b = batch size
            # c = output "channel" nodes
            # h = height of input planes in pixels
            # w = width of pixels
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) #squeeze dimension = 2 (h dim should disappear)
        conv = conv.permute(2, 0, 1) # [w, b, c]
        
        # RNN features
        output = self.rnn(conv)
        return output.transpose(1, 0) # Tbh -> bTh
 


 
class CustomCTCLoss(nn.Module):
    """Custom CTC loss algorithm"""
    """T x b x h -> Softmas on dimension 2"""
    def __init__(self, dim=2):
        super(CustomCTCLoss, self).__init__()
        self.dim = dim
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
    
    def sanitize(self, loss):
        EPS = 1e-7 
        
        if abs(loss.item() - float('inf')) < EPS:
            return torch.zeros_like(loss)
        
        if np.isnan(loss.item()):
            return torch.zeros_like(loss)
        
        return loss
    
    def debug(self, loss, logits, labels, prediction_sizes, target_sizes):
        if np.isnan(loss.item()):
            print("loss:", loss)
            print("logits:", logits)
            print("labels:", labels)
            print("prediction_sizes:", prediction_sizes)
            print("target_sizes:", target_sizes)
            raise Exception("NaN loss obtained. But why?")
        
        return loss
    
    def forward(self, logits, labels, prediction_sizes, target_sizes):
        loss = self.ctc_loss(logits, labels, prediction_sizes, target_sizes)
        loss = self.sanitize(loss)
        return self.debug(loss, logits, labels, prediction_sizes, target_sizes)
    
    
    

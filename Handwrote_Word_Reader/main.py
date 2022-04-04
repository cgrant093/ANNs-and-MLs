import os
import numpy as np

import torch
from torch.utils.data import random_split

from model import CRNN, CustomCTCLoss
from preprocessing import SynthCollator, SynthDataset, readCharFile, readArgsFile
from OCR import OCRTrainer
from learner import Learner


def gmkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# read in 79 characters (80 if you count "blank")
charList = readCharFile('model/charList.txt')

# set initial args/options dictionary
args = readArgsFile('model/argsList.txt')
args['nClasses'] = len(charList)+1
args['charList'] = charList
'''
# preprocess data and add things to args dictionary
data = SynthDataset(args)
args['collate_fn'] = SynthCollator()

training_split = int(0.8 * len(data))
validation_split = len(data) - training_split
args['data_train'], args['data_val'] = random_split(data, (training_split, validation_split))

print('Training Data Size:{}\nVal Data Size:{}'.format(
    len(args['data_train']), len(args['data_val'])))
'''
# create and save model 
model = CRNN(args)
'''
args['criterion'] = CustomCTCLoss()

savepath = os.path.join(args['save_dir'], args['name'])
gmkdir(savepath)
gmkdir(args['log_dir'])

optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
learner = Learner(model, optimizer, savepath=savepath, resume=args['resume'])
learner.fit(args)'''







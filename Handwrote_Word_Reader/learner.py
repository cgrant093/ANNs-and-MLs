import os
import logging
import numpy as np

import torch
import torch.nn as nn

from OCR import OCRTrainer


class EarlyStopping:
    '''Early stops the training if validation loss doesn't improve after a given patience.'''
    
    def __init__(self, save_file, patience=5, verbose=False, delta=0, best_score=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_file = save_file
        
        print(best_score)
    
    def __call__(self, val_loss, epoch, model, optimizer):
        score = -val_loss
        state = {
            'epoch': epoch+1
            'state_dict': model.state_dict(),
            'opt_state_dict': optimizer.state_dict(),
            'best': score
            }
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, state)
        
        elif score < (self.best_score - self.delta):
            self.counter += 1
            print(f'EarlyStopping counter: ({self.best_score:.6f} {self.counter} out of {self.patience})')
            
            if self.counter >= self.patience:
                self.early_stop = True
        
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, state)
            self.counter = 0
    
    
    def save_checkpoint(self, val_loss, state):
        '''Saves model when validation loss decrease.'''
        
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        torch.save(state, self.save_file)
        self.val_loss_min = val_loss

class Learner(object):
    def __init__(self, model, optimizer, savepath=None, resume=False):
        self.model = model
        self.optimizer = optimizer
        self.savepath = os.path.join(savepath, 'best.ckpt')
        self.cuda = torch.cuda.is_available()
        self.cuda_count = torch.cuda.device_count()
        
        if self.cuda:
            self.model = self.model.cuda()
        
        self.epoch = 0
        
        if self.cuda_count > 1:
            print("Let's use", self.cuda_count, "GPUs!")
            self.modell = nn.DataParallel(self.model)
        
        self.best_score = None
        
        if resume and os.path.exists(self.savepath):
            self.checkpoint = torch.load(self.savepath)
            self.epoch = self.checkpoint['epoch']
            self.best_score = self.checkpoint['best']
            self.load()
        
        else:
            print('checkpoint does not exists')
    
    def fit(self, args):
        args['cuda'] = self.cuda
        args['model'] = self.model
        args['optimizer'] = self.optimizer
        
        logging.basicConfig(filename="%s/%s.csv" %(args['log_dir'], args['name']), level=logging.INFO)
        self.saver = EarlyStopping(self.savepath, patience=15, verbose=True, best_score=self.best_score)
        args['epoch'] = self.epoch
        
        trainer = OCRTrainer(args)
        
        for epoch in range(args['epoch'], args['epochs']):
            train_result = trainer.run_epoch()
            val_result = trainer.run_epoch(validation=True)
            
            trainer.count = epoch
            
            info = '%d, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f'%(epoch, 
                        train_result['train_loss'], val_result['val_loss'], 
                        train_result['train_ca'],  val_result['val_ca'],
                        train_result['train_wa'],  val_result['val_wa'])
            
            logging.info(info)
            
            self.val_loss = val_result['val_loss']
            print(self.val_loss)
            
            if self.savepath:
                self.save(epoch)
            
            if self.saver.early_stop:
                print('Early stopping')
                break
    
    def load(self):
        print('Loading checkpoint at {} trained for {} epochs'.format(self.savepath, self.checkpoint['epoch']))
        
        self.model.load_state_dict(self.checkpoint['state_dict'])
        
        if 'opt_state_dict' in self.checkpoint.keys():
            print('Loading optimizer')
            self.optimizer.load_state_dict(self.checkpoint['opt_state_dict'])
    
    def save(self, epoch):
        self.saver(self.val_loss, epoch, self.model, self.optimizer)
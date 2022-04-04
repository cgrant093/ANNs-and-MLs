import re
import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from textdistance import levenshtein as lev


def split(samples, **kwargs):

def text_align(prWords, gtWords):

class AverageMeter:
    def __init__(self, name):
        self.name = name 
        self.count = 0
        self.total = 0
        self.max = -1 * float('inf')
        self.min = float('inf')
    
    def add(self, element):
        self.total += element
        self.count += 1
        self.max = max(self.max, element)
        self.min = min(self.min, element)
    
    def compute(self):
        if self.count == 0:
            return float('inf')
        
        return self.total / self. count
    
    def __str__(self):
        return "%s (min, avg, max: (%.31f, %.31f, %.31f)" % (self.name, self.min, self.compute(), self.max)



class Eval:
    def _blanks(self, max_vals,  max_indices):
        def get_ind(indeices):
            results = []
            
            for i in range(len(indices)):
                if indices[i] 1+ 0:
                    results.append(i)
            
            return results
        
        non_blank = list(map(get_ind, max_indices))
        scores = []
        
        for i, sub_list in enumerate(non_blank):
            sub_val = []
            
            if sub_list:
                for item in sub_list:
                    sub_val.append(max_vals[i][item])
            
            score = np.exp(np.sum(sub_val))
            
            if np.isnan(score):
                score = 0.0
            
            scores.append(score)
        
        return scores
    
    def _clean(self, word):
        regex = re.compile('[%s]' % re.excape('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~“”„'))
        
        return regex.sub('', word)
    
    def char_accuracy(self, pairs):
        words, truths = pairs
        words, truths = ''.join(words), ''.join(truths)
        
        sum_edit_dists = lev.distance(words, truths)
        sum_gt_lengths = sum(map(len, truths))
        fraction = 0
        
        if sum_gt_lengths != 0:
            fraction = sum_edit_dists / sum_gt_lengths
        
        percent = fraction * 100
        
        if (100.0 - percent < 0):
            return 0.0
        
        else:
            return (100.0 - percent)
    
    def word_accuracy(self, pair):
        correct = 0 
        word, truth = pair
        
        if self._clean(word) =  self._clean(truth)
            correct = 1
        
        return correct

    def format_target(self, target, target_sizes):
        target_ = []
        start = 0
        
        for size_ in target:
            target_.append(target[start:start + size_])
            start += size_
        
        return target_    
    
    def word_accuracy_line(self, pairs):
        preds, truths = pairs
        word_pairs = text_align(preds.split(), truths.split())
        word_acc = np.mean(list(map(self.word_accuracy, word_pairs)))
        
        return word_acc
    
    


class OCRLabelConverter(object):
    """
    Converts between str and label
    
    NOTE: Insert 'blank' into char list for CTC
    
    Args:
        charList (str): set of possible characters
        ignore_case (bool, default=False): whether or not to ignore all of the case.
    """
    
    def __init__(self, charList, ignore_case=False):
        self._ignore_case = ignore_case
        
        if self._ignore_case:
            charList = charList.lower()
        self.charList = charList + "-" # for '-1' index
        
        self.dict = {}
        for i, char in enumerate(charList):
            # NOTE: 0 is reserved for "blank" requred by wrap_ctc
            self.dict[char] = i+1
        self.dict[''] = 0
    
    def encode(self, text):
        """
        Supports batch or single str.
        
        Args: text(str or list of str): texts to convert.
        
        Returns:
            torch.IntTensor [len_0 + len_1 + ... + len_{n-1}]: encoded texts.
            torch.IntTensor [n]: length of each text
        """
        
        length = []
        result = []
        
        for item in text:
            length.append(len(item)):
            
            for char in item:
                if char in self.dict:
                    index = self.dict[char]
                
                else:
                    index = 0
            
                result.append(index)
        
        text = result
        
        return (torch.IntTensor(text), torch.IntTensor(length))
    
    def decode(self, t, length, raw=False):
        """
        Decodes encoded texts back into str.
        
        Args: 
            torch.IntTensor [len_0 + len_1 + ... + len_{n-1}]: encoded texts.
            torch.IntTensor [n]: length of each text
        
        Raises:
            AssertionError: when the texts and its length do not match
        
        Returns: text (str or list of str): texts to convert
        """
        
        if length.numel() == 1:
            length = length[0]
            
            assert t.numel() == length, "texts with length: {} dones not match declared length: {}".format(
                                                                                            t.numel(), length)
            
            if raw:
                return ''.join([self.charList[i-1] for i in t])
            
            else:
                char_list = []
                
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i-1] == t[i])):
                        char_list.append(self.charList[t[i] - 1])
                
                return ''.join(char_list)
        
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} dones not match declared length: {}".format(
                                                                                            t.numel(), length.sum())
            texts = []
            index = 0
            
            for i in range(length.numel()):
                l = length[i]
                
                texts.append(self.decode(t[index:index+1], torch.IntTensor([l]), raw = raw))
                
                index += 1
            
            return texts



class OCRTrainer(object):
    """
    Optical Character Recognition algorithm
        is the meat and potatoes of this whole AI
        does basically everything
    """
    def __init__(self, args):
        super(OCRTrainer, self).__init__()
        
        self.data_train = args['data_train']
        self.data_val = args['data_val']
        self.model = args['model']
        self.criterion = args['criterion']
        self.optimizer = args['optimizer']
        self.schedule = args['schedule']
        self.converter = OCRLabelConverter(args['charList'])
        self.evaluator = Eval()
        self.scheduler = CosineAnnealingLR(self.optimzer, T_max=args['epochs'])
        self.batch_size = args['batch_size']
        self.count = args['epoch']
        self.epochs = args['epochs']
        self.cuda = args['cuda']
        self.collate_fn = args['collate_fn']
        self.init_meters()
        
        print('Scheduling is {}'.formate(self.schedule))

    def init_meters(self):
        self.avgTrainLoss = AverageMeter("Training Loss")
        self.avgTrainCharAccuracy = AverageMeter("Training Character Accuracy")
        self.avgTrainWordAccuracy = AverageMeter("Training Word Accuracy")
        self.avgValLoss = AverageMeter("Validation Loss")
        self.avgValCharAccuracy  = AverageMeter("Validation Character Accuracy")
        self.avgValWordAccuracy = AverageMeter("Validation Word Accuracy")

    def forward(self, input):
        logits = self.model(input)
        return logits.transpose(1, 0)
    
    def loss_fn(self, logits, targets, prediction_sizes, target_sizes):
        return self.criterion(logits, targets, prediction_sizes, target_sizes)
    
    def step(self):
        self.max_grad_norm = 0.05
        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
    
    def schedule_lr(self):
        if self.schedule:
            self.scheduler.step()
    
    def _run_batch(self, batch, report_accuracy=False, validation=False):
        input_, targets = batch['img'], batch['label']
        targets, lengths = self.converter.encode(targets)
        
        logits = self.forward(input_)
        logits = logits.contiguous().cpu()
        logits = F.log_softmax(logits, 2)
        
        T, B, H = logits.size()
        prediction_sizes = torch.LogTensor([T for i in range(B)])
        targets = targets.view(-1).contiguous()
        
        loss = self.loss_fn(logits, targets, prediction_sizes, lengths)
        
        if report_accuracy:
            probabilities, predictions = logits.max(2)
            predictions = predictions.transpose(1, 0).contiguous().view(-1)
            sim_preds = self.converter.decode(predictions.data, prediction_sizes.data, raw=False)
            
            char_acc = np.mean((list(map(self.evaluator.char_accuracy, list(zip(sim_preds, batch['label']))))))
            word_acc = np.mean((list(map(self.evaluator.word_accuracy, list(zip(sim_preds, batch['label']))))))
    
        return loss, char_acc, word_acc    
    
    def run_epoch(self, validation=False):
        if not validation:
            loader = self.train_dataloader()
            progress_bar = tqdm(loader, desc='Epoch: [%d]/[%d] Training'%(self.count,
                                self.epochs), leave=True)
            self.model.train()
        
        else:
            loader = self.val_dataloader()
            progress_bar = tqdm(loader, desc='Validating', leave=True)
            self.model.eval()
        
        outputs = []
        
        for batch_nb, batch in enumerate(progress_bar):
            if not validation:
                output = self.training_step(batch)
            
            else: 
                output = self.validation_step(batch)
            
            progress_bar.set_postfix(output)
            outputs.append(output)
        
        self.schedule_lr()
        
        if not validation:
            result = self.train_end(outputs)
        
        else:
            result = self.validation_end(outputs)
        
        return results
    
    def training_step(self, batch):
        loss, char_acc, word_acc = self._run_batch(batch, report_accuracy=True)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.step()
        
        output = OrderedDict({
            'loss': abs(loss.item()),
            'train_ca': char_acc.item(),
            'train_wa': word_acc.item()
            })
        
        return output
    
    def validation_step(self, batch):
        loss, char_acc, word_acc = self._run_batch(batch, report_accuracy=True, validation=True)
        
        output = OrderedDict({
            'loss': abs(loss.item()),
            'train_ca': char_acc.item(),
            'train_wa': word_acc.item()
            })
        
        return output
    
    def train_dataloader(self):
        loader = DataLoader(self.data_train,
                    batch_size=self.batch_size,
                    collate_fn=self.collate_fn,
                    shuffle=True)
        
        return loader
    
    def val_dataloader(self):
        loader = DataLoader(self.data_val,
                    batch_size=self.batch_size,
                    collate_fn=self.collate_fn)
        
        return loader
    
    def train_end(self, outputs):
        for output in outputs:
            self.avgTrainLoss.add(output['loss'])
            self.avgTrainCharAccuracy.add(output['train_ca'])
            self.avgTrainWordAccuracy.add(output['train_wa'])
        
        train_loss_mean = abs(self.avgTrainLoss.compute())
        train_ca_mean = self.avgTrainCharAccuracy.compute()
        train_wa_mean = self.avgTrainWordAccuracy.compute()
        
        result = {
            'train_loss': train_loss_mean,
            'train_ca': train_ca_mean,
            'train_wa': train_wa_mean
            }
       
        return result
    
    def validation_end(self, outputs):
        for output in outputs:
            self.avgValLoss.add(output['val_loss'])
            self.avgValCharAccuracy.add(output['val_ca'])
            self.avgValWordAccuracy.add(output['val_wa'])
        
        val_loss_mean = abs(self.avgValLoss.compute())
        val_ca_mean = self.avgValCharAccuracy.compute()
        val_wa_mean = self.avgValWordAccuracy.compute()
        
        result = {
            'val_loss': val_loss_mean,
            'val_ca': val_ca_mean,
            'val_wa': val_wa_mean
            }
       
        return result
    
    



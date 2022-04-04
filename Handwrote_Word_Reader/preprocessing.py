import os

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def readCharFile(fileName):
    """read in the alphabet from a file"""
    with open(fileName) as f:
        for line in f:
            alphabet = line
    
    return alphabet



def readArgsFile(fileName):
    """read in the arguments/options from a file"""
    args = {}
    with open(fileName) as f:
        for line in f:
            (k, v) = line.split(" : ")
            v = v.partition("\n")[0]
            
            if v.isdigit():
                v = int(v)
            
            elif (v.partition(".")[1] == "."):
                v = float(v)
            
            args[k] = v
    
    return args



class SynthCollator(object):
    """
    Sets all images in batch to same size
        Height is autoset to 32 in args
        Finds images with largest width and sets all images to that width
    """
    
    def __call__(self, batch):
        width = [item['img'].shape[2] for item in batch]
        indexes = [item['idx'] for item in batch]
        imgs = torch.ones([len(batch), batch[0]['img'].shape[0], batch[0]['img'].shape[1], 
                           max(width)], dtype=torch.float32)
        
        for idx, item in enumerate(batch):
            try:
                imgs[idx, :, :, 0:item['img'].shape[2]] = item['img']
            except:
                print(imgs.shape)
        
        item = {'img': imgs, 'idx':indexes}
        
        if 'label' in batch[0].keys():
            labels = [item['label'] for item in batch]
            item['label'] = labels
        
        return item



class SynthDataset(Dataset):
    """
    Convert each image to greyscale and a tensor. 
    Then normalize to values [-1, 1]
    """
    
    def __init__(self, args):
        super(SynthDataset, self).__init__()
        
        self.path = os.path.join(args['path'], args['imgdir'])
        self.images = os.listdir(self.path)
        self.nSamples = len(self.images)
        
        f = lambda x: os.path.join(self.path, x)
        self.imagepaths = list(map(f, self.images))
        
        transform_list =  [transforms.Grayscale(1),
                            transforms.ToTensor(), 
                            transforms.Normalize((0.5,), (0.5,))]
        self.transform = transforms.Compose(transform_list)
        
        #self.collate_fn = SynthCollator()
    
    
    def __len__(self):
        return self.nSamples
    
    
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        
        imagepath = self.imagepaths[index]
        imagefile = os.path.basename(imagepath)
        img = Image.open(imagepath)
        
        if self.transform is not None:
            img = self.transform(img)
        
        item = {'img': img, 'idx': index}
        item['label'] = imagefile.split("_")[0]
        
        return item
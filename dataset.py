import os
import random
import json
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

class PARTNET(Dataset):
    def __init__(self, split='train'):
        super(PARTNET, self).__init__()
        
        assert split in ['train', 'val', 'test']
        self.split = split
        self.root_dir = "/content/CLEVR"     
        self.files = os.listdir(os.path.join(self.root_dir, self.split))
        self.img_transform = transforms.Compose([
               transforms.ToTensor()])

    def __getitem__(self, index):
        path = self.files[index]
        text_file = os.path.join(self.root_dir, self.split, path)

        with open(text_file) as f:
             line = f.readlines()[0]
        sample = {'text': line}
        return sample
            
    
    def __len__(self):
        return len(self.files)

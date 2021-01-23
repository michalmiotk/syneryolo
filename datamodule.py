import pytorch_lightning as pl
import torchvision
from my_transforms import Transform_img_labels
import torch
from torch.utils.data import DataLoader
class DataModule(pl.LightningDataModule):
    def __init__(self, bs):
        super().__init__()
        self.batch_size = bs
        self.transform_VOC = Transform_img_labels()
        self.year='2007'
        self.root_dir_train = "/home/m/Pobrane/VOC"+self.year+"_train"
        self.root_dir_val = "/home/m/Pobrane/VOC"+self.year+"_val"
        self.root_dir_test = "/home/m/Pobrane/VOC"+self.year+"_test"
        
    def my_collate(self, batch):
        data = [item[0].unsqueeze(0) for item in batch]
        data = torch.cat(data)
        target = [item[1] for item in batch]
        return [data, target]
    
    def setup(self, stage=None):
        train_dataset = torchvision.datasets.VOCDetection(self.root_dir_train, year=self.year,image_set='train', download=True, transforms=self.transform_VOC)
        length_train, length_val = 100, 10
        self.train_dataset = torch.utils.data.random_split(train_dataset, [length_train, len(train_dataset)-length_train])[0]
        trainval_dataset = torchvision.datasets.VOCDetection(self.root_dir_val, year=self.year,image_set='trainval', download=True, transforms=self.transform_VOC)
        self.trainval_dataset = torch.utils.data.random_split(trainval_dataset, [length_val, len(trainval_dataset)-length_val])[0]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, collate_fn=self.my_collate, num_workers=16, shuffle=True)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.trainval_dataset, batch_size=self.batch_size, collate_fn=self.my_collate, num_workers=16)
    
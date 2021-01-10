import pytorch_lightning as pl
import torchvision
from my_transforms import Transform_img_labels
import torch
class DataModule(pl.LightningDataModule):
    def __init__(self, bs):
        super().__init__()
        self.batch_size = bs
        self.transform_VOC = Transform_img_labels()
        self.root_dir_train = "/home/m/Pobrane/VOC2007train"
        self.root_dir_val = "/home/m/Pobrane/VOC2007val"
        self.root_dir_test = "/home/m/Pobrane/VOC2007test"

    def prepare_data(self):
        self.train_dataset2007 = torchvision.datasets.VOCDetection(self.root_dir_train, year='2007',image_set='train', download=True, transforms=self.transform_VOC)
        self.train_dataset2012 = torchvision.datasets.VOCDetection(self.root_dir_train, year='2012',image_set='train', download=True, transforms=self.transform_VOC)
        self.train_dataset = torch.utils.data.ConcatDataset([self.train_dataset2007,self.train_dataset2012])
        self.trainval_dataset = torchvision.datasets.VOCDetection(self.root_dir_val, year='2007',image_set='trainval', download=True, transforms=self.transform_VOC)
        self.val_dataset = torchvision.datasets.VOCDetection(self.root_dir_test, year='2007', image_set='val', download=True, transforms=self.transform_VOC)
    # Creating train batches
    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=3, pin_memory=True)


    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.trainval_dataset, batch_size=self.batch_size, shuffle=False, num_workers=3, pin_memory=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=3, pin_memory=True)

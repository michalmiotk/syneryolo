import pytorch_lightning as pl
import torchvision
from my_transforms import Transform_img_labels
import torch
class DataModule(pl.LightningDataModule):
    def __init__(self, bs):
        super().__init__()
        self.batch_size = bs
        self.transform_VOC = Transform_img_labels()
        self.year='2008'
        self.root_dir_train = "/home/m/Pobrane/VOC"+self.year+"_train"
        self.root_dir_val = "/home/m/Pobrane/VOC"+self.year+"_val"
        self.root_dir_test = "/home/m/Pobrane/VOC"+self.year+"_test"
        
    def my_collate(self, batch):
        data = [item[0].unsqueeze(0) for item in batch]
        data = torch.cat(data)
        target = [item[1] for item in batch]
        return [data, target]
    
    def setup(self, stage=None):
        self.train_dataset = torchvision.datasets.VOCDetection(self.root_dir_train, year=self.year,image_set='train', download=True, transforms=self.transform_VOC)
        self.trainval_dataset = torchvision.datasets.VOCDetection(self.root_dir_val, year=self.year,image_set='trainval', download=True, transforms=self.transform_VOC)
        self.val_dataset = torchvision.datasets.VOCDetection(self.root_dir_test, year=self.year, image_set='val', download=True, transforms=self.transform_VOC)
    # Creating train batches
    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, collate_fn=self.my_collate, num_workers=16)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.trainval_dataset, batch_size=self.batch_size, collate_fn=self.my_collate, num_workers=16)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, collate_fn=self.my_collate, num_workers=16)

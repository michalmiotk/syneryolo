import torch.nn as nn
import torch
import torchvision.models as models
import torchvision
from my_transforms import Transform_img_labels
from utils import Utils
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import json
import os

with open("api_key.txt", "r") as f:
    api_key = json.load(f)
    print("opened")
os.environ["WANDB_API_KEY"] = api_key["api_key"]


class Yolo(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.vgg16(pretrained=True)
        self.prelast_linear = torch.nn.Linear(512*7*7, 4096)
        self.last_linear = torch.nn.Linear(4096, 7*7*30)
        self.last_relu = torch.nn.ReLU()
        self.lambda_coord = torch.Tensor([0.5]).cuda()
        self.lambda_noobj = torch.Tensor([0.5]).cuda()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.00001)
        return optimizer

    def get_index(self, name):
        return self.cats.index(name)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        if len(y) == 2:
            img, annot = y
        else:
            img = y
            annot = {}
        outputs = self.forward(x)
        loss = self.loss_fn(outputs, annot)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        annot = y
        outputs = self.forward(x)
        loss = self.loss_fn(outputs, annot)
        self.log('val_loss', loss, on_step=True, on_epoch=False)
        return {'loss': loss}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        img, annot = y
        outputs = self.forward(x)
        loss = self.loss_fn(outputs, annot)

        
    
    def loss_fn(self, out, annot):
        loss1 = self.get_loc_error(out, annot)
        loss2 = self.get_w_h_error(out, annot)
        loss3 = self.get_conditional_class_prob_exist(out, annot)
        loss4 = self.get_conditional_class_prob_notexists(out, annot)
        loss5 = self.get_confidence_error(out, annot)
        return loss1 + loss2+loss3+loss4+loss5
    
    def some_iterator(self, predictions):
        predictions_view = predictions.view(-1, 7,7, 30)
        for image_nr in range(predictions_view.shape[0]):
            for y in range(predictions_view.shape[1]):
                for x in range(predictions_view.shape[2]):
                     
                    yield y,x,predictions_view[image_nr, y,x, :]
    


    def get_loc_error(self, predictions, targets):
        
        loss = 0
        for y,x,pred in self.some_iterator(predictions):
            x1,y1,_,_,_,x2,y2 = pred[:7]
            if (y,x) in targets:
                for i, tar in enumerate(targets[(y,x)]):
                   
                    ypred, xpred = tar[2][0], tar[2][1]
                    if i == 0:
                        loss += self.lambda_coord*((xpred-x1)*(xpred-x1)+ (ypred-y1)*(ypred-y1))          
                    elif i == 1:
                        loss += self.lambda_coord*((xpred-x2)*(xpred-x2)+ (ypred-y2)*(ypred-y2))
                    else:
                        break
                    
        return loss
    
    def get_loc_err_equation(self, wtar, wpred, htar, hpred):
        errw = self.lambda_coord*torch.pow(torch.sqrt(wpred)-torch.sqrt(wtar),2)
        errh = self.lambda_coord*torch.pow(torch.sqrt(hpred)-torch.sqrt(htar),2)
        return errw + errh
    
    def get_w_h_error(self, predictions, targets):
        loss = 0
        for y,x,pred in self.some_iterator(predictions):
            _,_,w1,h1,_,_,_, w2,h2 = pred[:9]
            if (y,x) in targets:
                
                for i, tar in enumerate(targets[(y,x)]):
                    wpred, hpred = tar[1]
                    if i == 0:
                        loss += self.get_loc_err_equation(w1, wpred, h1, hpred)      
                    elif i == 1:
                        loss += self.get_loc_err_equation(w2, wpred, h2, hpred)
                    else:
                        break
        return loss

    def get_confidence_error(self, predictions, targets):
        loss = 0
        for y,x,pred in self.some_iterator(predictions):
            x1,y1,w1,h1,p1,x2,y2,w2,h2,p2= pred[:10]
            if (y,x) in targets:
                for i, tar in enumerate(targets[(y,x)]):
                    tarw, tarh = torch.Tensor(tar[1]).cuda()
                    tary, tarx = torch.Tensor(tar[2]).cuda()
                    if i==0:
                        iou = Utils.get_iou(x, y, x1, y1, w1, h1, tarx, tary, tarw, tarh)
                        loss += torch.pow(iou-p1,2)
                    elif i==1:
                        iou = Utils.get_iou(x, y, x2, y2, w2, h2, tarx, tary, tarw, tarh)
                        loss += torch.pow(iou-p2,2)
                    else:
                        break
        
        return loss
                
    def get_conditional_class_prob_exist(self, predictions, targets):
        loss = 0
        for y,x,pred in self.some_iterator(predictions):
            pred_class = pred[10:]
            if (y,x) in targets:
                des_class = targets[(y,x)][0]
                for c in range(20):
                    if c == des_class:
                        loss += torch.pow(torch.Tensor([1]).cuda()-pred_class[c],2)
                    else:
                        loss += torch.pow(torch.Tensor([0]).cuda()-pred_class[c],2)                
        return loss

    def get_conditional_class_prob_notexists(self, predictions, targets):
        loss = 0
        for y,x,pred in self.some_iterator(predictions):
            pred_class = pred[10:]
            if (y,x) not in targets:
                for c in range(20):
                    loss += self.lambda_noobj*torch.pow(torch.Tensor([0]).cuda()-pred_class[c],2)                
        return loss

    def forward_imagenet(self, input):
        x = self.model.features(input)
        x = self.model.avgpool(x)
        
        return x

    def forward(self, input):
        x = self.forward_imagenet(input)
        x = x.view(-1, 512*7*7)
        x = self.prelast_linear(x)
        x = self.last_linear(x)
        x = self.last_relu(x)
        return x
    
class DataModule(pl.LightningDataModule):
    def __init__(self, bs):
        super().__init__()
        self.batch_size = bs
        self.transform_VOC = Transform_img_labels()
        self.root_dir_train = "/home/m/Pobrane/VOC2007train"
        self.root_dir_val = "/home/m/Pobrane/VOC2007val"
        self.root_dir_test = "/home/m/Pobrane/VOC2007test"

    def prepare_data(self):
        self.train_dataset = torchvision.datasets.VOCDetection(self.root_dir_train, year='2007',image_set='train', download=True, transforms=self.transform_VOC)
        self.trainval_dataset = torchvision.datasets.VOCDetection(self.root_dir_val, year='2007',image_set='trainval', download=True, transforms=self.transform_VOC)
        self.val_dataset = torchvision.datasets.VOCDetection(self.root_dir_test, year='2007', image_set='val', download=True, transforms=self.transform_VOC)
    # Creating train batches
    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=3, pin_memory=True)


    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.trainval_dataset, batch_size=self.batch_size, shuffle=False, num_workers=3, pin_memory=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=3, pin_memory=True)

model = Yolo()
dm = DataModule(bs=1)
dm.prepare_data()

wandb_logger = WandbLogger(project="yolo", name='first trials')
trainer = Trainer(max_epochs=11, fast_dev_run=False, gpus=1, profiler=False, progress_bar_refresh_rate=1, logger=wandb_logger, log_every_n_steps=1)
trainer.fit(model, dm)

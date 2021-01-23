import torch.nn as nn
import torch
import torchvision.models as models
import torchvision
from datamodule import DataModule
from iou_loss import Iou
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import json
import os
import cv2
import random 
import numpy as np
from my_transforms import Transform_img_labels
from losses import Losses
from pytorch_lightning.callbacks import LearningRateMonitor
with open("api_key.txt", "r") as f:
    api_key = json.load(f)

os.environ["WANDB_API_KEY"] = api_key["api_key"]


class Yolo(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.S = 14
        self.B = 2
        self.C = 20
        self.resmaxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.model = models.resnet18(pretrained=True)
        self.model.train()
        '''
        for ct, child in enumerate(self.model.children()):
            if ct < 1:
                print(child)
                for param in child.parameters():
                    param.requires_grad = False
        '''
        self.second_layer_size = 256
        self.prelast_linear = torch.nn.Linear(512*self.S*self.S, self.second_layer_size)
        self.prelast_batchnorm = nn.BatchNorm1d(self.second_layer_size)
        self.prelast_activation = torch.nn.LeakyReLU(negative_slope=0.1)
        
        self.last_linear = torch.nn.Linear(self.second_layer_size, self.S*self.S*(self.C+self.B*5))
        self.last_activation= torch.nn.Sigmoid()
        
        self.class_names = Transform_img_labels().class_list
        self.lr =  0.0002
        self.steps_per_epoch = 400
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr,weight_decay=0.0005)
        def func(step: int):
            
            if step < 10*self.steps_per_epoch:
                return 0.0002
            if 85*self.steps_per_epoch > step > 10*self.steps_per_epoch:
                how_many_epochs_since_10 =  (step-10*self.steps_per_epoch)/self.steps_per_epoch
                return 0.001 + how_many_epochs_since_10/75*0.0009
            if 115*self.steps_per_epoch >step > 85*self.steps_per_epoch:
                return 0.001
            if step > 115*self.steps_per_epoch:
                return 0.0001
            
            return 0.0001
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
        return {
            'optimizer': optimizer, 
            'lr_scheduler': {
                'scheduler': scheduler, 
                'interval': 'step'
            }
        }

    def get_index(self, name):
        return self.cats.index(name)
    
    def training_step(self, batch, batch_idx):
        
        if len(batch) == 2:
            img, annot = batch
        else:
            img = batch
            annot = {}

        outputs = self.forward(img)

        loss_loc ,loss_wh,loss_class,loss_confidence = self.loss_fn(outputs, annot)
        loss = loss_loc + loss_wh + loss_class + loss_confidence
        
        self.output_to_img(img, outputs)
        
        
        self.log('train_loss_loc', loss_loc, on_step=True, on_epoch=False)
        self.log('train_loss_wh',loss_wh, on_step=True, on_epoch=False)
        self.log('train_loss', loss_class, on_step=True, on_epoch=False)
        self.log('train_loss_confidence', loss_confidence, on_step=True, on_epoch=False)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        
        x, y = batch
        annot = y
        outputs = self.forward(x)
        loss_loc ,loss_wh,loss_class,loss_confidence = self.loss_fn(outputs, annot)
        return {'val_loss': loss_loc + loss_wh + loss_class + loss_confidence}
        
    def coor_trimer(self, in_coor):
        if in_coor>447:
            return 447
        if in_coor < 0:
            return 0
        return in_coor
    def output_to_img(self, img, outputs):   
        img = img.cpu().detach().numpy()
        out = outputs.cpu().detach().numpy()
        out = out[0]
        out = out.reshape(self.S,self.S,30)
        for y in range(self.S):
            for x in range(self.S):
                x1,y1,w1,h1,p1,x2,y2, w2,h2,p2 = out[y,x, :10]
                class_probs = out[y,x,10:]
                if p1>0.5:
                    predicted_class = np.argmax(class_probs)
                    if p1*class_probs[predicted_class] > 0.5:
                        class_name = self.class_names[predicted_class]
                        xleft,yup = int((x1+x)*64-w1*224), int((y1+y)*64+h1*224)
                        xright,ydown = int((x1+x)*64+w1*224), int((y1+y)*64-h1*224)
                        xleft, yup, xright, ydown=self.coor_trimer(xleft), self.coor_trimer(yup), self.coor_trimer(xright), self.coor_trimer(ydown)
                        img = img[0]
                        
                        if len(img.shape) !=3:
                            return
                        img = img*255
                        img = np.moveaxis(img,0,2)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.rectangle(img, (yup, xleft), (xright, ydown), (255, 255, 0), 1)
                        font = cv2.FONT_HERSHEY_SIMPLEX 
                        org = (int((yup+ydown)/2), int((xleft+xright)/2))
                        fontScale = 1
                        color = (255, 0, 0) 
                        thickness = 2
                        img = cv2.putText(img, class_name, org, font,  fontScale, color, thickness, cv2.LINE_AA)
                        name = str( random.randrange(1,1000))
                        cv2.imwrite("val_out/"+name+".png", img)
    
    def loss_fn(self, out, annot):
        loss_loc, loss_wh, loss_class, loss_confidence = [torch.autograd.Variable(torch.Tensor([0]).cuda())]*4
        
        for i in range(out.shape[0]):
            loss_loc += Losses.get_loc_error(out[i], annot[i])
            loss_wh += Losses.get_w_h_error(out[i], annot[i])
            loss_class += Losses.get_class_error(out[i], annot[i])
            loss_confidence += Losses.get_confidence_error(out[i], annot[i])
        return loss_loc ,loss_wh,loss_class,loss_confidence
        
    def forward_resnet(self, input):
        x = self.model.conv1(input)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.resmaxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4[0](x)
        x = self.model.layer4[1](x)
        return x

    def forward(self, input):
        x = self.forward_resnet(input)
        x = x.view(-1, 512*self.S*self.S)
        print(x.shape)
        x = self.prelast_linear(x)
        x = self.prelast_batchnorm(x)
        x = self.prelast_activation(x)
        x = self.last_linear(x)
        x = self.last_activation(x)
        return x
    

dm = DataModule(bs=2)
model = Yolo()
lr_logger = LearningRateMonitor(logging_interval='step', log_momentum=True)
wandb_logger = WandbLogger(project="yolo", name='maybe 1 good implementation')
trainer = Trainer(gpus=1, max_epochs=1201,  profiler=False, logger=wandb_logger,default_root_dir='checkpoints',progress_bar_refresh_rate=1,log_every_n_steps=1,gradient_clip_val=0.5, callbacks=[lr_logger] )
# Fit model
trainer.fit(model, datamodule=dm)
'''
model = Yolo()
print(model.forward(torch.ones([2,3,448,448])).shape)
'''
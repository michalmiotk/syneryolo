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
with open("api_key.txt", "r") as f:
    api_key = json.load(f)
    print("opened")
os.environ["WANDB_API_KEY"] = api_key["api_key"]


class Yolo(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.vgg16(pretrained=True)
        self.prelast_linear = torch.nn.Linear(512*7*7, 4096)
        self.prelast_activation = torch.nn.LeakyReLU(negative_slope=0.1)
        self.last_linear = torch.nn.Linear(4096, 7*7*30)
        self.last_relu = torch.nn.ReLU()
        self.lambda_coord = torch.Tensor([0.5]).cuda()
        self.lambda_noobj = torch.Tensor([0.5]).cuda()
        self.class_names = Transform_img_labels().class_dict
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer

    def get_index(self, name):
        return self.cats.index(name)
    
    def training_step(self, batch, batch_idx):
      
        if len(batch) == 2:
            img, annot = batch
        else:
            img = batch
            annot = {}
        outputs = self.forward(img)
        loss = self.loss_fn(outputs, annot)
        self.output_to_img(img, outputs)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        annot = y
        outputs = self.forward(x)
        
        loss = self.loss_fn(outputs, annot)
        self.log('val_loss', loss, on_step=True, on_epoch=False)
        
        return {'loss': loss}
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
        out = out.reshape(7,7,30)
        for y in range(7):
            for x in range(7):
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
                        print(img.shape)
                        img = np.moveaxis(img,0,2)
                        
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        print(img.shape, (yup, xleft), (xright, ydown))
                        
                        img = cv2.rectangle(img, (yup, xleft), (xright, ydown), (255, 255, 0), 1)
                        
                        '''
                        puttext
                        '''
                        font = cv2.FONT_HERSHEY_SIMPLEX 
  
                        # org 
                        org = (int((yup+ydown)/2), int((xleft+xright)/2))
                        
                        # fontScale 
                        fontScale = 1
                        
                        # Blue color in BGR 
                        color = (255, 0, 0) 
                        
                        # Line thickness of 2 px 
                        thickness = 2
                        
                        # Using cv2.putText() method 
                        img = cv2.putText(img, class_name, org, font,  fontScale, color, thickness, cv2.LINE_AA) 
                        
                        '''
                        end puttext
                        '''
                        
                        name = str( random.randrange(1,1000))
                        cv2.imwrite("val_out/"+name+".png", img)
                    
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
                    ypred, xpred = tar[2]
                    if i == 0:
                        loss += self.lambda_coord*(torch.pow(x1-xpred,2)+ torch.pow(y1-ypred,2))          
                    elif i == 1:
                        loss += self.lambda_coord*(torch.pow(x2-xpred,2)+ torch.pow(y2-ypred,2))
                    else:
                        break
                    
        return loss
    
    def get_loc_err_equation(self, wpred,wtar, hpred,htar):
        errw = self.lambda_coord*torch.pow(torch.sqrt(wpred)-torch.sqrt(wtar),2)
        errh = self.lambda_coord*torch.pow(torch.sqrt(hpred)-torch.sqrt(htar),2)
        return errw + errh
    
    def get_w_h_error(self, predictions, targets):
        loss = 0
        for y,x,pred in self.some_iterator(predictions):
            _,_,w1pred,h1pred,_,_,_, w2pred,h2pred = pred[:9]
            if (y,x) in targets:
                
                for i, tar in enumerate(targets[(y,x)]):
                    wtar, htar = tar[1]
                    if i == 0:
                        loss += self.get_loc_err_equation(w1pred, wtar, h1pred, htar)      
                    elif i == 1:
                        loss += self.get_loc_err_equation(w2pred, wtar, h2pred, htar)
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
                        iou = Iou.get_iou(x, y, x1, y1, w1, h1, tarx, tary, tarw, tarh)
                        loss += torch.pow(iou-p1,2)
                    elif i==1:
                        iou = Iou.get_iou(x, y, x2, y2, w2, h2, tarx, tary, tarw, tarh)
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
        x = self.prelast_activation(x)
        x = self.last_linear(x)
        x = self.last_relu(x)
        return x
    

model = Yolo()
#model.model.load_state_dict(torch.load("checkpoints/yolo/xxx/checkpoints"))
dm = DataModule(bs=1)
dm.prepare_data()

wandb_logger = WandbLogger(project="yolo", name='first trials')
trainer = Trainer(max_epochs=121, fast_dev_run=False, gpus=1, profiler=False, logger=wandb_logger, progress_bar_refresh_rate=1, log_every_n_steps=1,default_root_dir='checkpoints')#, 
trainer.fit(model, dm)
'''
dm = DataModule(bs=1)
dm.prepare_data()
for x in dm.train_dataset:
    print(len(x))
'''
import torch.nn as nn
import torch
import torchvision.models as models
import torchvision
from my_transforms import Transform_img_labels
from utils import Utils

class Yolo(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.vgg16(pretrained=True)
        self.prelast_linear = torch.nn.Linear(512*7*7, 4096)
        self.last_linear = torch.nn.Linear(4096, 7*7*30)
        self.last_relu = torch.nn.ReLU()
        self.lambda_coord = torch.Tensor([0.5])
        self.lambda_noobj = torch.Tensor([0.5])
        self.cats = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                     'dog', 'horse', 'motorbike', 'person', 'sheep', 'sofa', 'diningtable', 'pottedplant', 'train', 'tvmonitor']

    def get_index(self, name):
        return self.cats.index(name)

    def dataset_inspect(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=0.00001)
        root_dir = "/home/m/Pobrane/VOC2007"
        transformacje=Transform_img_labels()
        self.dataset = torchvision.datasets.VOCDetection(
            root_dir, year="2007", image_set='train', download=True, transforms=transformacje)
        for epoch_nr in range(10):
            for i, x in enumerate(self.dataset):
                image, annot = x
                image = image.unsqueeze(0)
                
                out = self.forward(image)
                loss1 = self.get_loc_error(out, annot)
                loss2 = self.get_w_h_error(out, annot)
                loss3 = self.get_conditional_class_prob_exist(out, annot)
                loss4 = self.get_conditional_class_prob_notexists(out, annot)
                loss5 = self.get_confidence_error(out, annot)
                loss = loss1 + loss2+loss3+loss4+loss5
                self.zero_grad()
                loss.backward()
                optimizer.step()
                if i%100 == 0:
                    print(epoch_nr,i, loss)
        
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
                    wpred, hpred = torch.Tensor([wpred]),torch.Tensor([hpred]) 
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
                    tarw, tarh = torch.Tensor(tar[1])
                    tary, tarx = torch.Tensor(tar[2])
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
                        loss += torch.pow(torch.Tensor([1])-pred_class[c],2)
                    else:
                        loss += torch.pow(torch.Tensor([0])-pred_class[c],2)                
        return loss

    def get_conditional_class_prob_notexists(self, predictions, targets):
        loss = 0
        for y,x,pred in self.some_iterator(predictions):
            pred_class = pred[10:]
            if (y,x) not in targets:
                for c in range(20):
                    loss += self.lambda_noobj*torch.pow(torch.Tensor([0])-pred_class[c],2)                
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


yolo = Yolo()
yolo.dataset_inspect()
#yolo.forward(torch.Tensor(2,3, 224, 224))

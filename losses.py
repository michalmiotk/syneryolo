import torch
import torch.nn as nn
from iou_loss import Iou

class Losses():
    lambda_coord = torch.Tensor([5]).cuda()
    lambda_noobj = torch.Tensor([0.5]).cuda()
    min_stab = torch.Tensor([1e-6]).cuda()
    cell_num = 14

    @staticmethod
    def get_loc_error(predictions, targets):
        loss = torch.autograd.Variable(torch.Tensor([0]).cuda())
        for y,x,pred in Losses.some_iterator(predictions):
            x1,y1, w1,h1,_,x2,y2, w2, h2 = pred[:9]
            if tar:=targets.get((y,x),None):
                ypred, xpred = tar[2]
                if Losses.first_has_highest_iou(pred,tar, x,y):
                    loss += Losses.lambda_coord*(torch.pow(x1-xpred,2)+ torch.pow(y1-ypred,2))
                else:
                    loss += Losses.lambda_coord*(torch.pow(x2-xpred,2)+ torch.pow(y2-ypred,2))      
                                   
        return loss
    
    @staticmethod
    def get_w_h_error(predictions, targets):
        loss = torch.autograd.Variable(torch.Tensor([0]).cuda())
        for y,x,pred in Losses.some_iterator(predictions):
            w1h1pred, w2h2pred =pred[2:4], pred[7:9] 
            if tar := targets.get((y,x),None):
                wtar, htar = tar[1]
                if Losses.first_has_highest_iou(pred,tar, x,y):
                    loss += Losses.get_wh_err_equation(w1h1pred[0], wtar, w1h1pred[1], htar)    
                else:  
                    loss += Losses.get_wh_err_equation(w2h2pred[0], wtar, w2h2pred[1], htar) 
        return loss
    
    @staticmethod
    def get_confidence_error(predictions, targets):
        loss = torch.autograd.Variable(torch.Tensor([0]).cuda())
        for y,x,pred in Losses.some_iterator(predictions):
            p1,p2= pred[4], pred[9]
            if tar := targets.get((y,x),None):
                if Losses.first_has_highest_iou(pred,tar, x,y):
                    loss += torch.pow(torch.Tensor([1]).cuda()-p1,2)
                else:
                    loss += torch.pow(torch.Tensor([1]).cuda()-p2,2)
            else:
                loss += Losses.lambda_noobj*torch.pow(torch.Tensor([0]).cuda()-p1,2)
                loss += Losses.lambda_noobj*torch.pow(torch.Tensor([0]).cuda()-p2,2)
        return loss
    
    @staticmethod            
    def get_class_error(predictions, targets):
        loss = torch.autograd.Variable(torch.Tensor([0]).cuda())
        for y,x,pred in Losses.some_iterator(predictions):
            
            pred_class = pred[10:]
            if tar := targets.get((y,x),None):
                des_class = tar[0]
                for c in range(20):
                    if c == des_class:
                        loss += torch.pow(torch.Tensor([1]).cuda()-pred_class[c],2)
                    else:
                        loss += torch.pow(torch.Tensor([0]).cuda()-pred_class[c],2)                
        return loss
    
    @staticmethod
    def some_iterator(predictions):
        predictions_view = predictions.view(-1, Losses.cell_num,Losses.cell_num, 30)
        for image_nr in range(predictions_view.shape[0]):
            for y in range(predictions_view.shape[1]):
                for x in range(predictions_view.shape[2]):
                    yield y,x,predictions_view[image_nr, y,x, :]
                    
    @staticmethod
    def get_wh_err_equation(wpred,wtar, hpred,htar):
        errw = Losses.lambda_coord*torch.pow(torch.sqrt(wpred+Losses.min_stab)-torch.sqrt(torch.Tensor([wtar]).cuda()+Losses.min_stab),2)
        errh = Losses.lambda_coord*torch.pow(torch.sqrt(hpred+Losses.min_stab)-torch.sqrt(torch.Tensor([htar]).cuda()+Losses.min_stab),2)
        return errw + errh
   
    @staticmethod
    def first_has_highest_iou(pred,tar, x,y):
        x1,y1,w1,h1,p1,x2,y2,w2,h2,p2= pred[:10]
        tarw, tarh = torch.Tensor(tar[1]).cuda()
        tary, tarx = torch.Tensor(tar[2]).cuda()
        iou1 = Iou.get_iou(x, y, x1, y1, w1, h1, tarx, tary, tarw, tarh)
        iou2 = Iou.get_iou(x, y, x2, y2, w2, h2, tarx, tary, tarw, tarh)
        if iou1 > iou2:
            return True
        else:
            return False
        
    


import torch
from iou_loss import Iou
class Losses():
    lambda_coord = torch.Tensor([0.5]).cuda()
    lambda_noobj = torch.Tensor([0.5]).cuda()
    
    @staticmethod
    def some_iterator(predictions):
        predictions_view = predictions.view(-1, 7,7, 30)
        for image_nr in range(predictions_view.shape[0]):
            for y in range(predictions_view.shape[1]):
                for x in range(predictions_view.shape[2]):
                     
                    yield y,x,predictions_view[image_nr, y,x, :]
    
    @staticmethod
    def get_loc_error(predictions, targets):
        
        loss = 0
        for y,x,pred in Losses.some_iterator(predictions):
            x1,y1,_,_,_,x2,y2 = pred[:7]
            if (y,x) in targets:
                for i, tar in enumerate(targets[(y,x)]):
                    ypred, xpred = tar[2]
                    if i == 0:
                        loss += Losses.lambda_coord*(torch.pow(x1-xpred,2)+ torch.pow(y1-ypred,2))          
                    elif i == 1:
                        loss += Losses.lambda_coord*(torch.pow(x2-xpred,2)+ torch.pow(y2-ypred,2))
                    else:
                        break
                    
        return loss
    @staticmethod
    def get_wh_err_equation(wpred,wtar, hpred,htar):
        errw = Losses.lambda_coord*torch.pow(torch.sqrt(wpred)-torch.sqrt(wtar),2)
        errh = Losses.lambda_coord*torch.pow(torch.sqrt(hpred)-torch.sqrt(htar),2)
        return errw + errh
    @staticmethod
    def get_w_h_error(predictions, targets):
        loss = 0
        for y,x,pred in Losses.some_iterator(predictions):
            _,_,w1pred,h1pred,_,_,_, w2pred,h2pred = pred[:9]
            if (y,x) in targets:
                
                for i, tar in enumerate(targets[(y,x)]):
                    wtar, htar = tar[1]
                    if i == 0:
                        loss += Losses.get_wh_err_equation(w1pred, wtar, h1pred, htar)      
                    elif i == 1:
                        loss += Losses.get_wh_err_equation(w2pred, wtar, h2pred, htar)
                    else:
                        break
        return loss
    @staticmethod
    def get_confidence_error(predictions, targets):
        loss = 0
        for y,x,pred in Losses.some_iterator(predictions):
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
    @staticmethod            
    def get_conditional_class_prob_exist(predictions, targets):
        loss = 0
        for y,x,pred in Losses.some_iterator(predictions):
            pred_class = pred[10:]
            if (y,x) in targets:
                des_class = targets[(y,x)][0][0]
                for c in range(20):
                    if c == des_class:
                        loss += torch.pow(torch.Tensor([1]).cuda()-pred_class[c],2)
                    else:
                        loss += torch.pow(torch.Tensor([0]).cuda()-pred_class[c],2)                
        return loss
    @staticmethod
    def get_conditional_class_prob_notexist(predictions, targets):
        loss = 0
        for y,x,pred in Losses.some_iterator(predictions):
            pred_class = pred[10:]
            if (y,x) not in targets:
                for c in range(20):
                    loss += Losses.lambda_noobj*torch.pow(torch.Tensor([0]).cuda()-pred_class[c],2)                
        return loss

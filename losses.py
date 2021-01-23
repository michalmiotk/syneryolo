import torch
import torch.nn as nn
from iou_loss import Iou

class Losses():
    lambda_coord = torch.Tensor([5]).cuda()
    lambda_noobj = torch.Tensor([0.5]).cuda()
    min_stab = torch.Tensor([1e-6]).cuda()
    cell_num = 7
    @staticmethod
    def get_tar_vector(targets):
        tar_vector = torch.zeros((len(targets), Losses.cell_num, Losses.cell_num, 30)).cuda()
        for i,target in enumerate(targets):
            for y in range(Losses.cell_num):
                for x in range(Losses.cell_num):
                    if cell_tar:=target.get((y,x),None): 
                        if tar_vector[i,y,x,4] == 0:
                            #fill prob
                            tar_vector[i,y,x,4] = 1
                            #fill class
                            tar_vector[i,y,x,5+cell_tar[0]] = 1
                            #fill xy
                            tar_vector[i,y,x,0] = cell_tar[1][0]
                            tar_vector[i,y,x,1] = cell_tar[1][1]
                            #fill wh
                            tar_vector[i,y,x,2] =  cell_tar[2][0]
                            tar_vector[i,y,x,3] =  cell_tar[2][1]
        return tar_vector
        
    @staticmethod
    def get_loc_error(predictions, targets):
        highest_iou = Losses.who_has_highest_iou(predictions, targets)
        highest_iou = highest_iou.reshape(targets.shape[0], targets.shape[1], targets.shape[2],1)

        first_1 = 1-highest_iou[:,:,:,0]
        print("torch max iou", torch.max(highest_iou[:,:,:,0]))
        print("torch min iou", torch.min(highest_iou[:,:,:,0]))
        predictions = predictions.reshape(targets.shape[0], targets.shape[1], targets.shape[2],30)
        loss = first_1*(torch.pow(predictions[:,:,:,0]-targets[:,:,:,0],2)+ torch.pow(predictions[:,:,:,1]-targets[:,:,:,1],2)) 
        loss += highest_iou[:,:,:,0]*(torch.pow(predictions[:,:,:,5]-targets[:,:,:,0],2)+ torch.pow(predictions[:,:,:,6]-targets[:,:,:,1],2))                                   
        loss *= targets[:,:,:, 4]
        loss = loss.cuda()

        return Losses.lambda_coord*loss
    '''
    @staticmethod
    def get_w_h_error(predictions, targets):
       
        return Losses.lambda_coord*loss
    
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
    '''
    @staticmethod
    def who_has_highest_iou(pred,tar):
        pred = pred.reshape(-1, Losses.cell_num, Losses.cell_num, 30)

        x1,y1,w1,h1 = pred[:,:,:,0], pred[:,:,:,1], pred[:,:,:,2], pred[:,:,:,3]
        x2,y2,w2,h2 = pred[:,:,:,5], pred[:,:,:,6], pred[:,:,:,7], pred[:,:,:,8]

        tarx, tary = tar[:,:,:, 0], tar[:,:,:, 1]
        tarw, tarh =tar[:,:,:, 2], tar[:,:,:, 3]

        iou1 = Iou.get_iou(x1, y1, w1, h1, tarx, tary, tarw, tarh)
        iou2 = Iou.get_iou(x2, y2, w2, h2, tarx, tary, tarw, tarh)

        return torch.argmax(torch.cat((iou1.unsqueeze(0), iou2.unsqueeze(0))), dim=0)
        
    


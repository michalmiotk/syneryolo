
import torch 

class Utils:
    size = 448
    cellnr = 7
    celldim = size/cellnr
    @staticmethod
    def get_xmin_ymin_xmax_ymax(xcelloffset,ycelloffset,w,h, xcell, ycell):
        centerx = (xcell +xcelloffset)*Utils.celldim
        centery = (ycell +ycelloffset)*Utils.celldim
        xmin = centerx - w*Utils.size/2
        xmax = centerx + w*Utils.size/2
        ymin = centery - h*Utils.size/2
        ymax = centery + h*Utils.size/2
        return xmin, xmax, ymin, ymax
    @staticmethod
    def get_iou(xcell, ycell, predx, predy, predw, predh, tarx, tary, tarw, tarh):
        xminpred, xmaxpred, yminpred, ymaxpred = Utils.get_xmin_ymin_xmax_ymax(predx, predy, predw, predh, xcell, ycell)
        xmintar, xmaxtar, ymintar, ymaxtar = Utils.get_xmin_ymin_xmax_ymax(tarx, tary, tarw, tarh, xcell, ycell)
        if (torch.pow(xminpred, 2) + torch.pow(yminpred, 2)) < (torch.pow(xmintar, 2) + torch.pow(ymintar, 2)): #pred is over tar
            xminup, xmaxup, yminup, ymaxup = xmintar, xmaxtar, ymintar, ymaxtar
            xmindown, xmaxdown, ymindown, ymaxdown = xminpred, xmaxpred, yminpred, ymaxpred
        else:
            xminup, xmaxup, yminup, ymaxup = xminpred, xmaxpred, yminpred, ymaxpred
            xmindown, xmaxdown, ymindown, ymaxdown =  xmintar, xmaxtar, ymintar, ymaxtar
        
        if (torch.pow(xmindown, 2)+ torch.pow(ymindown, 2)) < (torch.pow(xmaxup, 2)+ torch.pow(ymaxdown, 2)):
            intersection = (xmaxup - xmindown)*(ymaxdown-yminup)
            union = (xmaxup-xminup)*(ymaxup-yminup) + (xmaxdown-xminup)*(ymaxup-yminup) - intersection
            return intersection/union
        else:
            return 0
            
             
            
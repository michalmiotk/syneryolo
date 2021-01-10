
import torch 

class Iou:
    size = 448
    cellnr = 7
    celldim = size/cellnr
    @staticmethod
    def get_xmin_ymin_xmax_ymax(xcelloffset,ycelloffset,w,h, xcell, ycell):
        centerx = (xcell +xcelloffset)*Iou.celldim
        centery = (ycell +ycelloffset)*Iou.celldim
        xmin = centerx - w*Iou.size/2
        xmax = centerx + w*Iou.size/2
        ymin = centery - h*Iou.size/2
        ymax = centery + h*Iou.size/2
        '''
        -------------------
        |ymin,xmin          |
        |                   |
        |                   |
        |                   |
        |                   |
        |                   |
        |                   |
        |         ymax,xmax |
        -------------------
        '''
        return xmin, ymin, xmax, ymax
    @staticmethod
    def get_iou(xcell, ycell, predx, predy, predw, predh, tarx, tary, tarw, tarh):
        '''
        --------------------
        | yminup,xminup     |
        |                   |
        |   ymindown,xmindown
        |            _______|________
        |           |       |       |
        |           |       |       |
        |           |       |       |
        ------------|--------       |
                    |  ymaxup,xmaxup|
                    |               |
                    |               |
                    |               |
                    |               |
                    -----------------ymaxdown,xmaxdown
        '''
        xminpred, yminpred, xmaxpred, ymaxpred = Iou.get_xmin_ymin_xmax_ymax(predx, predy, predw, predh, xcell, ycell)
        xmintar, ymintar, xmaxtar, ymaxtar = Iou.get_xmin_ymin_xmax_ymax(tarx, tary, tarw, tarh, xcell, ycell)
        if (torch.pow(xminpred, 2) + torch.pow(yminpred, 2)) < (torch.pow(xmintar, 2) + torch.pow(ymintar, 2)): #pred is over tar
            xminup, xmaxup, yminup, ymaxup = xmintar, xmaxtar, ymintar, ymaxtar
            xmindown, xmaxdown, ymindown, ymaxdown = xminpred, xmaxpred, yminpred, ymaxpred
        else:
            xminup, xmaxup, yminup, ymaxup = xminpred, xmaxpred, yminpred, ymaxpred
            xmindown, xmaxdown, ymindown, ymaxdown =  xmintar, xmaxtar, ymintar, ymaxtar
        
        if (torch.pow(xmindown, 2)+ torch.pow(ymindown, 2)) < (torch.pow(xmaxup, 2)+ torch.pow(ymaxup, 2)):
            intersection = (xmaxup - xmindown)*(ymaxup-ymindown)
            union = (xmaxup-xminup)*(ymaxup-yminup) + (xmaxdown-xmindown)*(ymaxdown-ymindown) - intersection
            return intersection/union
        else:
            return 0
            
             
            
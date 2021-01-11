
import torch 
import torchvision

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
        intersection_union = torchvision.ops.box_iou(torch.Tensor([[xminpred, yminpred, xmaxpred, ymaxpred]]).cuda(), torch.Tensor([[xmintar, ymintar, xmaxtar, ymaxtar]]).cuda())
        if intersection_union>1:
            print(intersection_union)
        return intersection_union

            
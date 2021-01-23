
import torch 
import torchvision

class Iou:
    size = 448
    cellnr = 7
    celldim = size/cellnr
    @staticmethod
    def get_xmin_ymin_xmax_ymax(inxcelloffset,inycelloffset,inw,inh):
        out_xmin, out_xmax, out_ymin, out_ymax = torch.zeros((inw.shape[0], inw.shape[1], inw.shape[2])), torch.zeros((inw.shape[0], inw.shape[1], inw.shape[2])), torch.zeros((inw.shape[0], inw.shape[1], inw.shape[2])), torch.zeros((inw.shape[0], inw.shape[1], inw.shape[2]))
        for ycell in range(Iou.cellnr):
            for xcell in range(Iou.cellnr):
                centerx = (xcell +inxcelloffset[:,ycell,xcell])*Iou.celldim
                centery = (ycell +inycelloffset[:,ycell,xcell])*Iou.celldim
                out_xmin[:,ycell,xcell] = centerx - inw[:,ycell,xcell]*Iou.size/2
                out_xmax[:,ycell,xcell] = centerx + inw[:,ycell,xcell]*Iou.size/2
                out_ymin[:,ycell,xcell] = centery - inh[:,ycell,xcell]*Iou.size/2
                out_ymax[:,ycell,xcell] = centery + inh[:,ycell,xcell]*Iou.size/2
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
        return out_xmin, out_ymin, out_xmax, out_ymax
    @staticmethod
    def get_iou(predx, predy, predw, predh, tarx, tary, tarw, tarh):
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
        xminpred, yminpred, xmaxpred, ymaxpred = Iou.get_xmin_ymin_xmax_ymax(predx, predy, predw, predh)
        bsize=xminpred.shape[0]
        xminpred, yminpred, xmaxpred, ymaxpred = xminpred.view(-1,1), yminpred.view(-1,1), xmaxpred.view(-1,1), ymaxpred.view(-1,1)
        
        xmintar, ymintar, xmaxtar, ymaxtar = Iou.get_xmin_ymin_xmax_ymax(tarx, tary, tarw, tarh)
        xmintar, ymintar, xmaxtar, ymaxtar = xmintar.view(-1,1), ymintar.view(-1,1), xmaxtar.view(-1,1), ymaxtar.view(-1,1)

        boxes1 = torch.cat((xminpred, yminpred, xmaxpred, ymaxpred), dim=1).cuda()
        boxes2 = torch.cat((xmintar, ymintar, xmaxtar, ymaxtar), dim=1).cuda()

        intersection_union = torchvision.ops.box_iou(boxes1, boxes2)
        intersection_union = torch.diagonal(intersection_union)

        return intersection_union

            
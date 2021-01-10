from iou_loss import Iou
import torch
xcell, ycell = 3,3
predx, predy, predw, predh = [torch.Tensor([x]) for x in [0.2,0.2,0.3,0.3]]
tarx, tary, tarw, tarh = [torch.Tensor([x]) for x in [0.8,0.8, 0.3,0.3]]

print(Iou.get_iou(xcell, ycell, predx, predy, predw, predh, tarx, tary, tarw, tarh))
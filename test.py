from iou_loss import Iou
import torch
from losses import Losses
import unittest
cell_num = 14
xcell, ycell = 3,3
predx, predy, predw, predh = [torch.Tensor([x]) for x in [0.2,0.2,0.4,0.4]]
tarx, tary, tarw, tarh = [torch.Tensor([x]) for x in [0.8,0.8, 0.4,0.4]]
iou = Iou.get_iou(xcell, ycell, predx, predy, predw, predh, tarx, tary, tarw, tarh)
assert 0.43 < iou < 0.45


some_target_for_loc_error = {(3,3):[[0, [1,1], [torch.Tensor([0.1]).cuda(), torch.Tensor([0.1]).cuda()]]]} 
some_predictions_for_loc_error = torch.zeros((1,cell_num,cell_num,30))
for y in range(cell_num):
    for x in range(cell_num):
        some_predictions_for_loc_error[0,y,x,:] = torch.Tensor([0.2, 0.2,0,0,0,0.2,0.2,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
unittest.TestCase().assertAlmostEqual(0.01, Losses.get_loc_error(some_predictions_for_loc_error, some_target_for_loc_error).cpu().detach().numpy()[0], 2)

some_targets_class_prob_exist = {(3,3):[[1, [0,0], [torch.Tensor([0]).cuda(), torch.Tensor([0]).cuda()]]]}
some_predictions_class_prob_exist = torch.zeros((1,cell_num,cell_num,30))
for y in range(cell_num):
    for x in range(cell_num):
        if y == 3 and x == 3:
            some_predictions_class_prob_exist[0,y,x,:] = torch.Tensor([0,0,0,0,0,0,0,0,0,0, 0.2,0.2, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        else:
            some_predictions_class_prob_exist[0,y,x,:] = torch.zeros((30))

unittest.TestCase().assertAlmostEqual(0.68, Losses.get_conditional_class_prob_exist(some_predictions_class_prob_exist, some_targets_class_prob_exist).cpu().detach().numpy()[0], 2)

some_targets_class_prob_notexist = {}
some_predictions_class_prob_notexist = torch.zeros((1,cell_num,cell_num,30))
for y in range(cell_num):
    for x in range(cell_num):
        some_predictions_class_prob_notexist[0,y,x,:] = torch.ones((30))

unittest.TestCase().assertAlmostEqual(0.5(noobject_coef*20*cell_num*cell_num), Losses.get_conditional_class_prob_notexist(some_predictions_class_prob_notexist, some_targets_class_prob_notexist).cpu().detach().numpy()[0], 2)
#0.5(noobject_coef*20objects*7*7) == 490

target_for_wh = {(3,3):[[0, [torch.Tensor([0.36]).cuda(),torch.Tensor([0.64]).cuda()], [0, 0]]]} 
predictions_wh = torch.zeros((1,cell_num,cell_num,30))
for y in range(cell_num):
    for x in range(cell_num):
        if y == 3 and x == 3:
            predictions_wh[0,y,x,:] = torch.Tensor([0,0,0.04,0.09,0,0,0,0,0,0, 0, 0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        else:
            predictions_wh[0,y,x,:] = torch.zeros((30))


#0.5*((0.6-0.2)^2 + (0.8-0.3)^2 ) = 0.5(0.16+0.25) = 0.5*0.41 = 0.205
unittest.TestCase().assertAlmostEqual(0.205, Losses.get_w_h_error(predictions_wh, target_for_wh).cpu().detach().numpy()[0],3)
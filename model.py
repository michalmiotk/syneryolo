import torch.nn as nn
import torch
import torchvision.models as models
import torchvision
from my_transforms import Transform_img_labels


class Yolo(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18(pretrained=True)
        self.prelast_linear = torch.nn.Linear(512*7*7, 4096)
        self.last_linear = torch.nn.Linear(4096, 7*7*30)
        self.lambda_coord = torch.Tensor([0.5])
        self.lambda_noobj = torch.Tensor([0.5])
        self.cats = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                     'dog', 'horse', 'motorbike', 'person', 'sheep', 'sofa', 'diningtable', 'pottedplant', 'train', 'tvmonitor']

    def get_index(self, name):
        return self.cats.index(name)

    def dataset_inspect(self):
        root_dir = "/home/m/Pobrane/VOC2007"
        transformacje=Transform_img_labels()
        self.dataset = torchvision.datasets.VOCDetection(
            root_dir, year="2007", image_set='train', download=True, transforms=transformacje)

        for i, x in enumerate(self.dataset):
            if i <60:
                image, annot = x
                print(annot)
            else:
                break

    def get_loc_error(self):
        pass

    def get_w_h_error(self):
        pass

    def get_confidence_error(self):
        pass

    def get_conditional_class_prob_exist(self):
        pass

    def get_conditional_class_prob_notexists(self):
        pass

    def forward_imagenet(self, input):
        x = self.model.conv1(input)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        return x

    def forward(self, input):
        x = self.forward_imagenet(input)
        x = x.view(-1, 512*7*7)
        print(x.shape)
        x = self.prelast_linear(x)
        x = self.last_linear(x)
        print(x.shape)


yolo = Yolo()
yolo.dataset_inspect()
#yolo.forward(torch.Tensor(2,3, 224, 224))

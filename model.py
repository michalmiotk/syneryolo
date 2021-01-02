import torch.nn as nn
import torch

class Yolo(nn.Module):
    def __init__(self):
        super().__init__()
        self.create_model_vars()
    
    def one_pad_conv(self, in_channels, out_channels, kernel_size, stride=(1,1)):
        return nn.Conv2d(in_channels, out_channels,  kernel_size, stride, padding=1)

    def create_model_vars(self):
        self.conv1 = self.one_pad_conv(3, 64, (7,7))
        self.maxpool1 = nn.MaxPool2d((2,2), stride=2)
        self.conv2 = self.one_pad_conv(64, 192,(3,3))
        self.maxpool2 = nn.MaxPool2d((2,2), stride=2)
        self.conv3_1 = self.one_pad_conv(192, 256, (1,1))
        self.conv3_2 = self.one_pad_conv(256, 256, (3,3))
        self.conv3_3 = self.one_pad_conv(256, 512, (1,1))
        self.conv3_4 = self.one_pad_conv(512, 256, (3,3))
        self.maxpool3 = nn.MaxPool2d((2,2), 2)
        self.conv4_1_1 = self.one_pad_conv(256,512 , (1,1))
        self.conv4_1_2 = self.one_pad_conv(512, 256, (3,3))
        self.conv4_1_3 = self.one_pad_conv(256,512 , (1,1))
        self.conv4_1_4 = self.one_pad_conv(512, 256, (3,3))
        self.conv4_1_5 = self.one_pad_conv(256,512 , (1,1))
        self.conv4_1_6 = self.one_pad_conv(512, 256, (3,3))
        self.conv4_1_7 = self.one_pad_conv(256,512 , (1,1))
        self.conv4_1_8 = self.one_pad_conv(512, 512, (3,3))
        self.conv4_2 = self.one_pad_conv(512, 1024, (1,1))
        self.conv4_3 = self.one_pad_conv(1024, 512, (3,3))
        self.maxpool4 = nn.MaxPool2d((2,2), 2)
        self.conv5_1_1 = self.one_pad_conv(512,1024, (1,1))
        self.conv5_1_2 = self.one_pad_conv(1024, 512, (3,3))
        self.conv5_1_3 = self.one_pad_conv(512,1024, (1,1))
        self.conv5_1_4 = self.one_pad_conv(1024, 512, (3,3))
        self.conv5_2 = self.one_pad_conv(512, 1024, (3,3))
        self.conv5_3 = self.one_pad_conv(1024, 1024, (3,3), stride=2)
        self.conv6_1 = self.one_pad_conv(1024, 1024, (3,3))
        self.conv6_2 = self.one_pad_conv(1024, 1024, (3,3))

        self.linear_7 = nn.Linear(7*7*1024, 4096)
        self.out_8 = nn.Linear(4096, 7*7*30)

        self.out_imagenet =  nn.Linear(4096, 1000)

    def forward_common(self, input):
        x = self.conv1(input)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.maxpool3(x)
        x = self.conv4_1_1(x)
        x = self.conv4_1_2(x)
        x = self.conv4_1_3(x)
        x = self.conv4_1_4(x)
        x = self.conv4_1_5(x)
        x = self.conv4_1_6(x)
        x = self.conv4_1_7(x)
        x = self.conv4_1_8(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.maxpool4(x)
        x = self.conv5_1_1(x)
        x = self.conv5_1_2(x)
        x = self.conv5_1_3(x)
        x = self.conv5_1_4(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv6_1(x)
        x = self.conv6_2(x)
        return x
    
    def forward_imagenet(self, input):
        x = self.forward_common(input)
        print(x.shape)
        x = x.view(-1, 7*7*1024)
        x = self.linear_7(x)
        x = self.out_imagenet(x)
        return x

yolo = Yolo()
print(yolo.forward_imagenet(torch.Tensor(2,3, 224, 224)))
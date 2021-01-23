import torchvision, torch
model = torchvision.models.resnet18()
maxpo = torch.nn.MaxPool2d(kernel_size=3, stride=2)
def forward(inp, model):
    x = model.conv1(inp)
    x = model.bn1(x)
    x = model.relu(x)
    x = maxpo(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4[0](x)
    x = model.layer4[1](x)
    return x
print(forward(torch.ones([1,3,448,448]), model).shape)
from torch.utils.data import DataLoader
import torchvision
import torch

year = '2007'
root_dir_train = "/home/m/Pobrane/VOC"+year+"_train"
def my_collate(batch):
        data = [item[0].unsqueeze(0) for item in batch]
        data = torch.cat(data)
        target = [item[1] for item in batch]
        return [data, target]
train_dataset = torchvision.datasets.VOCDetection(root_dir_train, year=year,image_set='train', download=True, transform=torchvision.transforms.Compose([torchvision.transforms.Resize(
            (448,448)), torchvision.transforms.ToTensor()]))
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False, collate_fn=my_collate)

def get_mean_std(loader):
    #VAR[x] = E[X**2] - E[X]**2
    channels_sum, channels_squared_sum, num_batches = 0,0,0
    
    for data, _ in loader:
        print(num_batches)
       
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1
        
    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches-mean**2)**0.5
    
    return mean, std

mean, std  =get_mean_std(train_loader)
print(mean, std)
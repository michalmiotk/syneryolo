import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

class LitAutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 10))
        self.lr = 1e-3
        self.mnist = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        #print(y)
        z = self.encoder(x)
        out = self.decoder(z)
        out_proper = torch.zeros(10)
        out_proper[y] = 1
        loss = 3*torch.mean(torch.pow(out-out_proper,2))
        print(loss)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    def train_dataloader(self):
        return self.mnist

autoencoder = LitAutoEncoder()
trainer = pl.Trainer()

autoencoder.lr = 0.0001

# Fit model
trainer.fit(autoencoder)
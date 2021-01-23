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
        self.decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))
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
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        #print(loss)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    def train_dataloader(self):
        return self.mnist

autoencoder = LitAutoEncoder()
trainer = pl.Trainer()
# Run learning rate finder
lr_finder = trainer.tuner.lr_find(autoencoder,early_stop_threshold=None,num_training=100, max_lr=1)

# Results can be found in
lr_finder.results

# Plot with
fig = lr_finder.plot(suggest=True)
fig.show()
fig.savefig("lrx.png")
# Pick point based on plot, or get suggestion
new_lr = lr_finder.suggestion()
print("new lr is", new_lr)
# update hparams of the model
autoencoder.lr = new_lr

# Fit model
trainer.fit(autoencoder)
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.datasets import MNIST
import pytorch_lightning as pl

class MNIST_lightning(pl.LightningModule):
  def __init__(self, n_model):
    super().__init__()
    self.model = n_model
  def training_step(self, batch, batch_idx):
      x,y = batch
      #print(x.shape)
      #x = x.view(x.size(0),-1)
      y_hat = self.model(x)
      loss = nn.functional.cross_entropy(y_hat,y)
      self.log("train_loss",loss)
      return loss
  def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        y_hat = self.model(x)
        loss = nn.functional.cross_entropy(y_hat,y)
        self.log("train_loss",loss)
        return loss
  def configure_optimizers(self):
    optimize = optim.SGD(self.parameters(), lr = 1e-3)
    return optimize
  
  
  # Code from Nhu Thao
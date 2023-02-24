import os
import numpy as np
import pandas as pd

from torchvision.transforms import ToTensor # transform PIL image to torch.Tensor
import torch
from torch.utils.data import DataLoader # mini-batch loader
from torch import nn
from torch.utils.data import random_split
from torchvision.models import resnet50

import hydra
from hydra.core.config_store import ConfigStore

from config import MNISTConfig
from model import MNIST_lightning
from data import CustomDataset

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl


cs = ConfigStore.instance()
cs.store(name="mnist_config", node=MNISTConfig)

labels_item = {
    '0' : 0,
    '1' : 1,
    '2' : 2,
    '3' : 3,
    '4' : 4,
    '5' : 5,
    '6' : 6,
    '7' : 7,
    '8' : 8,
    '9' : 9,
    
 }

def toCSVfile(input_dir, output_dir, file_name):
    dir_list = os.listdir(input_dir)
    img_dir = []
    labels = []

    for dir in dir_list:
        current_path = os.path.join(input_dir, dir)
        idir = os.listdir(current_path)
        lb = [labels_item[dir]]*len(idir)

        img_dir += np.core.defchararray.add(current_path + "/", np.array(idir)).tolist()
        labels += lb
    df = pd.DataFrame({'filename': img_dir, 'label':labels})
        
    out_dir = output_dir + '/' + file_name
    if not os.path.exists(output_dir):
            os.mkdir(out_dir)
            if os.path.exists(out_dir):
                os.remove(out_dir)
    df.to_csv(out_dir, index = False)


@hydra.main(config_path='../configs', config_name='train', version_base=None)
def main(cfg: MNISTConfig):
    toCSVfile(cfg.paths.data + '/' + cfg.files.train_folder, 
            cfg.paths.data,
            cfg.files.train_file)

    
    data = CustomDataset(cfg.paths.data + '/' + cfg.files.train_file, dir = None, transform = ToTensor())

    # use 20% of training data for validation
    train_set_size = int(len(data) * cfg.train_size)
    valid_set_size = len(data) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(cfg.params.seed)
    train_set, valid_set = random_split(data, [train_set_size, valid_set_size], generator=seed)

    # define dataloader
    train_loader = DataLoader(train_set, batch_size = cfg.params.batch_size, shuffle = True)
    valid_loader = DataLoader(valid_set, batch_size = cfg.params.batch_size, shuffle = False)

    # defind model
    model = MNIST_lightning(resnet50(num_classes = 10))
    trainer = pl.Trainer(devices=1, 
                            accelerator="gpu", 
                            callbacks=[EarlyStopping(monitor="train_loss", mode="min")], 
                            max_epochs = cfg.params.n_epoch)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)


if __name__ == '__main__':
    main()

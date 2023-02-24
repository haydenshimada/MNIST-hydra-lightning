from dataclasses import dataclass


@dataclass
class Paths:
    data: str


@dataclass
class Files:
    train_folder: str
    train_file: str


@dataclass
class Params:
    n_epoch: int
    lr: float
    batch_size: int
    num_workers: int
    seed: int

# @dataclass
# class Data:
#     train_size: float

@dataclass
class MNISTConfig:
    paths: Paths
    files: Files
    params: Params
    # data: Data
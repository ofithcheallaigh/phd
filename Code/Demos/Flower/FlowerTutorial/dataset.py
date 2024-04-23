import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import random_split, DataLoader


def get_mnist(data_path: str = './data'):

    tr = Compose([ToTensor(), Normalize((0.1307,),(0.3081,))])
    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)

    return trainset, testset


def prepare_dataset(num_partitions: int, 
                    batch_side: int, 
                    val_ratio: float = 0.1):

    trainset, testset = get_mnist()

    # Split trainset into 'num_partitions' trainsets
    num_images = len(trainset) // num_partitions # number of imagines in each partition
    partition_len = [num_images] * num_partitions

    # split randomly. This returns a list of trainsets, each with `num_images` training examples
    # Note this is the simplest way of splitting this dataset. A more realistic (but more challenging) partitioning
    # would induce heterogeneity in the partitions in the form of for example: each client getting a different
    # amount of training examples, each client having a different distribution over the labels (maybe even some
    # clients not having a single training example for certain classes). If you are curious, you can check online
    # for Dirichlet (LDA) or pathological dataset partitioning in FL. A place to start is: https://arxiv.org/abs/1909.06335

    trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2023))

    # Create dataloaders with train+val support

    trainloaders = []
    valloaders = []
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(trainset_, (num_train, num_val), torch.Generator().manual_seed(2023))

        trainloaders.append(DataLoader(for_train, batch_size=batch_side,shuffle=True,num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_side,shuffle=False,num_workers=2))

    testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader
     
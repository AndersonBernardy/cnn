import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from matplotlib import pyplot as plt
from torchvision import datasets, transforms

def get_device():
    #DEVICE = ('cuda' if torch.cuda.is_available else 'cpu')
    DEVICE = 'cpu'
    return DEVICE

def load_dataset(train=True):
    if train:
        data_path = './datasets/chest_xray/train/'
    else:
        data_path = './datasets/chest_xray/test/'

    train_dataset = datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(size=(256, 256)),
            transforms.ToTensor(),
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=False
    )
    return train_loader

def showimage(data_loader, index):
    sample = data_loader.dataset[index][0]
    nparray = (sample.numpy()*255).astype('uint8').reshape(256, 256)
    img = Image.fromarray(nparray, mode='L')
    plt.imshow(img, cmap='gray')
    class_map = dict(map(reversed, data_loader.dataset.class_to_idx.items()))
    plt.title(class_map[data_loader.dataset[index][1]])
    plt.show()
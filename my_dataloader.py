import numpy
import matplotlib
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def load_train_mnist():
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    path = '../datasets'
    trainDataset = datasets.MNIST(root=path, train=True, transform=trans, download=True)
    trainDataLoader = DataLoader(trainDataset, batch_size=100, shuffle=False)
    classes = {0,1,2,3,4,5,6,7,8,9}
    return trainDataLoader, classes

def load_test_mnist():
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    path = '../datasets'
    testDataset = datasets.MNIST(root=path, train=False, transform=trans, download=True)
    testDataLoader = DataLoader(testDataset, batch_size=100, shuffle=False)
    classes = {0,1,2,3,4,5,6,7,8,9}
    return testDataLoader, classes

def load_train_fashion():
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    path = '../datasets'
    trainDataset = datasets.FashionMNIST(root=path, train=True, transform=trans, download=True)
    trainDataLoader = DataLoader(trainDataset, batch_size=100, shuffle=False)
    classes = {0 :'T-shirt/top', 1 :'Trouser', 2 :'Pullover', 3 :'Dress',
                  4 :'Coat', 5 :'Sandal', 6 :'Shirt', 7 :'Sneaker', 8 :'Bag',
                  9 :'Ankle boot'}
    return trainDataLoader, classes

def load_test_fashion():
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    path = '../datasets'
    testDataset = datasets.FashionMNIST(root=path, train=False, transform=trans, download=True)
    testDataLoader = DataLoader(testDataset, batch_size=100, shuffle=False)
    classes = {0 :'T-shirt/top', 1 :'Trouser', 2 :'Pullover', 3 :'Dress',
                  4 :'Coat', 5 :'Sandal', 6 :'Shirt', 7 :'Sneaker', 8 :'Bag',
                  9 :'Ankle boot'}
    return testDataLoader, classes


def total_count(loader):
    totalClassCount = [0,0,0,0,0,0,0,0,0,0]

    for batch_id,(images,labels) in enumerate(loader):
        for label in labels:
            totalClassCount[int(label)] += 1
    return totalClassCount

def my_hist(loader, classes, title):
    count = total_count(loader)
    fig, ax = matplotlib.pyplot.subplots()
    ax.barh(y=classes, width=count)
    ax.set_xlabel('# Examples')
    ax.set_ylabel('# Classes')
    ax.set_title(title)
    
def my_plot(loader, index):
    img = loader.dataset[index][0].numpy()
    img = numpy.reshape(a=img, newshape=(img.shape[1],img.shape[2]))
    matplotlib.pyplot.figure()
    matplotlib.pyplot.imshow(img)
import pickle
import numpy
import pandas as pd
import seaborn as sns
import torch

from PIL import Image
from matplotlib import pyplot as plt
from torchvision import datasets, transforms

import cnn

def load_dataset(train=True):
    path = './dataset'
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    dataset = datasets.FashionMNIST(root=path, train=train, transform=trans, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=train)
    return dataloader

def train_model():
    dataloader = load_dataset(train=True)
    model = cnn.Net()
    model.to(cnn.get_device())
    model.train()
    model, dictionary = cnn.train_model(model=model, dataloader=dataloader, num_epochs=10)
    print("Time spent: {:.2f}s".format(dictionary['exec_time']))
    torch.save(model.state_dict(), "./modelo/fashion.pt")
    save_stats(dictionary)

def test_model():
    dataloader = load_dataset(train=False)
    model = cnn.Net()
    model.load_state_dict(torch.load("./modelo/fashion.pt"))
    model.to(cnn.get_device())
    model.eval()
    dictionary = cnn.test_model(model=model, dataloader=dataloader)
    classes = list(dataloader.dataset.class_to_idx.keys())
    dictionary['classes'] = classes
    dictionary = {**load_stats(), **dictionary}
    save_stats(dictionary)
    show_stats(dictionary)

def save_stats(dictionary):
    pickle_out = open("dict.pickle", "wb")
    pickle.dump(dictionary, pickle_out)
    pickle_out.close()

def load_stats():
    pickle_in = open("dict.pickle", "rb")
    return pickle.load(pickle_in)

def show_stats(dictionary):
    classes = dictionary['classes']
    matrix = dictionary['confusion_matrix']
    exec_time = dictionary['exec_time']
    dataframe = pd.DataFrame(data=matrix, index=classes, columns=classes)
    correct = numpy.trace(matrix)
    total = matrix.sum()
    print("\n__________________________________________________\n")
    print("Accuracy = {:.2f}\n".format(100*(correct/total)))
    print("Execution Time = {:.2f}\n".format(exec_time))
    print(dataframe)
    print("\n__________________________________________________\n")

    sns.heatmap(dataframe, cmap="Blues", annot=True, fmt="d")
    plt.title("Actual X Predicted")
    plt.show()

def show_image(dataloader, index):
    sample = dataloader.dataset[index][0]
    nparray = (sample.numpy()*255).astype('uint8').reshape(28, 28)
    img = Image.fromarray(nparray, mode='L')
    plt.imshow(img, cmap='gray')
    class_map = dict(map(reversed, dataloader.dataset.class_to_idx.items()))
    plt.title(class_map[dataloader.dataset[index][1]])
    plt.show()

def assert_image(dataloader, index):
    model = cnn.Net()
    model.load_state_dict(torch.load("./modelo/fashion.pt"))
    model.to(cnn.get_device())
    model.eval()

    image = dataloader.dataset[index][0]
    image = image.to(cnn.get_device())
    image = image[None]
    image = image.type('torch.FloatTensor')
    predictated = cnn.assert_image(model, image)
    class_map = dict(map(reversed, dataloader.dataset.class_to_idx.items()))
    print(class_map[predictated])
    show_image(dataloader, index)
    return predictated

# assert_image(load_dataset(train=False), 7512)

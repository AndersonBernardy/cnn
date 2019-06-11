import pickle
import pandas as pd
import seaborn as sns
import torch

from PIL import Image
from matplotlib import pyplot as plt
from torchvision import datasets, transforms

import cnn

def load_dataset(train=True):
    if train:
        data_path = './dataset/chest_xray/train/'
    else:
        data_path = './dataset/chest_xray/test/'

    dataset = datasets.ImageFolder(
        root=data_path,
        transform=transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(size=(256, 256)),
            transforms.ToTensor(),
        ])
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )
    return dataloader

def train_model():
    dataloader = load_dataset(train=True)
    model = cnn.Net()
    model.to(cnn.get_device())
    model.train()
    model, dictionary = cnn.train_model(model=model, dataloader=dataloader, num_epochs=10)
    print("Time spent: {:.2f}s".format(dictionary['exec_time']))
    torch.save(model.state_dict(), "./modelo/pneumonia.pt")

def test_model():
    dataloader = load_dataset(train=False)
    model = cnn.Net()
    model.load_state_dict(torch.load("./modelo/pneumonia.pt"))
    model.to(cnn.get_device())
    model.eval()
    dictionary = cnn.test_model(model=model, dataloader=dataloader)
    classes = list(dataloader.dataset.class_to_idx.keys())
    dictionary['classes'] = classes
    save_stats(dictionary)
    show_stats(dictionary)

def save_stats(dictionary):
    print(dictionary)
    pickle_out = open("dict.pickle", "wb")
    pickle.dump(dictionary, pickle_out)
    pickle_out.close()

def load_stats():
    pickle_in = open("dict.pickle", "rb")
    return pickle.load(pickle_in)

def show_stats(dictionary):
    print("\n__________________________________________________\n")
    classes = dictionary['classes']
    matrix = dictionary['confusion_matrix']
    dataframe = pd.DataFrame(data=matrix, index=classes, columns=classes)
    print("Accuracy = {:.2f}\n".format(100*(dictionary['correct']/dictionary['total_size'])))
    print(dataframe)
    sns.heatmap(dataframe, cmap="Blues")
    plt.title("Actual X Predicted")
    print("\n__________________________________________________\n")
    plt.show()

def show_image(dataloader, index):
    sample = dataloader.dataset[index][0]
    nparray = (sample.numpy()*255).astype('uint8').reshape(256, 256)
    img = Image.fromarray(nparray, mode='L')
    plt.imshow(img, cmap='gray')
    class_map = dict(map(reversed, dataloader.dataset.class_to_idx.items()))
    plt.title(class_map[dataloader.dataset[index][1]])
    plt.show()

def assert_image(dataloader, index):
    show_image(dataloader, index)

test_model()

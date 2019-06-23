import time
import numpy
import torch

import torch.nn.functional as F
import torch.nn as nn

DEVICE = ('cuda' if torch.cuda.is_available else 'cpu')

def get_device():
    return DEVICE

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=2)

        self.fc1 = nn.Linear(14400, 360)
        self.fc2 = nn.Linear(360, 90)
        self.fc3 = nn.Linear(90, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 1)

        # reshape
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(model, dataloader, num_epochs=10):

    dictionary = {}
    start_time = time.time()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss().to(get_device())

    print("Starting Time: {}".format(time.strftime('%H:%M:%S')))
    for epoch in range(num_epochs):

        print("Epoch = {}/{} ".format(epoch + 1, num_epochs), end=" ")

        for _, (image, label) in enumerate(dataloader):
            optimizer.zero_grad()
            image = torch.autograd.Variable(image)
            image = image.to(get_device())
            label = torch.autograd.Variable(label)
            label = label.to(get_device())
            output = model.forward(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

        print("Loss = {:.5f}".format(loss.item()))

    print("Ending Time: {}".format(time.strftime('%H:%M:%S')))

    end_time = time.time()
    exec_time = end_time - start_time

    dictionary['exec_time'] = exec_time

    return model, dictionary

def test_model(model, dataloader):

    print("Testing")

    dictionary = {}
    confusion_matrix = numpy.zeros((2, 2), dtype=numpy.long)

    for _, (image, label) in enumerate(dataloader):

        image = torch.autograd.Variable(image)
        image = image.to(get_device())
        label = torch.autograd.Variable(label)
        label = label.to(get_device())
        output = model.forward(image)

        _, predictated = torch.max(output.data, 1)
        for x in range(len(predictated)):
            confusion_matrix[label[x]][predictated[x]] += 1

    dictionary['confusion_matrix'] = confusion_matrix
    return dictionary

def assert_image(model, image):
    assertion = model(image)
    _, predictated = torch.max(assertion, 1)
    return predictated.data.cpu().numpy()[0]

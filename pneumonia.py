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


# loader = load_dataset(True)
# showimage(loader, 0)



#     def train_model(self, model, dataloader, classes, num_epochs=10):

#         optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
#         criterion = nn.CrossEntropyLoss().to(DEVICE)

#         dictionary = {}
#         totalLoss = []
#         # totalLRs = []
#         # LR = 0


#         startTime = time.time()

#         for epoch in range(num_epochs):

#             # LR = decay(optimizer, epoch)
#             # totalLRs.append(LR)
#             print("Epoch = {}/{} ".format(epoch + 1, num_epochs), end=" ")
#             for _, (image, label) in enumerate(dataloader):
#                 optimizer.zero_grad()
#                 image = torch.autograd.Variable(image)
#                 label = torch.autograd.Variable(label)
#                 image = image.to(DEVICE)
#                 label = label.to(DEVICE)
#                 output = model.forward(image)
#                 loss = criterion(output, label)
#                 loss.backward()
#                 optimizer.step()

#             print("Loss = {:.5f}".format(loss.item()))
#             totalLoss.append(loss.data.item())

#         end_time = time.time()   
#         exec_time = end_time - startTime

#         dictionary['execTime'] = exec_time
#         dictionary['totalLoss'] = totalLoss
#         # dictionary['totalLRs'] = totalLRs

#         return model, dictionary

#     def decay(optimizer, epoch):
#         for param in optimizer.param_groups:
#             lr = param['lr'] * (0.1**(epoch//7))
#             param['lr'] = lr
#         return lr

# def test_model(model, dataloader, classes):
#     print("Testing")

#     dictionary = {}

#     correct = 0
#     incorrect = 0
#     total = 0

#     confusionMatrix = torch.zeros(len(classes), len(classes), dtype=torch.long).to(DEVICE)

#     for _,(image, label) in enumerate(dataloader):

#         image = torch.autograd.Variable(image)
#         label = torch.autograd.Variable(label)
#         image = image.to(DEVICE)
#         label = label.to(DEVICE)
#         output = model.forward(image)

#         _, predictated = torch.max(output.data, 1)
#         for x in range(dataloader.batch_size):
#             confusionMatrix[predictated[x]][label[x]] += 1
#             if predictated[x] == label[x]: correct += 1
#             else: 
#                 incorrect += 1
#                 total += label.size(0)

#     dictionary['correct'] = correct
#     dictionary['totalSize'] = total
#     dictionary['confusionMatrix'] = confusionMatrix.to('cpu')

#     return dictionary





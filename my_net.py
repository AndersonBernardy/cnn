import torch
import torch.nn as nn
import torch.nn.functional as F
import time

device = ('cuda' if torch.cuda.is_available else 'cpu')
#device = 'cpu'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        
        self.fc1 = nn.Linear(1024,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),1)
        
        # reshape
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def train_model(model, dataloader, classes, num_epochs=10):
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss().to(device)

    dictionary = {}
    totalLoss = []
    # totalLRs = []
    # LR = 0
    
    
    startTime = time.time()
    
    for epoch in range(num_epochs):
        
        LR = decay(optimizer, epoch)
        # totalLRs.append(LR)
            
        print("Epoch = {}/{} ".format(epoch + 1, num_epochs), end=" ")
        for batch_id,(image, label) in enumerate(dataloader):
        
            optimizer.zero_grad()
            image = torch.autograd.Variable(image)
            label = torch.autograd.Variable(label)
            image = image.to(device)
            label = label.to(device)
            output = model.forward(image)
            loss = criterion(output,label)
            loss.backward()
            optimizer.step()

        print("Loss = {:.5f}".format(loss.item()))
        totalLoss.append(loss.data.item())
        
    endTime = time.time()   
    execTime = endTime - startTime
    
    dictionary['execTime'] = execTime
    dictionary['totalLoss'] = totalLoss
    # dictionary['totalLRs'] = totalLRs

    
    return model, dictionary

def decay(optimizer, epoch):
     for param in optimizer.param_groups:
        LR = param['lr'] * (0.1**(epoch//7))
        param['lr'] = LR
        return LR;
    
def test_model(model, dataloader, classes):
    print("Testing")
    
    dictionary = {}
    
    correct = 0
    incorrect = 0
    total = 0
    
    confusionMatrix = torch.zeros(len(classes), len(classes), dtype=torch.long).to(device)
    
    for batch_id,(image, label) in enumerate(dataloader):
        
        image = torch.autograd.Variable(image)
        label = torch.autograd.Variable(label)
        image = image.to(device)
        label = label.to(device)
        output = model.forward(image)
        
        _, predictated = torch.max(output.data, 1)
        for x in range(dataloader.batch_size):
            confusionMatrix[predictated[x]][label[x]] += 1
            if predictated[x] == label[x]: correct += 1
            else: incorrect += 1
        total += label.size(0)
        
    dictionary['correct'] = correct
    dictionary['totalSize'] = total
    dictionary['confusionMatrix'] = confusionMatrix.to('cpu')
    
    return dictionary
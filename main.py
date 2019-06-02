import torch
import my_dataloader
import my_net

def train_mnist():
    # Load mnist dataset
    mnist_train_data, classes = my_dataloader.load_train_mnist()
    # build and train de model
    model = my_net.Net()
    model.to(my_net.device)
    model, dictModel = my_net.train_model(model=model, dataloader=mnist_train_data, classes=classes, num_epochs=10)
    print("Time spent: {:.2f}s".format(dictModel['execTime']))
    # save the model
    torch.save(model.state_dict(), "../modelos/mnist_cnn.pt")
    
def test_mnist():
    mnist_test_data, classes = my_dataloader.load_test_mnist()
    model = my_net.Net()
    model.load_state_dict(torch.load("../modelos/mnist_cnn.pt"))
    model.to(my_net.device)
    dictModel = my_net.test_model(model=model, dataloader=mnist_test_data, classes=classes)
    print("\nConfusion Matrix: predicted x classes\n{}\n".format(dictModel['confusionMatrix']))
    print("Accuracy == ", 100*(dictModel['correct']/dictModel['totalSize']))

def train_fashion():
    # Load mnist dataset
    fashion_train_data, classes = my_dataloader.load_train_fashion()
    # build and train de model
    model = my_net.Net()
    model.to(my_net.device)
    model, dictModel = my_net.train_model(model=model, dataloader=fashion_train_data, classes=classes, num_epochs=10)
    print("Time spent: {:.2f}s".format(dictModel['execTime']))
    # save the model
    torch.save(model.state_dict(), "../modelos/fashion_cnn.pt")

def test_fashion():
    fashion_test_data, classes = my_dataloader.load_test_fashion()
    model = my_net.Net()
    model.load_state_dict(torch.load("../modelos/fashion_cnn.pt"))
    model.to(my_net.device)
    dictModel = my_net.test_model(model=model, dataloader=fashion_test_data, classes=classes)
    print("\nConfusion Matrix: predicted x classes\n{}\n".format(dictModel['confusionMatrix']))
    print("Accuracy == ", 100*(dictModel['correct']/dictModel['totalSize']))

# train_mnist()
# test_mnist()
# train_fashion()
test_fashion()


import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset

'''class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.name = "Network"
        self.base = nn.Sequential(
          nn.Conv2d(1, 8, 3, 1, 1),
          nn.BatchNorm2d(8),
          nn.ReLU(),
          nn.Conv2d(8, 8, 3, 1, 1),
          nn.BatchNorm2d(8),
          nn.ReLU(),
          nn.Conv2d(8, 16, 1, 1),
          nn.BatchNorm2d(16)
        )
        self.resBlock1 = nn.Sequential(
          nn.Conv2d(16, 16, 3, 1, 1),
          nn.BatchNorm2d(16),
          nn.ReLU(),
          nn.Conv2d(16, 16, 3, 1, 1),
          nn.BatchNorm2d(16),
          nn.MaxPool2d(3, 2, 1),
        )
        self.res2 = nn.Sequential(
          nn.Conv2d(16, 32, 1, 1),
          nn.BatchNorm2d(32)
        )
        self.resBlock2 = nn.Sequential(
          nn.Conv2d(32, 32, 3, 1, 1),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.Conv2d(32, 32, 3, 1, 1),
          nn.BatchNorm2d(32),
          nn.MaxPool2d(3, 2, 1)
        )
        self.res3 = nn.Sequential(
          nn.Conv2d(32, 64, 1, 1),
          nn.BatchNorm2d(64)
        )
        self.resBlock3 = nn.Sequential(
          nn.Conv2d(64, 64, 3, 1, 1),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.Conv2d(64, 64, 3, 1, 1),
          nn.BatchNorm2d(64),
          nn.MaxPool2d(3, 2, 1)
        )
        self.res4 = nn.Sequential(
          nn.Conv2d(64, 128, 1, 1),
          nn.BatchNorm2d(128)
        )
        self.resBlock4 = nn.Sequential(
          nn.Conv2d(128, 128, 3, 1, 1),
          nn.BatchNorm2d(128),
          nn.ReLU(),
          nn.Conv2d(128, 128, 3, 1, 1),
          nn.BatchNorm2d(128),
          nn.MaxPool2d(3, 2, 1)
        )
        self.inference = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 137),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(137, 137),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(137, 7)
        )

    def forward(self, x):
        x = self.base(x)
        x += self.resBlock1(x)
        x = self.res2(x)
        x += self.resBlock2(x)
        x = self.res3(x)
        x += self.resBlock3(x)
        x = self.res4(x)
        x += self.resBlock4(x)
        return self.inference(x)'''

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.name = "Network"
        self.inference = nn.Sequential(
          nn.Conv2d(1, 64, 3, 1, 1),
          nn.ReLU(),
          nn.BatchNorm2d(64),
          nn.Conv2d(64, 64, 3, 1, 1),
          nn.ReLU(),
          nn.BatchNorm2d(64),
          nn.MaxPool2d(2, 2),
          nn.Dropout(0.3),
          
          nn.Conv2d(64, 128, 3, 1, 1),
          nn.ReLU(),
          nn.BatchNorm2d(128),
          nn.Conv2d(128, 128, 3, 1, 1),
          nn.ReLU(),
          nn.BatchNorm2d(128),
          nn.Conv2d(128, 128, 3, 1, 1),
          nn.ReLU(),
          nn.BatchNorm2d(128),
          nn.MaxPool2d(2, 2),
          nn.Dropout(0.3),
          
          nn.Conv2d(128, 256, 3, 1, 1),
          nn.ReLU(),
          nn.BatchNorm2d(256),
          nn.Conv2d(256, 256, 3, 1, 1),
          nn.ReLU(),
          nn.BatchNorm2d(256),
          nn.Conv2d(256, 256, 3, 1, 1),
          nn.ReLU(),
          nn.BatchNorm2d(256),
          nn.Conv2d(256, 256, 3, 1, 1),
          nn.ReLU(),
          nn.BatchNorm2d(256),
          nn.MaxPool2d(2, 2),
          nn.Dropout(0.3),
          
          nn.Conv2d(256, 256, 3, 1, 1),
          nn.ReLU(),
          nn.BatchNorm2d(256),
          nn.Conv2d(256, 256, 3, 1, 1),
          nn.ReLU(),
          nn.BatchNorm2d(256),
          nn.Conv2d(256, 256, 3, 1, 1),
          nn.ReLU(),
          nn.BatchNorm2d(256),
          nn.Conv2d(256, 256, 3, 1, 1),
          nn.ReLU(),
          nn.BatchNorm2d(256),
          nn.MaxPool2d(2, 2),
          nn.Dropout(0.3),
          
          nn.Conv2d(256, 512, 3, 1, 1),
          nn.ReLU(),
          nn.BatchNorm2d(512),
          nn.Conv2d(512, 512, 3, 1, 1),
          nn.ReLU(),
          nn.BatchNorm2d(512),
          nn.Conv2d(512, 512, 3, 1, 1),
          nn.ReLU(),
          nn.BatchNorm2d(512),
          nn.Conv2d(512, 512, 3, 1, 1),
          nn.ReLU(),
          nn.BatchNorm2d(512),
          nn.MaxPool2d(2, 2),
          nn.Dropout(0.3),
          
          nn.Flatten(),
          nn.Linear(512, 7)
        )

    def forward(self, x):
        return self.inference(x)

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

def get_data_loader(batch_size):
    data = pd.read_csv('./fer2013.csv')
    train_set = data[(data.Usage == 'Training')] 
    val_set = data[(data.Usage == 'PublicTest')]
    test_set = data[(data.Usage == 'PrivateTest')] 
    X_train = np.array(list(map(str.split, train_set.pixels)), np.float32) 
    X_val = np.array(list(map(str.split, val_set.pixels)), np.float32) 
    X_test = np.array(list(map(str.split, test_set.pixels)), np.float32) 

    X_train = torch.FloatTensor(X_train.reshape(X_train.shape[0], 1, 48, 48) / 255.0)
    X_val = torch.FloatTensor((X_val.reshape(X_val.shape[0], 1, 48, 48) / 127.5) - 1.0)
    X_test = torch.FloatTensor((X_test.reshape(X_test.shape[0], 1, 48, 48) / 127.5) - 1.0)

    y_train = torch.LongTensor(train_set.emotion.values)
    y_val = torch.LongTensor(val_set.emotion.values)
    y_test = torch.LongTensor(test_set.emotion.values)

    train_transform = transforms.Compose(
            [transforms.ToPILImage(),
            transforms.RandomAffine(10, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])
    
    train = CustomTensorDataset((X_train, y_train), transform=train_transform)
    valid = CustomTensorDataset((X_val, y_val), transform=None)
    test = CustomTensorDataset((X_test, y_test), transform=None)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, num_workers=0)
    
    return train_loader, val_loader, test_loader
    
def get_model_name(name, batch_size, learning_rate, epoch):
    path = "chkpt/model_{0}_bs{1}_lr{2}_epoch{3}".format(name,
                                                   batch_size,
                                                   learning_rate,
                                                   epoch)
    return path

def evaluate(net, loader, criterion):
    """ Evaluate the network on the validation set.

     Args:
         net: PyTorch neural network object
         loader: PyTorch data loader for the validation set
         criterion: The loss function
     Returns:
         err: A scalar for the avg classification error over the validation set
         loss: A scalar for the average loss function over the validation set
     """
    total_loss = 0.0
    total_acc = 0.0
    total_epoch = 0
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)
        corr = (predicted == labels)
        total_acc += int(corr.sum())
        total_loss += loss.item()
        total_epoch += len(labels)
    acc = float(total_acc) / total_epoch
    loss = float(total_loss) / (i + 1)
    return acc, loss

# Training Curve
def plot_training_curve(path):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation error/loss.

    Args:
        path: The base path of the csv files produced during training
    """
    import matplotlib.pyplot as plt
    train_acc = np.loadtxt("{}_train_acc.csv".format(path))
    val_acc = np.loadtxt("{}_val_acc.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    plt.title("Train vs Validation Accuracy")
    n = len(train_acc) # number of epochs
    plt.plot(range(1,n+1), train_acc, label="Train")
    plt.plot(range(1,n+1), val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()
    plt.title("Train vs Validation Loss")
    plt.plot(range(1,n+1), train_loss, label="Train")
    plt.plot(range(1,n+1), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    plt.show()

def train_net(net, batch_size=64, learning_rate=0.01, num_epochs=30):

    train_loader, val_loader, test_loader = get_data_loader(batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-6)

    train_acc = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_acc = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    start_time = time.time()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        total_train_loss = 0.0
        total_train_acc = 0.0
        total_epoch = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Calculate the statistics
            _, predicted = torch.max(outputs, 1)
            corr = (predicted == labels)
            total_train_acc += int(corr.sum())
            total_train_loss += loss.item()
            total_epoch += len(labels)
        train_acc[epoch] = float(total_train_acc) / total_epoch
        train_loss[epoch] = float(total_train_loss) / (i+1)
        val_acc[epoch], val_loss[epoch] = evaluate(net, val_loader, criterion)
        print(("Epoch {}: Train accuracy: {:.2f}, Train loss: {:.4f} |"+
               "Validation accuracy: {:.2f}, Validation loss: {:.4f}").format(
                   epoch + 1,
                   train_acc[epoch] * 100,
                   train_loss[epoch],
                   val_acc[epoch] * 100,
                   val_loss[epoch]))
        # Save the current model (checkpoint) to a file
        model_path = get_model_name(net.name, batch_size, learning_rate, epoch)
        torch.save(net.state_dict(), model_path)
    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
    # Write the train/test loss/err into CSV file for plotting later
    np.savetxt("{}_train_acc.csv".format(model_path), train_acc)
    np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
    np.savetxt("{}_val_acc.csv".format(model_path), val_acc)
    np.savetxt("{}_val_loss.csv".format(model_path), val_loss)
    
'''if __name__ == "__main__":
    model = Model()
    if torch.cuda.is_available():
        model.cuda()
        print('CUDA is available!  Training on GPU ...')
    train_net(model, batch_size=64, learning_rate=0.0001, num_epochs=100)
    model_path = get_model_name("Network", batch_size=64, learning_rate=0.0001, epoch=99)
    plot_training_curve(model_path)
    
    _, _, test_loader = get_data_loader(64)

    total_acc = 0.0
    total_epoch = 0
    for i, data in enumerate(test_loader):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        corr = (predicted == labels)
        total_acc += int(corr.sum())
        total_epoch += len(labels)
    acc = float(total_acc) / total_epoch
    print("test classification ccuracy: {:.2f}".format(acc*100))'''

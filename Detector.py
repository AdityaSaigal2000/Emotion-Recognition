import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.name = "Network"
        self.inference = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 7, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 4 * 4, 79),
            nn.ReLU(),
            nn.Linear(79, 7)
        )

    def forward(self, x):
        return self.inference(x)


def get_data_loader(batch_size):
    data = pd.read_csv('./fer2013.csv')  
    images = data['pixels'].tolist()
    input_data = []
    for image in images:
        temp = [int(pixel) for pixel in image.split(' ')]
        temp = np.asarray(temp).reshape(48, 48)
        input_data.append(temp.astype('float32'))
    input_data = np.asarray(input_data)
    input_data = np.expand_dims(input_data, 1)
#    target = pd.get_dummies(data['emotion']).values
    target = data['emotion'].tolist()
    
    tensor_data = torch.stack([torch.Tensor(i) for i in input_data])
#    tensor_target = torch.stack([torch.Tensor(i) for i in target])
    tensor_target = torch.LongTensor(target)
    dataset = torch.utils.data.TensorDataset(tensor_data, tensor_target)

    # Get the list of indices to sample from
    indices = np.array(range(len(dataset)))
    split1 = 28709
    split2 = 32298
    train_indices, val_indices, test_indices = indices[:split1], indices[split1:split2], indices[split2:]
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               num_workers=0, sampler=train_sampler)
    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              num_workers=0, sampler=val_sampler)
    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              num_workers=0, sampler=test_sampler)
    
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
    total_err = 0.0
    total_epoch = 0
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)
        corr = (predicted != labels)
        total_err += int(corr.sum())
        total_loss += loss.item()
        total_epoch += len(labels)
    err = float(total_err) / total_epoch
    loss = float(total_loss) / (i + 1)
    return err, loss

# Training Curve
def plot_training_curve(path):
    """ Plots the training curve for a model run, given the csv files
    containing the train/validation error/loss.

    Args:
        path: The base path of the csv files produced during training
    """
    import matplotlib.pyplot as plt
    train_err = np.loadtxt("{}_train_err.csv".format(path))
    val_err = np.loadtxt("{}_val_err.csv".format(path))
    train_loss = np.loadtxt("{}_train_loss.csv".format(path))
    val_loss = np.loadtxt("{}_val_loss.csv".format(path))
    plt.title("Train vs Validation Error")
    n = len(train_err) # number of epochs
    plt.plot(range(1,n+1), train_err, label="Train")
    plt.plot(range(1,n+1), val_err, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
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
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    start_time = time.time()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        total_train_loss = 0.0
        total_train_err = 0.0
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
            corr = (predicted != labels)
            total_train_err += int(corr.sum())
            total_train_loss += loss.item()
            total_epoch += len(labels)
        train_err[epoch] = float(total_train_err) / total_epoch
        train_loss[epoch] = float(total_train_loss) / (i+1)
        val_err[epoch], val_loss[epoch] = evaluate(net, val_loader, criterion)
        print(("Epoch {}: Train err: {}, Train loss: {} |"+
               "Validation err: {}, Validation loss: {}").format(
                   epoch + 1,
                   train_err[epoch],
                   train_loss[epoch],
                   val_err[epoch],
                   val_loss[epoch]))
        # Save the current model (checkpoint) to a file
        model_path = get_model_name(net.name, batch_size, learning_rate, epoch)
        torch.save(net.state_dict(), model_path)
    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
    # Write the train/test loss/err into CSV file for plotting later
    np.savetxt("{}_train_err.csv".format(model_path), train_err)
    np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
    np.savetxt("{}_val_err.csv".format(model_path), val_err)
    np.savetxt("{}_val_loss.csv".format(model_path), val_loss)
    
if __name__ == "__main__":
    model = Model()
    if torch.cuda.is_available():
        model.cuda()
        print('CUDA is available!  Training on GPU ...')
        
    train_net(model, batch_size=128, learning_rate=0.001, num_epochs=30)
    model_path = get_model_name("Network", batch_size=128, learning_rate=0.001, epoch=29)
    plot_training_curve(model_path)
    

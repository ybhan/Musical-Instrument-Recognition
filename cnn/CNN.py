# -*- coding: utf-8 -*-
""" By Yuanbo Han, 2018-11-15."""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
import numpy as np

# Load data
data_set = np.load("data_set.npy")
target_set = np.load("target_set.npy")
num_instru = int(np.max(target_set)) + 1

# Hyper parameters
EPOCH = 30
BATCH_SIZE = 200
LEARNING_RATE = 1e-3

# Convert to Tensor
X = torch.from_numpy(data_set)
X = Variable(X.unsqueeze_(1).float())
Y = Variable(torch.from_numpy(target_set))

train_data = data.TensorDataset(X, Y)
train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

#test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.
#test_y = test_data.test_lables[:2000]


# CNN structure
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # Input shape: 1*64*200
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=5,
                      stride=2,
                      padding=0),  # Shape: 16*30*98
            nn.MaxPool2d(kernel_size=2)  # Shape: 16*15*49
        )
        self.full_conn = nn.Sequential(
            nn.Linear(16*15*49, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_instru)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)  # Flatten to (BATCH_SIZE, 16*15*49)
        output = self.full_conn(x)
        return output


cnn = CNN()

# Optimizer
optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

# Loss Function
loss_func = nn.CrossEntropyLoss()


# Training process
for epoch in range(EPOCH):
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)

        # Perform forward pass
        output = cnn(batch_x)

        # Compute loss
        loss = loss_func(output, batch_y)
        # Clear the gradients
        optimizer.zero_grad()
        # Perform backward pass
        loss.backward()
        # Update the parameters
        optimizer.step()

    # Print progress
    if epoch % 5 == 0:
        predicted = torch.argmax(cnn(X), dim=1).view(-1)
        # Calculate and print the training accuracy
        total = Y.size(0)
        correct = predicted == Y
        print('Epoch [%d/%d]  Accuracy: %.4f%%'
              % (epoch, EPOCH, 100*sum(correct.numpy())/total))

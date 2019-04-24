import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.optim as optim
#librerias para cargar datos de manera eficiente
from torch.utils.data import DataLoader, TensorDataset


x = torch.tensor([


   [0, 0, 1, 1],
   [0, 1, 1, 0],
     [1, 0, 1, 0],
     [1, 1, 1, 1]
])


target_y = torch.tensor([0, 1, 1, 0])

inputs = x
labels = target_y

train = TensorDataset(inputs,labels)
#cantidad de valores que se van a evaluar 
train_loader = DataLoader(train,batch_size=4,shuffle=False)


NUMBER_EPOCHS = 10
LEARNING_RATE = 1e-1




class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fun1 = nn.Linear(4,2)
        self.fun2 = nn.Linear(2,1)#modelos lineales
        self.sigmoid = nn.Sigmoid()

    
    def forward(self,x):
        x = self.fun1(x)
        x = self.sigmoid(x)#red neuronal sigmoidea
        x = self.fun2(x)
        x = self.sigmoid(x)

        return x


net = Net()
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)
loss_function = nn.MSELoss()


for epoch in range(NUMBER_EPOCHS):
    trainer_iterator = iter(train_loader)
    for bat_idx , (inputs,labels) in enumerate(trainer_iterator):
        net.zero_grad()
        inputs,labels = Variable(inputs.float()), Variable(labels.float())
        ouput = net(inputs)

        loss = loss_function(ouput,labels)

        loss.backward()
        optimizer.step()

        print(net(Variable(x.float())))

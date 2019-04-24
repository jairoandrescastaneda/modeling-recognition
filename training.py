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
train_loader = DataLoader(train,batch_size=2,shuffle=False)

lineal_model = nn.Linear(4,1)


NUMBER_EPOCHS = 3
LEARNING_ERROR = 1e-4
loss_function = nn.MSELoss()
optimizer = optim.SGD(lineal_model.parameters(),lr=LEARNING_ERROR)

for epoch in range(NUMBER_EPOCHS):
    training_iter = iter(train_loader)
   
    for bat_idx, (inputs,labels) in enumerate(training_iter):
       
        lineal_model.zero_grad()
        inputs,labels = Variable(inputs.float()), Variable(labels.float())
        predictive_y = lineal_model(inputs)
        print(inputs)
        loss = loss_function(predictive_y,labels)
        loss.backward()
        optimizer.step()
        print(lineal_model(Variable(x.float())))
    
    print("_______________________________________")




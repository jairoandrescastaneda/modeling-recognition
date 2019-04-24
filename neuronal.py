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

lineal_model = nn.Linear(4,2)

# es una funcion una neuorona que se encarga de tener enfocado el recalculo de bias and weight
sigmoid = nn.Sigmoid()

#el modelo lineal debe poder hacer calculos
lineal_model_2 = nn.Linear(2,1)

NUMBER_EPOCHS = 100000
LEARNING_RATE = 1e-1
optimizer = optim.SGD(lineal_model.parameters(), lr=LEARNING_RATE)
loss_function = nn.MSELoss()


for epoch in range(NUMBER_EPOCHS):
  training_iterator = iter(train_loader)
  
  for bat_id, (inputs,labels) in enumerate(training_iterator):
    lineal_model.zero_grad()
    inputs,labels = Variable(inputs.float()),Variable(labels.float())
    #se calcula el modelo lineal
    model_lineal1_ouput = lineal_model(inputs)
    #se le pasa la ecuacion de la red neuronal de sigmoid
    simoid_oput = (model_lineal1_ouput)
  
    #vuelve y se calcula el modelo lineal 
    model_lineal2_ouput = lineal_model_2(simoid_oput)
    sigmoid_ouput2 = sigmoid(model_lineal2_ouput)

    #la perdida de funcion

    loss = loss_function(sigmoid_ouput2,labels)
    loss.backward()
    optimizer.step()

    print(sigmoid(lineal_model_2(sigmoid(lineal_model(Variable(x.float()))))))
 


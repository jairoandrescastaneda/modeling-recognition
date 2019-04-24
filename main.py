import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.optim as optim

x1 = torch.Tensor([1,2,3,4])
model_neuranal = nn.Linear(4,1)
#Variable encapsula lo mejora cosas que no hace un tensor comun
x1_ar = Variable(x1,requires_grad=True) #encapsula para faciltiar el tema de calculo de gradiente 
 # es necesario realizar la gradiente para poder calcular el loss function
target_y = Variable(torch.Tensor([0]),requires_grad=False)
 #a calcular la perdida de diferencia entre el valor lineal predecido y el valor que espero que me de
optimizer = optim.SGD(model_neuranal.parameters(),lr=1e-4)

NUMBER_EPOCHS  = 3
LEARNING_RATE = 1e-4
for epoch in range(NUMBER_EPOCHS):
    #cuando se hacen calculos dentro de ciclos es importante hacer set ala gradiente
    model_neuranal.zero_grad()

    #se predice el valor utilizando un modelo lineal
    predictive_y = model_neuranal(x1_ar)
    lossfunction = nn.MSELoss()
    loss = lossfunction(predictive_y,target_y)
    loss.backward()
    optimizer.step()
    print(model_neuranal(x1_ar))



#antes cambiar el wight y el bias se calcula la gradiente  de la loss function
#loss.backward()



#optimizer.step()

#print(model_neuranal(x1_ar))


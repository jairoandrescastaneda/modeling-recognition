import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#librerias para cargar datos de manera eficiente
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt # for plotting
from PIL import Image


#Reglas para cada tensor de imagenes
tranformada = transforms.Compose([


    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

tranformada2 = transforms.Compose([
transforms.Resize((32,32)),

    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

traninset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=tranformada)
trainsetloader = DataLoader(traninset,batch_size=4,shuffle=False)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tranformada)
tesloader = DataLoader(testset, batch_size=4, shuffle=False)
"""
cargo la propia data desde mi carpeta pero debe estar dentro de subfolders que se supone es la posicion de la clase , sim embargo
como no es de entrenamiento no se necesitan los labels, mas abajo se eliminan en el ciclo con el _
"""

my_data = torchvision.datasets.ImageFolder('./images',transform=tranformada2)
mydataloader = DataLoader(my_data,batch_size=4,shuffle=False)


#mostrar imagen
def imshow(img):
    print(img.size())
    img = img / 2 + 0.5   
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

"""
dataiter = iter(trainsetloader)
images,labels = dataiter.next()


"""


#imshow(torchvision.utils.make_grid(images))
"""


#4 por que devuelve la iteracion en batch_size
for j in range(4):
    print(str(classes[labels[j]]))

"""

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        #self.conv1 = nn.Conv2d(3,6,5)#son 3 canales rgb ahora 6 canales usando kernel de 5X5, obitiene las partes importantes de una imagen
        self.conv1 = nn.Conv2d(3,10,5)
        self.pool = nn.MaxPool2d(2,2) # crear una matriz de 2X2 y sacar el maximo valor de ese pixel eso entregara uno por cada 4 valores
        #self.fun1 = nn.Linear(1176,120)#25 modelos escondidos
        #self.fun2 = nn.Linear(120,10)#modelos lineales
        #self.sigmoid = nn.Sigmoid()
        self.fun1 = nn.Linear(20*5*5,120)
        self.fun2 = nn.Linear(120,10)
        self.conv2 = nn.Conv2d(10,20,5)

    
    def forward(self,x):
        #reajuste para entrar el modelo lineal, el -1 es un reajuste que se adapta alas demas
        #fin de unificar y hacer comparaciones mas eficnete
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

    
       # x = x.view(-1,1176)#multiplicacion matricial 
        x = x.view(-1,20*5*5)
        
        x = self.fun1(x)
        x = F.relu(x)
       
        #x = self.sigmoid(x)#red neuronal sigmoidea
        x = self.fun2(x)
        x = F.relu(x)

        #x = self.sigmoid(x)

        return x

net = Net()

NUMBER_EPOCHS = 1
LEARNING_RATE = 1e-2
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=LEARNING_RATE)






#print(images_t.size())



for epoch in range(NUMBER_EPOCHS):
    train_loader_iter = iter(trainsetloader)

    for bat_idx , (inputs,labels) in enumerate(train_loader_iter):
       
        net.zero_grad()
        #print(inputs[0].size())
        inputs,labels = Variable(inputs.float()), Variable(labels)
        #print(labels)
        ouput = net(inputs.float())
        loss = loss_function(ouput.float(),labels)
        loss.backward()
        optimizer.step()
        
    if epoch%5 is 0:
        print('iteration :'+ str(epoch+1))



#dataiter = iter(tesloader)
#images,labels = dataiter.next()

#ouputs = net(Variable(images_test[0]))
#_,predictec = torch.max(ouputs.data,1)



for data,label in mydataloader:
    
    ouputs = net(Variable(data))
    _,predictec = torch.max(ouputs.data,1)
    for i in range(4):
        print(str(classes[predictec[i]]))
    
    imshow(torchvision.utils.make_grid(data))
    






print('Deberia ------')





from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,10,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(10,20,5) # 10 los canales que recibe , 20 los que devuelve y 5 es el filter size
        self.fun1 = nn.Linear(20*72*72,512) # valor de las redes neuronales ocultas debe ser minima
        self.fun2 = nn.Linear(512,4)#las redes de salida
        
        #modelo matematico es la red neuronal
    def forward(self,x):
        x = self.conv1(x) # aplica un modelo convolucioal convirtiendo 3 canales a 10 canales , pinta o resalta, si cambia el tamaño - Reduccion dada por formula
        x = F.relu(x) #funcion de activacion
        x = self.pool(x) # sacar los pixeles que son mayores si cambia el tamaño , es dividir en 4 los pixeles
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1,20*72*72) # el tamño del tensor que viene
        x = self.fun1(x)
        x = F.relu(x) #multpilicar por una funcion para acercar los valores a valores razonables
        x = self.fun2(x)

        
        

        return x

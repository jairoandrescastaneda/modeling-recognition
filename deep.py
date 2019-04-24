#import pdb
#pdb.set_trace()
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
from redNeuronal import Net


#siempre hacer el resize primero o da error 
tranformadaTraining = transforms.Compose([
     transforms.Resize((300,300)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

dataTraining = torchvision.datasets.ImageFolder('./entrenamiento',transform=tranformadaTraining)
dataTrainingLoader = DataLoader(dataTraining,batch_size=4,shuffle=False)

#data de testeo

dataTesting = torchvision.datasets.ImageFolder('./testeo',transform=tranformadaTraining)
dataTestingLoader = DataLoader(dataTesting,batch_size=4,shuffle=False)

#la posicion de la carpeta hace referenica ala posicion de la tupla
clases = ('vive100','harinapan','arrozgelvez','televisor')
#creo un dispositivo de prueba y entrenamiento 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
redNeuronal = Net().to(device)



def mostrarImagen(img):
   
    img = img / 2 + 0.5   
    npimg = img.numpy()

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def salidaConsola(numeroAcierto,numeroImagen):
    porcentajeAcierto = (numeroAcierto/numeroImagen)*100
    print('         Resultados        ')
    print('Aciertos : '+str(numeroAcierto))
    print('Porcentaje de aciertos '+str(porcentajeAcierto))


def saveModel():
    torch.save(redNeuronal.state_dict(),'./modelo/modelo1.pt')

def loadModel():
    ubicacion = 'modelo1.pt'
    try:
        
        redNeuronal.load_state_dict(torch.load('./modelo/'+ubicacion))
        redNeuronal.eval()
    except FileNotFoundError:
        print('No existe el archivo  '+ubicacion)





def entrenamientoData():
    #datos basicos y red neuronal
    
    NUMBER_EPOCHS = 10000
    LEARNING_RATE = 1e-2
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.SGD(redNeuronal.parameters(),lr=LEARNING_RATE)
    cantidadLossCalculado = 0
    lossTotal = 0
    # proceso de entramiento
    for epoch in range(NUMBER_EPOCHS):
        dataTrainingIter = iter(dataTrainingLoader)
        for data,labels in dataTrainingIter:
            redNeuronal.zero_grad()
            data,labels = Variable(data.float().to(device)),Variable(labels.to(device))
            output = redNeuronal(data)
            loss = lossFunction(output,labels)
            loss.backward()# Gradiente en funcion de los datos que da loss function 
            optimizer.step()
            lossTotal+=loss.item()
            cantidadLossCalculado+=1

        #cada 5 ciclos da un informe
        if epoch%5==0:
            promedioPerdida = (lossTotal/cantidadLossCalculado)
            print('loss Promedio '+str(promedioPerdida))
            print('iteracion de aprendizaje '+str(epoch+1))
            lossTotal = 0
            cantidadLossCalculado = 0






def testeoData():
    
    dataTestingIterator = iter(dataTestingLoader)
    cantidadImagen = 0
    cantidadAcierto = 0
   
    for data,labels in dataTestingIterator:
   
        output = redNeuronal(Variable(data.to(device)))
        labels = labels.to(device)

        _,prediccion = torch.max(output.data,1)#Revisar que devuelve torchmax
        #Size devuelve tipo de dato tensor
        cantidadData = list(data.size())
    
        print('Las imagenes analizadas son:')
        #para darle formato a los batches
        for i in range(cantidadData[0]):
            cantidadImagen+=1
            if prediccion[i].item()==labels[i].item():
                cantidadAcierto+=1
        
            print(clases[prediccion[i]])
    
        mostrarImagen(torchvision.utils.make_grid(data))
        print('--------------------------------')

    salidaConsola(cantidadAcierto,cantidadImagen)


 
    

#loadModel()
entrenamientoData()
testeoData()
saveModel()


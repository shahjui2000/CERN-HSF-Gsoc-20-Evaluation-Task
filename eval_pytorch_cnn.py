import numpy as np
from sklearn import metrics as metrics
import matplotlib.pyplot as plt

##############################LIBRARIES########################################################
from sys import exit
from time import gmtime, strftime
try:
    import numpy as np
except:
    print("Failed to import numpy.")
    exit()
try:
    import matplotlib.pyplot as plt
except:
    print("Failed to import matplotlib.")
    exit()
try:
    from sklearn.model_selection import train_test_split as tts
except:
    print("Failed to import from sklearn.")
    exit()
try:
    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.autograd import Variable
    import torch.optim as optim
except:
    print("Cannot fetch/detect PyTorch.")
    exit()   
try:
    import pandas as pd
except:
    print("Failed to import Pandas.")
    exit()
#################################LOADING THE DATA##############################################
signal = np.load('./pytorch_signal.npy').reshape(10000,32,32)
background = np.load('./pytorch_background.npy').reshape(10000,32,32)

X1, X_test1, y1, y_test1 = tts(signal, np.zeros((10000,1)), test_size = 0.2)
X2, X_test2, y2, y_test2 = tts(background, np.ones((10000,1)), test_size = 0.2)

X_train = np.concatenate((X1, X2), axis=0)
y_train = np.concatenate((y1, y2), axis=0)

X_test = np.concatenate((X_test1, X_test2), axis=0)
y_test = np.concatenate((y_test1, y_test2), axis=0)

#################################IMAGES########################################################
img = signal[862,:,:]   
plt.figure(1)
plt.imshow(np.transpose(img))
plt.savefig('Class 0')

img = background[908,:,:]    
plt.figure(2)
plt.imshow(np.transpose(img))
plt.savefig('Class 1')
plt.show()
####################################DATALOADER#################################################
class dataset_class(Dataset):

    def __init__(self, x, y, train=True):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        # a = self.x[i:i+1000,:,:]
        return self.x[index,:], self.y[index,:]
    
    def __len__(self):
        return int((self.x.shape[0]))

trainset = dataset_class(X_train, y_train)
testset = dataset_class(X_test, y_test)

trainloader = DataLoader(dataset = trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = DataLoader(dataset = testset, batch_size=128, shuffle=False, num_workers=2)

sigmoid = nn.Sigmoid()
######################################CLASS####################################################
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 32, 3)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(21632, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flat(x)
        # print(x.shape)
        x = F.softmax(self.fc1(x))
        return x
#################################INTIALIZERS##################################################
net = CNN()
net = net.double()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.1)
nepochs = 10
###################################TRAINING####################################################
isTrain = True

PATH = './pytorch_cnn_net.pth'

if(isTrain):
    net = net.train()
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    for epoch in range(nepochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
           
            inputs, labels = data
            
            inputs = inputs.reshape(inputs.shape[0],1,32,32)

            inputs, labels = Variable(inputs), Variable(labels)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.double())
            # print(outputs.shape)
            # print(labels.shape)
            loss = criterion(outputs, labels[:,0].type(torch.long))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # if i % 1000 == 999:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.18f' %
            (epoch + 1, i + 1, running_loss /125))


    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

    torch.save(net.state_dict(), PATH)
    print('Finished Training')
#####################################TESTING###################################################
net.load_state_dict(torch.load(PATH))
predicted = np.array(([4]))

print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
with torch.no_grad():
    for data in trainloader:
        images, labels = data
        
        images = images.reshape(images.shape[0],1,32,32)
        images, labels = Variable(images), Variable(labels)
           
        outputs = net(images)
        _, pred_label=torch.max(outputs.data, dim=1)
        
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        
        images = images.reshape(images.shape[0],1,32,32)
        images, labels = Variable(images), Variable(labels)
           
        outputs = net(images)
        # print(predicted.shape)
        _, pred_label =torch.max(outputs.data, dim=1)
        # print(np.array(pred_label).T.shape)
        predicted = np.concatenate((predicted, np.array(pred_label).T))
       
print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

predicted = predicted[1:]
######################################METRICS##################################################
from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted)
plt.figure(1)
plt.plot(fpr, tpr)
plt.grid()
plt.show()
print('AUC score:',end = '')
print(metrics.roc_auc_score(y_test, predicted))

print('\n\nPrecision Recall F_score: ' , end='')
print(metrics.precision_recall_fscore_support( y_test, predicted , average='binary'), end='\n')
print('\n\nAccuracy: ', end ='')
print(metrics.accuracy_score( y_test, predicted)*100,end ='%')
# ###############################################################################################

#!/usr/bin/env python

from ROOT import TMVA, TFile, TTree, TCut
from subprocess import call
from os.path import isfile
import sys
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, Reshape
from keras.regularizers import l2
from keras.optimizers import SGD

# Setup TMVA
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

output = TFile.Open('TMVA_CNN_PyKeras.root', 'RECREATE')
factory = TMVA.Factory('TMVAClassification', output,
                       '!V:!Silent:Color:DrawProgressBar:Transformations=None:AnalysisType=Classification')

############################Loading the data file
data = TFile.Open("/home/jui/Desktop/tmva/sample_images_32x32.root")
# print(data.ls())
signal = data.Get('sig_tree;2')
background = data.Get('bkg_tree;2')

dataloader = TMVA.DataLoader('dataset_evaltest')

imgSize = 1024

dataloader.AddVariablesArray("vars",imgSize)

dataloader.AddSignalTree(signal, 1.0)
dataloader.AddBackgroundTree(background, 1.0)
dataloader.PrepareTrainingAndTestTree(TCut(''),
                                      'nTrain_Signal=8000:nTrain_Background=8000:SplitMode=Random:NormMode=NumEvents:!CalcCorrelations:!V')

# Generate model

# Define model
model = Sequential()
model.add(Reshape((32, 32,1), input_shape=(1024,)))
model.add(Conv2D(6, kernel_size=5, activation='relu', input_shape = (32,32,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

# Set loss and optimizer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',])

# Store model to file
model.save('model_2.h5')
model.summary()

# Book methods
factory.BookMethod(dataloader, TMVA.Types.kPyKeras, 'PyKeras',
                   'H:!V:VarTransform=None:FilenameModel=model_2.h5:NumEpochs=10:BatchSize=128')

# Run training, test and evaluation
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()

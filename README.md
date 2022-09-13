# CNN
PyKeras-TMVA, PyTorch, TMVA on dataset loaded from https://cernbox.cern.ch/index.php/s/mba2sFJ3ugoy269



This repo attempts classification using CNN in 3 different ways (but the same architecture):

TMVA: ClassificationCNN.C

Toolkit for Multivariate Analysis - It is the root integrated environment of CERN that performs the ML task directly on root images.


PyKeras-TMVA: ClassificationCNNKeras.py

PyKeras TMVA is the code that integrates Keras implementation into CERN TMVA and excutes classification.


PyTorch: eva_pytorch_cnn.py

For pytorch, first converted root file to numpy array and saved it using root2array.py
PyTorch framework is used to compute classification of the converted numpy array format images.

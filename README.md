# CNN
PyKeras-TMVA, PyTorch, TMVA

For pytorch, firt converted root file to numpy array and saved it using the following python code lines:

##################################################################


from ROOT import TMVA, TFile, TTree, TCut

import numpy as np


data = TFile.Open("/home/jui/Desktop/tmva/sample_images_32x32.root")

print(data.ls())


sig = data.Get('sig_tree;2')

bkg = data.Get('bkg_tree;2')


signal = np.asarray([[sig.vars] for event in sig]).reshape(10000,32,32)

background = np.asarray([[bkg.vars] for event in bkg]).reshape(10000,32,32)


np.save('pytorch_signal',signal)

np.save('pytorch_background',background)

#################################################################




#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"

#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Tools.h"

int ClassificationCNN(TString myMethodList = "")
{
    TMVA::Tools::Instance();
    std::cout << "==> Start TMVAClassification using CNN" << std::endl;
   
    TFile *input(0);
    TString fname = "/home/jui/Desktop/tmva/sample_images_32x32.root";
    input = TFile::Open( fname ); 

    std::cout << "--- TMVAClassification       : Using input file: " << input->GetName() << std::endl;

    TTree *signalTree     = (TTree*)input->Get("sig_tree;2");
    TTree *background     = (TTree*)input->Get("bkg_tree;2");

    // Create a ROOT output file where TMVA will store ntuples, histograms, etc.
    TString outfileName( "TMVA_CNN.root" );
    TFile* outputFile = TFile::Open( outfileName, "RECREATE" );

    TMVA::Factory *factory = new TMVA::Factory( "ClassificationCNN", outputFile,
                                            "!V:!Silent:Color:DrawProgressBar:Transformations=None:AnalysisType=Classification" );

    TMVA::DataLoader *dataloader=new TMVA::DataLoader("dataset");

    dataloader->AddVariablesArray("vars",1024);

    // Double_t signalWeight     = 1.0;
    // Double_t backgroundWeight = 1.0;

    // You can add an arbitrary number of signal or background trees
    dataloader->AddSignalTree    ( signalTree );
    dataloader->AddBackgroundTree( background );
    // dataloader->SetBackgroundWeightExpression( "weight" );

    dataloader->PrepareTrainingAndTestTree( "", "",
                                        "nTrain_Signal=8000:nTrain_Background=8000:SplitMode=Random:NormMode=NumEvents:!CalcCorrelations:!V");



      // General layout.
      TString inputLayoutString("InputLayout=1|32|32");
      TString batchLayoutString("BatchLayout=128|1|1024");

       TString layoutString("Layout=CONV|6|5|5|1|1|0|0|RELU,CONV|32|3|3|1|1|0|0|RELU,"
                     "RESHAPE|FLAT,DENSE|1|LINEAR");
                                                                     
      // Training strategies.
      TString training0("LearningRate=1e-3,Momentum=0.9,Repetitions=1,"
                     "ConvergenceSteps=10,BatchSize=128,TestRepetitions=1,"
                     "MaxEpochs=10,WeightDecay=1e-4,Regularization=None,"
                     "Optimizer=ADAM");
 
    TString trainingStrategyString ("TrainingStrategy=");
    trainingStrategyString += training0; // + "|" + training1 + "|" + training2;   }

    // General Options.                                                                                                                              
    TString cnnOptions ("!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=None:"
                        "WeightInitialization=XAVIERUNIFORM");

    cnnOptions.Append(":"); cnnOptions.Append(inputLayoutString);
    cnnOptions.Append(":"); cnnOptions.Append(batchLayoutString);
    cnnOptions.Append(":"); cnnOptions.Append(layoutString);
    cnnOptions.Append(":"); cnnOptions.Append(trainingStrategyString);
    cnnOptions.Append(":Architecture=CPU");
    factory->BookMethod(dataloader, TMVA::Types::kDL, "DL_CNN", cnnOptions);
    
    factory->TrainAllMethods();
    factory->TestAllMethods();
    factory->EvaluateAllMethods();



    c1 = factory.GetROCCurve(dataloader);
    c1->Draw();


    outputFile->Close();

    std::cout << "==> Wrote root file: " << outputFile->GetName() << std::endl;
    std::cout << "==> TMVACNNClassification is done!" << std::endl;

    delete factory;
    delete dataloader;
    return 0;
}
int main( int argc, char** argv )
{
    TString methodList = "CNN";
    return ClassificationCNN(methodList);
}


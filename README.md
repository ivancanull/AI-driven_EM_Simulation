# AI-driven_EM_Simulation

## Data Preparation
Unpack `Data/Tml_sweep.7z` or Tml_sweep data of other formats into `Data/Tml_sweep`

## model selection
This experiment will test several S parameters prediction model on different NN models and other settings to decide which model is optimal.

## SoTA model
After the initial experiment on partial S parameters, this notebook will train all the S parameters with the State of The Art Model. The models' parameters are saved in `Models/SoTA_model/`.

## data analysis
This notebook is used after SoTA models are trained. It will analyze the prediction error on the test data and plot the curves of some test cases, including the worst and best test cases.
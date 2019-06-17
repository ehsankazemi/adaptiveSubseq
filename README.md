# adaptiveSubseq


The code accompanying the paper entitled "Adaptive Sequence Submodularity".
https://arxiv.org/abs/1902.05981\

This code is only for the Amazon application. The Wikipedia application requires a 1GB file to run and was thus not included.

To run, just write (in command line):

python amazonMain.py 


amazonFunction.py contains all the important function definitions (i.e. algorithms and everything).
amazonMain.py is the main file to look at, here you can edit the hyperparameter settings and get the graphs. 

Default hyperparameter settings (to get graphs in Figure 2b and 2c):

path = 'videoGames.csv' #File we read from.
m = 50 #Minimum number of reviews for a product to be considered.
percTrain = 0.8 #Fraction of data to be used for training.
given = 4 #Array for the different number of given products to be given to each algorithm at start.
k = 6 #Number of products to be predicted.
thresh = 0.05 #In Adaptive Sequence Greedy, only consider edges with this value (just makes code faster).
inputSteps = given #Number of input steps to LSTM is equal to number of given items.
numTrials = 5 #Number of trials to run for experiments (each trial is a new split of training and testing).
numNodes = 256 #Number of nodes in feed-forward network hidden layer.
numLSTM = 8 #Number of LSTM nodes in LSTM hidden layer.

To get the graphs in Figure 2e and 2f set: percTrain = 0.01

Note that if percTrain = 0.8, it will take about 20-30 minutes per trial (runtime of each algorithm will be output as it finishes). If you set percTrain = 0.01, it should only take about 1 minute per trial.

Note that this code requires keras and matplotlib to run. It is possible that the way I import matplotlib doesn't work the same way on other machines. 

——————————

adaptiveSubseq_notebook.ipynb is a jupyter notebook that runs the same code. Depending on your preference it may be easier to follow/understand.

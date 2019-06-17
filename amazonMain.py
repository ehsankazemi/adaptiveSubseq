#=========
# Imports
#=========

from amazonFunctions import * #Import function definitions from other file.

import matplotlib #Plotting tools.
matplotlib.use('TkAgg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

#==========================
# Main function definition
#==========================

def runAlgorithms():
	
	nn = np.zeros((2,k)) #Arrays to hold results of various algorithms.
	nnAdaptive = np.zeros((2,k))
	lstm = np.zeros((2,k))
	lstmAdaptive = np.zeros((2,k))
	ASG = np.zeros((2,k))
	SG = np.zeros((2,k))
	freq = np.zeros((2,k))

	for i in range(0,numTrials):
		print ('------\n','Trial Number',i+1)
		trainDic,testDic = trainTestSplit(users,percTrain,productMap)
		trainX,trainY = formatNeuralNetworkData(users,productMap,trainDic) #Format training/testing data for the feed forward NN.
		trainXlstm,trainYlstm = formatLSTMData(users,productMap,trainDic,inputSteps) #Format training/testing data for the LSTM.
	
		start = time.time()
		p1,p2 = productDic(users,trainDic,thresh) #Train the graph for Adaptive Sequence Greedy (and sequence-greedy).
		print ('Train graph:',time.time()-start,'seconds')
	
		start = time.time()
		model = trainNetwork(trainX,trainY,numNodes)
		print ('Train feed-forward network:',time.time()-start,'seconds')
	
		start = time.time()
		nn += testNetwork(users,productMap,idMap,testDic,model,given,k)/numTrials
		print ('Test feed-forward network:',time.time()-start,'seconds')
	
		start = time.time()
		nnAdaptive += testNetworkAdaptive(users,productMap,idMap,testDic,model,given,k)/numTrials
		print ('Test adaptive feed-forward network:',time.time()-start,'seconds')
		
		start = time.time()
		model = trainLSTM(trainXlstm,trainYlstm,numLSTM)
		print ('Train LSTM:',time.time()-start,'seconds')
	
		start = time.time()
		lstm += testLSTM(users,productMap,idMap,testDic,model,given,k,inputSteps)/numTrials
		print ('Test LSTM:',time.time()-start,'seconds')
	
		start = time.time()
		lstmAdaptive += testLSTMAdaptive(users,productMap,idMap,testDic,model,given,k,inputSteps)/numTrials
		print ('Test adaptive LSTM:',time.time()-start,'seconds')
	
		start = time.time()
		ASG += adaptive(p1,p2,users,testDic,given,k)/numTrials
		print ('Test Adaptive Sequence Greedy:',time.time()-start,'seconds')
	
		start = time.time()
		SG += nonadaptive(p1,p2,users,testDic,given,k)/numTrials
		print ('Test Sequence Greedy:',time.time()-start,'seconds')
	
		start = time.time()
		freq += nonsequence(p1,p2,users,testDic,given,k)/numTrials
		print ('Test Frequency:',time.time()-start,'seconds')

# 	print nnAdaptive.tolist()
# 	print nn.tolist()
# 	print lstmAdaptive.tolist()
# 	print lstm.tolist()
# 	print ASG.tolist()
# 	print SG.tolist()
# 	print freq.tolist()
	
	#Graph the results!
	graphResults(nnAdaptive,nn,lstmAdaptive,lstm,ASG,SG,freq,k,m,percTrain) 

#====================
# Variable settings
#====================

path = 'videoGames.csv' #File we read from.
m = 50 #Minimum number of reviews for a product to be considered.
percTrain = 0.01 #Fraction of data to be used for training. Use 0.01 to get Figures 2e and 2f, and 0.8 for 2b and 2c.
given = 4 #Array for the different number of given products to be given to each algorithm at start.
k = 6 #Number of products to be predicted.
thresh = 0.05 #In Adaptive Sequence Greedy, only consider edges with this value (just makes code faster).
inputSteps = given #Number of input steps to LSTM is equal to number of given items.
numTrials = 5 #Number of trials to run for experiments (each trial is a new split of training and testing).
numNodes = 256 #Number of nodes in feed-forward network hidden layer.
numLSTM = 8 #Number of LSTM nodes in LSTM hidden layer.

#======
# Main 
#======

arr = importData(path) #Import the data
users = userDic(arr) #Parse out the sequence for each user.

productMap,idMap = removeProducts(users,m) #productMap is a dictionary that maps each remaining product (after removing all products with fewer than m reviews) to a unique integer ID; idMap is an array that we use to map the integer ID back to the product ID.
print ("Number of products:",len(productMap))
print ("Number of users:",len(users))

runAlgorithms() #Main function that runs the algorithms for the desired number of trials and then outputs the results.



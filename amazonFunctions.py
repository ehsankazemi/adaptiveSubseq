#=========
# Imports
#=========

import time #Basic python libraries.
import random
import numpy as np

from keras.models import Sequential #Import keras libraries.
from keras.layers import LSTM
from keras.layers import Dense
from keras.utils import np_utils
import keras.callbacks

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #Suppress tensorflow warnings.

import matplotlib #Plotting tools.
matplotlib.use('TkAgg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

#=============================
# Data Loading and Formatting
#=============================

def importData(path):
	#Read in data from given .csv file
	#Format of each line is: userID,productID,unixTimestamp
	#Output is an array where each entry is [userId,productId,unixTimestamp]
	
	start = time.time()
	arr = []
	print ("Importing data...")
	with open(path, 'r') as f:
		content = f.readlines()		
		for row in content: #For each line in the file.
			
			row2 = row.split(',') #Split up by commas.
			temp = [row2[0],row2[1],int(row2[2])] #Turn the timestamp into an int (from a string).
			arr.append(temp)
			
	#print 'Finished in: ',time.time()-start	
	return arr
	
def userDic(arr):
	#Group activity by user.
	#i.e. dic[userID] contains all the products reviewed by that user,
	#where each entry is [productID,unixTimestamp]
	#For the final output, we sort by the unixTimestamp and then drop the time so that 
	#each dic[usedID] is a list of products in the order they were reviewed.
	
	print ('Building userDic...')
	dic = {}
	for i in range(0,len(arr)):
		userID = arr[i][0]
		productID = arr[i][1]
		unixTime = arr[i][2]
		
		if userID not in dic:
			dic[userID] = []
		dic[userID].append([productID,unixTime])
	
	#sort each array by unixTimestamp
	for userID in dic:
		temp = sorted(dic[userID], key=lambda x: x[1]) #Sort by unixTimestamp.
		dic[userID] = [item[0] for item in temp] #Drop the timestamp and keep only productID.
		
	return dic

def removeProducts(users,m):
	#Remove all products with fewer than m reviews.
	#Return dictionary such that dic[pID] = ind, where 'pID' is the product ID and 'ind' is a unique integer index for that product.

	countP = {} #countP[pID] counts the number of times the product 'pID' is reviewed.
	for uid in users:
		for entry in users[uid]:
			if entry not in countP:
				countP[entry] = 0.0
			countP[entry] += 1
	
	rmP = {} #List of products to remove.
	count = 0
	dic = {} #The aforementioned dictionary that basically maps each product to a unique integer index.
	idArr = [] #Array so we can map the ID back to the product
	for pid in countP:
		if countP[pid] < m:
			rmP[pid] = 0
		else:
			dic[pid] = count
			idArr.append(pid)
			count += 1
	
			
	for uid in users:
		temp = []
		for pid in users[uid]: #For each product that the user 'uid' reviewed:
			if pid not in rmP: #If the product is not in the list of products to be removed, add it to our temp list.
				temp.append(pid)
		users[uid] = temp #Update the list of products that user 'uid' reviewed.
	
	return dic,idArr

def trainTestSplit(users,percTrain,productMap):
	#Given a dictionary of users where users[uid] gives the sequence of movies reviewed by user uid,
	#Split the data into a training and testing set:
	#i.e. trainDic contains all the users in the train set, testDic is all users in test set.
	#percTrain is fraction of data we want to use for training.

	trainDic = {} #Dictionary of training points
	testDic = {} #Dictionary of testing points.
	
	numProds = len(productMap) #Total number of products.
	numUser = len(users) #Total number of users.
	
	userList = [] #We want to partition the data into a train set and a test set, so first make a list of all users.
	for uid in users:
		userList.append(uid)
	random.shuffle(userList) #Randomize the order
	
	numTrain = int(numUser*percTrain) #Number of training points.
	trainList = userList[:numTrain] #First part of shuffled list is training set.
	testList = userList[numTrain:] #Second part is test set.
	
	for uid in trainList: 
		trainDic[uid] = 1
	for uid in testList:
		testDic[uid] = 1
		
	return trainDic,testDic	
	
#=====================================
# Neural network training and testing 
#=====================================

def formatNeuralNetworkData(users,productMap,trainDic):
	#To be used to create training/testing data for the neural network.
	
	#users: dictionary where user[uid] is the ordered list of products reviewed by user 'uid'.
	#productMap: dictionary that maps each product ID to a unique integer between 0 and the total number of products.
		
	trainX = [] #Input values. A vector where the index = 1 if that corresponding product has been reviewed, and 0 otherwise.
	trainY = [] #Output labels. A  one-hot encoded vector for the next product to be reviewed.
	numProds = len(productMap)
	for uid in trainDic:
		for i in range(1,len(users[uid])): 
			xpids = users[uid][:i] #First i movies reviewed by user 'uid'
			ypids = users[uid][i] #Next movie user 'uid' reviewed.
		
			tempX = np.zeros(numProds) #Set vector to 0 for all products
			for pid in xpids:
				ind = productMap[pid] #Find unique integer index of product 'pid'
				tempX[ind] = 1 #Set the corresponding index for 'pid' to 1.
		
			tempY = np.zeros(numProds)
			ind = productMap[ypids]
			tempY[ind] = 1 #One-hot encoded vector where all is 0 except index corresponding to next product to be reviewed.
	
			trainX.append(tempX)
			trainY.append(tempY)
	
	trainX = np.array(trainX)
	trainY = np.array(trainY)

	return trainX,trainY

def formatLSTMData(users,prodMap,trainDic,inputSteps):
	#To be used for training/testing data for LSTM neural network.

	# reshape input to be [samples, time steps, features]

	trainX = [] #Input values. A vector where the index = 1 if that corresponding product has been reviewed, and 0 otherwise.
	trainY = [] #Output labels. A  one-hot encoded vector for the next product to be reviewed.
	numProds = len(prodMap)
	for uid in trainDic:
		
		current = ['x']*(inputSteps-1) + users[uid][:] #Deep copy of current path with 'x's padded at start (to ensure we get the desired amount of input steps).
		
		for i in range(0,len(current)-inputSteps):
			tempX = []
			for j in range(i,i+inputSteps):
				temp = np.zeros(numProds)
				if current[j] != 'x': #'x' means padding, so just append array of all zeros.
					ind = prodMap[current[j]] #otherwise, append one-hot encoded array (all 0s and 1 in index of corresponding page).
					temp[ind] = 1
				tempX.append(temp)
			
			tempY = np.zeros(numProds)
			ind = prodMap[current[i+inputSteps]]
			tempY[ind] = 1
				
			tempX = np.array(tempX) #Turn into numpy array.
			tempY = np.array(tempY)
			
			trainX.append(tempX)
			trainY.append(tempY)
			
	trainX = np.array(trainX)
	trainY = np.array(trainY)


	return trainX,trainY

def topK(arr,k,seq,idMap):
	#Given a softmaxed prediction vector 'arr', return the product IDs of the k highest values that are not already in 'seq'.
	#Helper function for testing non-adaptive feed-forward network and LSTM.
	
	best = [] #holds the value and index of k highest value indices in arr.
	for i in range(0,k):
		best.append([0,0]) #[value,index]
		
	for i in range(0,len(arr)):
		if i not in seq and arr[i] > best[0][0]: #Make sure this element has high enough value and is not already in the sequence.
			best[0][0] = arr[i] #Replace first (and lowest value) entry in 'best' array.
			best[0][1] = i
			
			best = sorted(best, key = lambda x: x[0]) #Re-sort so that lowest value entry is in first position (to be replaced, if appropriate).
			
	pids = [] #Holds sorted (from highest to lowest) list of product IDs (so we can compare our guess against the truth).
	for i in range(k-1,-1,-1): #Go from highest value to lowest.
		pids.append(idMap[best[i][1]]) #Map the integerID to the product ID.
		
	return pids				

def trainNetwork(trainX,trainY,numNodes):
	
	#Code to train simple feed forward neural network with one hidden layer of 'numNodes'.
	
	#Adjust batch size (bs) depending on how much training data we have.
	#i.e. in the low data regime use a batch size of 32, otherwise use 1024.
	if len(trainX) < 2000:
		bs = 32
	else:
		bs = 1024
	
	model = Sequential()  
	model.add(Dense(numNodes, activation = 'relu', input_dim=trainX.shape[1]))
	model.add(Dense(trainY.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	
	early_stopping_monitor = keras.callbacks.EarlyStopping(monitor = 'val_loss',min_delta = 0.01, patience = 1, verbose = 0, mode = 'auto') #Code for early stopping of training.
	model.fit(trainX, trainY, epochs=100, batch_size=bs, validation_split = 0.2, verbose = 0, shuffle = True,	callbacks = [early_stopping_monitor])
	return model		

def trainLSTM(trainX,trainY,numNodes):
	#Code for training LSTM with 'numNodes' LSTM neurons in one hidden layer.
	
	#Adjust batch size (bs) depending on how much training data we have.
	#i.e. in the low data regime use a batch size of 32, otherwise use 256.
	if len(trainX) < 2000:
		bs = 32
	else:
		bs = 256
	
	model = Sequential()  
	model.add(LSTM(numNodes, input_shape= (trainX.shape[1],trainX.shape[2]))) #input_shape = (timesteps,features)
	model.add(Dense(trainY.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	
	early_stopping_monitor = keras.callbacks.EarlyStopping(monitor = 'val_loss',min_delta = 0.01, patience = 1, verbose = 0, mode = 'auto')
	model.fit(trainX, trainY, epochs=100, batch_size=bs, validation_split = 0.2, verbose = 0, shuffle = True,	callbacks = [early_stopping_monitor])
	
	return model		

def testNetwork(users,productMap,idMap,testDic,model,given,k):
	#Test the trained neural network (non-adaptive).
	
	#For each user in the test set, convert their sequence into a vector (Basically just input a vector with 1's in the corresponding spots of the given products and 0s elsewhere),
	#pass this vector into the trained network, output the top k highest value predictions (in order),
	#and then compare the prediction against the true sequence.
	
	numProds = len(productMap) #Total number of products.
	
	saveScores = np.zeros((2,k)) #Holds the average scores of the predictions (one row for accuracy score, one for sequence score).
	count = 0.0
	for uid in testDic:
		if len(users[uid]) > given+k: #Only look at users that have reviewed at least 'given+k' movies.
			count += 1
			xpids = users[uid][:given] #First 'given' movies reviewed by user 'uid'

			tempX = np.zeros(numProds) #Set vector to 0 for all products
			seq = [] #Hold the movies already given/predicted.
			for pid in xpids:
				ind = productMap[pid] #Find unique integer index of product 'pid'
				seq.append(ind)
				tempX[ind] = 1 #Set the corresponding index for 'pid' to 1.
		
			tempX = tempX.reshape((1,numProds)) #Reshape so we can pass the vector into the model and do prediction.
			prediction = model.predict(tempX)[0] #Pass the input into the model and do prediction.
			guess = topK(prediction,k,seq,idMap) #Take the k highest value indices from the prediction vector (not including the indices that correspond to the given products).
			
			for i in range(1,k+1):
				saveScores[0][i-1] += accuracyScore(guess[:i],users[uid][given:])
				saveScores[1][i-1] += sequenceScore(guess[:i],users[uid][given:])
				
	saveScores = saveScores/count
	return saveScores
	
def testNetworkAdaptive(users,productMap,idMap,testDic,model,given,k):
	#Test the trained neural network (adaptive).
	#Basically input a vector with 1's in the corresponding spots of the given products (and 0's elsewhere),
	#and then output the highest value prediction (not already guessed).
	#If the guess is correct, update our input and re-run (otherwise just take the next highest prediction).
	
	numProds = len(productMap) #Total number of products.	
	saveScores = np.zeros((2,k)) #Holds the average scores of the predictions (one row for accuracy score, one for sequence score).
	count = 0.0
	for uid in testDic:
		if len(users[uid]) > given+k:
			count += 1
			xpids = users[uid][:given] #First i movies reviewed by user 'uid'

			tempX = np.zeros(numProds) #Set vector to 0 for all products
			tried = [] #Hold the products already given/predicted; used so we don't double guess.
			for pid in xpids:
				ind = productMap[pid] #Find unique integer index of product 'pid'
				tried.append(ind)
				tempX[ind] = 1 #Set the corresponding index for 'pid' to 1.
		
			outputSeq = [] #Our prediction for the next products to be reviewed (each entry is a product ID, not the integer ID, this is how it differs from the other seq variable).
			flag = 1 #Tells us whether we need to re-predict (i.e. re-run inference through network). This just helps save time.
			for i in range(0,k):
		
				if flag == 1:
					input = tempX.reshape((1,numProds)) #Reshape so we can pass the vector into the model and do prediction.
					prediction = model.predict(input)[0] #Pass the input into the model and do prediction.
				
				guess = topK(prediction,1,tried,idMap)[0] #Get the productID of the highest value in the prediction.
				guessInd = productMap[guess] #Get the integer ID of our guess.
				tried.append(guessInd) #Record the fact that we have guessed this product, so we don't try it again.
				outputSeq.append(guess) #Append our guess to our output sequence.
					
				if guess in users[uid]:
					tempX[guessInd] = 1 #If our guess is actually reviewed, use it to make future predictions.
					flag = 1
				else:
					flag = 0
					
			for i in range(1,k+1):
				saveScores[0][i-1] += accuracyScore(outputSeq[:i],users[uid][given:])
				saveScores[1][i-1] += sequenceScore(outputSeq[:i],users[uid][given:])
			
	saveScores = saveScores/count
	return saveScores

def testLSTM(users,productMap,idMap,testDic,model,given,k,inputSteps):
	#Non-adaptive version of LSTM, assume each guess is correct.
	#Given a trained LSTM network, we make one prediction and take the top k values (in order).
	
	#inputSteps is the max number of timesteps to input into the LSTM. 
	
	numProducts = len(productMap) #Total number of products.
		
	saveScores = np.zeros((2,k))
	count = 0.0 #Count the number of paths with more than 'given' pages in the path.
	for uid in testDic:
		if len(users[uid]) > given+k:
			count += 1
			tempX = []
			for i in range(0,inputSteps-given): #If 'given' is less than 'inputSteps', pre-pad the input with all-zero vectors.
				temp = np.zeros(numProducts)
				tempX.append(temp)
			for i in range(0,given): #For each given product, append one-hot encoded array (all 0s and 1 in index of corresponding product).
				temp = np.zeros(numProducts)
				ind = productMap[users[uid][i]] 
				temp[ind] = 1
				tempX.append(temp)
			
			tempX = np.array(tempX) #Turn into numpy array.
			tempX = tempX.reshape( (1,inputSteps,numProducts) )
			
			tried = users[uid][:given] #List of all the given products
			prediction = model.predict(tempX)[0] #Pass the input into the model and do prediction.
			guess = topK(prediction,k,tried,idMap) #Take the k highest value indices from the prediction vector (not including the indices that correspond to the given products).
			
			for i in range(1,k+1):
				saveScores[0][i-1] += accuracyScore(guess[:i],users[uid][given:])
				saveScores[1][i-1] += sequenceScore(guess[:i],users[uid][given:])	
	
	saveScores = saveScores/count
	return saveScores

def testLSTMAdaptive(users,productMap,idMap,testDic,model,given,k,inputSteps):
	#Adaptive version of LSTM (same process as testNetworkAdaptive, but with LSTM).
	
	#inputSteps is the max number of timesteps to input into the LSTM. 
	
	numProducts = len(productMap) #Total number of products.
		
	saveScores = np.zeros((2,k))
	count = 0.0 #Count the number of paths with more than 'given' pages in the path.
	for uid in testDic:
		if len(users[uid]) > given+k:
			count += 1
			tempX = []
			tried = [] #Hold the products already given/predicted; used so we don't double guess.
			for i in range(0,inputSteps-given): #If 'given' is less than 'inputSteps', pre-pad the input with all-zero vectors.
				temp = np.zeros(numProducts)
				tempX.append(temp)
			for i in range(0,given): #For each given page, append one-hot encoded array (all 0s and 1 in index of corresponding page).
				temp = np.zeros(numProducts)
				ind = productMap[users[uid][i]]
				tried.append(ind) 
				temp[ind] = 1
				tempX.append(temp)
			
			tempX = np.array(tempX) #Turn into numpy array.

			seq = [] #This is going to be our output sequence.
			flag = 1 #Tells us if we need to re-predict.
			for i in range(0,k):
				if flag == 1:
					tempX = tempX.reshape( (1,inputSteps,numProducts) )
					prediction = model.predict(tempX)[0]
			
				guessProd = topK(prediction,1,tried,idMap)[0] #Get the productID of the highest value in the prediction.
				maxInd = productMap[guessProd] #Get the integer ID of our guess.
				tried.append(maxInd) #Record the fact that we have guessed this product, so we don't try it again.
				seq.append(guessProd)
				
				if guessProd in users[uid]:
					tempX2 = [] 
					for j in range(1,inputSteps): #Basically we are shifting everything down by 1 and adding one new vector (representing the new current page).
						tempX2.append(tempX[0][j]) #Add everything from the old input (except the first one, which is now dropped).
					temp = np.zeros(numProducts) #Add a new vector, with a 1 in the maxInd spot.
					temp[maxInd] = 1
					tempX2.append(temp)
					tempX = np.array(tempX2)
					flag = 1
				else:
					flag = 0 #If we guessed wrong, no need to re-predict, just go to next highest value in same prediction vector.
					
			for i in range(1,k+1):
				saveScores[0][i-1] += accuracyScore(seq[:i],users[uid][given:])
				saveScores[1][i-1] += sequenceScore(seq[:i],users[uid][given:])	
		
	
	saveScores = saveScores/count
	return saveScores
	
#==================================================================
# Adaptive (and non-adaptive) Sequence Greedy training and testing
#==================================================================
	
def productDic(users,trainDic,thresh):
	#Create dic and dic2 such that:
	#dic[i][i] is number of times product i has been reviewed
	#dic[i][j] is number of times product j was reviewed after product i was reviewed first.
	#dic2[i][i] is frequency of product i being reviewed.
	#dic2[i][j] is probability of product i being reviewed given that product j was reviewed first
	
	#This is used as input to the sequence-greedy based algorithms as dic2 is basically the value of each edge.

	N = float(len(users)) #N is the number of users.
	dic = {}
	for uid in trainDic: #Now fill up dic[i][j]:
		temp = users[uid] #temp is the ordered list of products reviewed by user 'uid'. 
		for j in range(0,len(temp)):
			for i in range(0,j+1):
				pi = temp[i]
				pj = temp[j] #Either pi == pj or product 'pi' was reviewed before 'pj'.
				
				if pi not in dic:
					dic[pi] = {}
				if pj not in dic[pi]:
					dic[pi][pj] = 0.0
				dic[pi][pj] += 1
	
	dic2 = {} #dic2 is basically just a re-scaling of the dic we filled in above.
	for key1 in dic:
		for key2 in dic[key1]:
							
				if key1 == key2:
					score = dic[key1][key1]/N #The number of times key1 was reviewed, divided by the total number of users.
				else:
					score = dic[key1][key2]/(dic[key1][key1]) #The number of times key2 was reviewed after key1, divided by the number of times key1 was reviewed. 
				
				if score > thresh or key1 == key2: #To make it faster, only consider scores above a certain threshold (but include all individual item frequencies).
					if key1 not in dic2:
						dic2[key1] = {}
					dic2[key1][key2] = round(score,4)	
		
	return (dic,dic2)	
	
def maxValidAdaptive(p2,seq,states):
	#Given 'seq' (products already reviewed/guessed) and associated 'states',
	#output the highest value valid edge.
	
	maxEdge = ['x','x',0.0] #default value for [start vertex,end vertex,probability].
	for key1 in states: #First check all edges from reviewed products:
		if key1 in p2:
			for key2 in p2[key1]:
				if key2 not in seq and p2[key1][key2] > maxEdge[2]:
					maxEdge = [key1,key2,p2[key1][key2]]
	
	for key in p2: #Next check all self loops:
		if key not in seq and p2[key][key] > maxEdge[2]:
			maxEdge = [key,key,p2[key][key]]
	
	return maxEdge	
	
def maxValidNonadaptive(p2,seq,given):
	#Non-adaptive version of maxValidAdaptive (above). 
	#Output the highest value valid edge.
	
	maxEdge = ['x','x',0.0] #default value for [start vertex,end vertex,probability].
	for key1 in seq[:given]: #First check all edges from reviewed products:
		if key1 in p2:
			for key2 in p2[key1]:
				if key2 not in seq and p2[key1][key2] > maxEdge[2]:
					maxEdge = [key1,key2,p2[key1][key2]]
	
	for key in p2: #Next check all self loops:
		if key not in seq and p2[key][key] > maxEdge[2]:
			maxEdge = [key,key,p2[key][key]]
	
	return maxEdge			
	
def adaptive(p1,p2,users,testDic,given,k):
	#Adaptive sequence-greedy.
	#Main algorithm from this paper.
	#Outputs average scores (both accuracy and sequence scores) of adaptive sequence-greedy on the test set.
	
	saveScores = np.zeros((2,k))
	count = 0.0
	for uid in testDic:
		if len(users[uid]) > given+k: #Only consider users who have reviewed enough products.
			count += 1
			user = users[uid]
			seq = user[:given] #First 'given' products reviewed by this user.
			states = {} #keep state of each node
			for key in seq: #all nodes start in state 0, except those already in seq.
				states[key] = 1 
		
			while len(seq) < k+given:	
				[u,v,weight] = maxValidAdaptive(p2,seq,states)
	
				if u not in seq: #check if we need to add start node to sequence
					seq.append(u)
					if u in user: #check if start node should be in state 1
						states[u] = 1
				if u != v: #if not self-loop, append the end vertex.
					seq.append(v) #append end node to sequence
					if v in user:
						states[v] = 1 #check if end node should be in state 1.
			
			for i in range(1,k+1):
				saveScores[0][i-1] += accuracyScore(seq[given:given+i],user[given:])
				saveScores[1][i-1] += sequenceScore(seq[given:given+i],user[given:])
	
	saveScores = saveScores/count			
	return saveScores
			
def nonadaptive(p1,p2,users,testDic,given,k):
	#Non-adaptive sequence-greedy.
	#Outputs average scores (both accuracy and sequence scores) of sequence-greedy on the test set.
	#Main algorithm from "Submodularity on Hypergraphs: From Sets to Sequences".
	#https://arxiv.org/pdf/1802.09110.pdf
	
	saveScores = np.zeros((2,k)) #
	count = 0.0
	for uid in testDic:
		if len(users[uid]) > given+k:
			count += 1
			
			seq = users[uid][:given]
			while len(seq) < k+given:	
				[u,v,weight] = maxValidNonadaptive(p2,seq,given)
	
				if u not in seq: #check if we need to add start node to sequence
					seq.append(u)
				if u != v: #If not self-loop:
					seq.append(v) #append end node to sequence
				
			for i in range(1,k+1):
				saveScores[0][i-1] += accuracyScore(seq[given:given+i],users[uid][given:])
				saveScores[1][i-1] += sequenceScore(seq[given:given+i],users[uid][given:])
	
	saveScores = saveScores/count	
	return saveScores
	
def nonsequence(p1,p2,users,testDic,given,k):
	#Output the k most common items not already reviewed by each user.
	#This the Frequency baseline in the paper.
	
	saveScores = np.zeros((2,k))
	count = 0.0
	for uid in testDic:
		if len(users[uid]) > given+k:
			count += 1
			
			seq = users[uid][:given]
			valid = []
			for key in p2:
				if key not in seq:
					valid.append([key,p2[key][key]])
			valid = sorted(valid, key = lambda x: x[1], reverse = True) #sort by decreasing frequency
	
			for i in range(0,k):
				seq.append(valid[i][0]) #append the most common products not already in the sequence.
			
			
			for i in range(1,k+1):
				saveScores[0][i-1] += accuracyScore(seq[given:given+i],users[uid][given:])
				saveScores[1][i-1] += sequenceScore(seq[given:given+i],users[uid][given:])	
	
	saveScores = saveScores/count	
	return saveScores	

#==============================
# Scoring and graphing results
#==============================

def genPairs(seq):
	#seq is an ordered list, we want to return list of ordered pairs.
	#for example: if fseq = [1,2,3,4]
	#then we return [(1,1), (1,2), (1,3), (1,4), (2,2), (2,3), (2,4), (3,3), (3,4), (4,4)]
	
	pairs = []
	k = len(seq)
	for i in range(0,k):
		for j in range(0,i+1):
			pairs.append([seq[j],seq[i]])
			
	return pairs
	
def sequenceScore(ans,fseq):
	#This is basically a modified version of Kendall tau distance.

	#ans is our guess for the next k movies to be reviewed.
	#fseq are the actual movies that were reviewed.
	
	#Basically, for each ordered pair of movies in fseq,
	#we will check if that pair appears in the correct order in ans.
	
	#for example: if fseq = [1,2,3,4]
	#then we will check for [(1,1), (1,2), (1,3), (1,4), (2,2), (2,3), (2,4), (3,3), (3,4), (4,4)]
	#therefore if ans = [1,4,2,5], then it only contains the ordered pairs
	#(1,1), (2,2), (4,4), (1,4), and (1,2).
	
	ansPairs = genPairs(ans)
	fseqPairs = genPairs(fseq)
	
	score = 0.0
	for entry in ansPairs:
		if entry in fseqPairs:
			score += 1
			
	return score
	
def accuracyScore(guess,truth):
	#'guess' is our prediction for which products the reviewer will review next, and 'truth' are the products they actually reviewed.
	#accuracyScore just counts the number of guessed products that were actually reviewed.
	#Note that order of output does not matter for this score.
	
	score = 0.0
	for entry in guess:
		if entry in truth:
			score += 1
	
	return score

def graphResults(nnAdaptive,nn,lstmAdaptive,lstm,ASG,SG,freq,k,m,percTrain):
	#Use matplotlib to graph the given results.
	#nnAdaptive is the average scores of the adaptive feed-forward network,
	#nn is average scores non-adaptive feed-forward network,
	#lstmAdaptive is the average scores of the adaptive LSTM,
	#lstm is average scores non-adaptive LSTM,
	#ASG is average scores of adaptive sequence-greedy,
	#SG is average scores of non-adaptive sequence-greedy,
	#freq is average scores of Frequency baseline.
	
	#Save name of graphs is 'amazonGraph-m-p-acc.pdf' and 'amazonGraph-m-p-seq.pdf'
	#where m is the minimum number of reviews for a product to be considered
	#and p is the percentage of data used for training.
	
	xList = range(1,k+1)
	plt.xlabel('Number of Recommendations',size = '25')
	plt.ylabel('Accuracy Score',size='25')
	plt.xticks(xList,size='22')
	plt.yticks(size='22')
	green_line = mlines.Line2D([], [], color='g', marker='s', linestyle='-',markersize=5, label='Adaptive Sequence-Greedy')
	blue_line = mlines.Line2D([], [], color='b', marker='s', linestyle=':',markersize=5, label='Sequence-Greedy')
	red_line = mlines.Line2D([], [], color='r', marker='>', linestyle=':',markersize=5, label='Frequency')
	teal_line = mlines.Line2D([], [], color='c', marker='o', linestyle='-',markersize=5, label='Adaptive Feed Forward NN')
	teal_dotted_line = mlines.Line2D([], [], color='c', marker='o', linestyle=':',markersize=5, label='Non-Adaptive Feed Forward NN')
	purple_line = mlines.Line2D([], [], color='m', marker='P', linestyle='-',markersize=5, label='Adaptive LSTM')
	purple_dotted_line = mlines.Line2D([], [], color='m', marker='P', linestyle=':',markersize=5, label='Non-Adaptive LSTM')
	plt.legend(fontsize = 'large',handles=[green_line,blue_line,red_line,teal_line,teal_dotted_line,purple_line,purple_dotted_line])
	plt.plot(xList, nnAdaptive[0], '-co',linewidth=2.0,markersize=5.0)
	plt.plot(xList, nn[0], ':co',linewidth=2.0,markersize=5.0)
	plt.plot(xList, lstmAdaptive[0], '-mP',linewidth=2.0,markersize=5.0)
	plt.plot(xList, lstm[0], ':mP',linewidth=2.0,markersize=5.0)
	plt.plot(xList, ASG[0], '-gs',linewidth=2.0,markersize=5.0)
	plt.plot(xList, SG[0], ':bs',linewidth=2.0,markersize=5.0)
	plt.plot(xList, freq[0], ':r>',linewidth=2.0,markersize=5.0)
	savename = 'amazonGraph-'+str(m)+'-'+str(int(percTrain*100))+'-acc.pdf'
	plt.savefig(savename,bbox_inches='tight')
	plt.show()

	plt.xlabel('Number of Recommendations',size = '25')
	plt.ylabel('Sequence Score',size='25')
	plt.xticks(xList,size='22')
	plt.yticks(size='22')
	green_line = mlines.Line2D([], [], color='g', marker='s', linestyle='-',markersize=5, label='Adaptive Sequence-Greedy')
	blue_line = mlines.Line2D([], [], color='b', marker='s', linestyle=':',markersize=5, label='Sequence-Greedy')
	red_line = mlines.Line2D([], [], color='r', marker='>', linestyle=':',markersize=5, label='Frequency')
	teal_line = mlines.Line2D([], [], color='c', marker='o', linestyle='-',markersize=5, label='Adaptive Feed Forward NN')
	teal_dotted_line = mlines.Line2D([], [], color='c', marker='o', linestyle=':',markersize=5, label='Non-Adaptive Feed Forward NN')
	purple_line = mlines.Line2D([], [], color='m', marker='P', linestyle='-',markersize=5, label='Adaptive LSTM')
	purple_dotted_line = mlines.Line2D([], [], color='m', marker='P', linestyle=':',markersize=5, label='Non-Adaptive LSTM')
	plt.legend(fontsize = 'large',handles=[green_line,blue_line,red_line,teal_line,teal_dotted_line,purple_line,purple_dotted_line])
	plt.plot(xList, nnAdaptive[1], '-co',linewidth=2.0,markersize=5.0)
	plt.plot(xList, nn[1], ':co',linewidth=2.0,markersize=5.0)
	plt.plot(xList, lstmAdaptive[1], '-mP',linewidth=2.0,markersize=5.0)
	plt.plot(xList, lstm[1], ':mP',linewidth=2.0,markersize=5.0)
	plt.plot(xList, ASG[1], '-gs',linewidth=2.0,markersize=5.0)
	plt.plot(xList, SG[1], ':bs',linewidth=2.0,markersize=5.0)
	plt.plot(xList, freq[1], ':r>',linewidth=2.0,markersize=5.0)
	savename = 'amazonGraph-'+str(m)+'-'+str(int(percTrain*100))+'-seq.pdf'
	plt.savefig(savename,bbox_inches='tight')
	plt.show()







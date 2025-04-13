######################################################
#		    	 KEY 			     #
#	                                             #
#	#### = Start or End of Function 	     #
#	                                             #
#	# = notes and simple comments  	             #
# 						     #
#	### = Explanation of function processes      #
#                                                    #
#	######### = End of Functions in Class	     #
#						     #
#	##### = section identifier 		     #
#                                                    #
#                                                    #
###################################################-MA
import pandas 
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import random



class heartAttackTrees: 
	def findNShow():
	#### Start of findNShow
	###getting our data file, reading the data and making a data tree on that plot, print out accuracy
		def createWindow(feats):
		#### Start of createWindow
		###used to create window and change around settings 
			plt.figure(feats,figsize=(10,7))
			#plt.tight_layout()
		#### End of createWindow
		
		dataHold = pandas.read_csv("data.csv")	
		features = ['age','sex','cp','trestbps','chol','thalach']
		testAge = random.randint(28, 65)
		testSex = random.randint(0, 1)
		testCp = random.randint(0, 4)
		testRBpm = random.randint(98, 200)
		testChol = random.randint(85,603)
		testThal = random.randint(85,185)
		#num 0 = false or "no heart attack
		target = 'num'
		overAllSum = 0


		#loop "inside" of features to print out predictions based on each "feature" or category 
		for i in range(len(features)):		
			x = dataHold[features]
			y = dataHold[target]
			#train our model on the data above
			xTrain, xTest, yTrain, yTest = train_test_split(x.values, y, shuffle =True, test_size=.4, random_state=31,train_size=.6)
			#making our classifier, making a tree based on the trained data, print out accuracy
			dtree = DecisionTreeClassifier()
			dtree = dtree.fit(xTrain, yTrain)
			yPred = dtree.predict(xTest)
			accuracy = accuracy_score(yTest, yPred)
			overAllSum+=accuracy



			#displaying everything
			print(f"Test Result {i+1} Age:{testAge} Sex:{testSex} Cp:{testCp} RestBpm:{testRBpm} Cholesterol:{testChol} MaxHeartRate:{testThal} :","HA:",dtree.predict([[testAge,testSex,testCp,testRBpm,testChol,testThal]]), end=" | ")
			print(f"Accuracy: {accuracy}\n")

			#make window, and print out data tree , filled and 5 font
			createWindow(i)

			tree.plot_tree(dtree, feature_names=features, filled =True, fontsize = 5, class_names=True)
		#show all (features[i]) number of data forest
		#display out all random forsest to compare 
		print("Overall accuracy of all Trees:",overAllSum/len(features))
		#plt.show()
	#### End of findNShow()
				#########End Of Functions in Class#########
	##### Run Sequence #####		
	findNShow()

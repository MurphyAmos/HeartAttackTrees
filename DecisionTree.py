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
		features = ['age','sex','cp','trestbps','chol']
		#num 0 = false or "no heart attack
		target = 'num'
	
		#loop "inside" of features to print out predictions based on each "feature" or category 
		for i in range(len(features)):		
			x = dataHold[features]
			y = dataHold[target]
			#train our model on the data above
			xTrain, xTest, yTrain, yTest = train_test_split(x.values, y, test_size=0.2, random_state=50)
			#making our classifier, making a tree based on the trained data, print out accuracy
			dtree = DecisionTreeClassifier()
			dtree = dtree.fit(xTrain, yTrain)
			yPred = dtree.predict(xTest)
			accuracy = accuracy_score(yTest, yPred)
			print(f"Accuracy of Tree {i}: {accuracy}")

			#make window, and print out data tree , filled and 5 font
			createWindow(i)
			tree.plot_tree(dtree, feature_names=features, filled =True, fontsize = 5, class_names=True)
		#show all (features[i]) number of data forest
		plt.show()
	#### End of findNShow()
				#########End Of Functions in Class#########
	##### Run Sequence #####		
	findNShow()

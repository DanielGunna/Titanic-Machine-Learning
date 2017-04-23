 # Data Analysis
import pandas as pd
import numpy as np
import random as rnd

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


def printGeneralStatistics( data ):
	print( data.describe() )					# Statistics
	print( data.describe(include=['O']) )		# Distribution

def printGeneralInformation( data ):
	print( data.columns.values )				# Feature names
	print( data.info )							# Data Types

def setAgeBoundaries (  ):
	for dataset in combine:
		dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
		dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
		dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
		dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
		dataset.loc[ dataset['Age'] > 64, 'Age']

def normalizeFamily( ):
	for dataset in combine:
		dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

def pivotingData ( data, entry1, entry2, groupBy, sortBy ):
	return data[[ entry1 , entry2 ]].groupby([groupBy], as_index=False).mean().sort_values(by=sortBy, ascending=False)

def printPivotedData( set ):
	#only categorical values
	print ( pivotingData ( 'Pclass','Survived','Pclass','Survived' ) )
	print ( pivotingData ( 'Sex','Survived','Sex','Survived' ) )
	print ( pivotingData ( 'SibSp','Survived','SibSp','Survived' ) )
	print ( pivotingData ( 'Parch','Survived','Parch','Survived' ) )


def visualizeNumericalCorrelation( set, feature1, feature2 ):
	g = sns.FacetGrid(set, col=feature2)
	g.map(plt.hist, feature1, bins=20)

def visualizeSurvivedCorrelation( set, feature1, feature2 ):
	grid = sns.FacetGrid(train_df, col='Survived', row=feature2, size=2.2, aspect=1.6)
	grid.map(plt.hist, feature1, alpha=.5, bins=20)
	grid.add_legend();

def classifyWithLogisticRegression ( trainingData, results, testData ):
	clf_logreg = LogisticRegression()
	clf_logreg.fit(trainingData, results)
	return clf_logreg.predict(testData)
	
def classifyWithDecisionTree ( trainingData, results, testData ):
	clf_tree = tree.DecisionTreeClassifier()
	clf_tree.fit(trainingData, results)
	return clf_tree.predict(testData)

def classifyWithSVM ( trainingData, results, testData ):
	clf_svm = SVC()
	clf_svm.fit(trainingData,results)
	return clf_svm.predict(testData)

def classifyWithPerceptron ( trainingData, results, testData ):
	clf_perceptron = Perceptron()
	clf_perceptron.fit(trainingData,results)
	return clf_perceptron.predict(testData)

def classifyWithKNeighbors ( trainingData, results, testData ):
	clf_KNN = KNeighborsClassifier()
	clf_KNN.fit(trainingData,results)
	return clf_KNN.predict(testData)

def classifyWithGaussianNaiveBayes ( trainingData, results, testData ):
	clf_GaussianNB = GaussianNB()
	clf_GaussianNB.fit(trainingData,results)
	return clf_GaussianNB.predict(testData)

def classifyWithStochasticGradientDescent ( trainingData, results, testData ):
	sgd = SGDClassifier()
	sgd.fit(trainingData, results)
	return sgd.predict(testData)

def classifyWithLinearSVC ( trainingData, results, testData ):
	linear_svc = LinearSVC()
	linear_svc.fit(trainingData, results)
	return linear_svc.predict(testData)

def classifyWithRandomForest ( trainingData, results, testData ):
	random_forest = RandomForestClassifier(n_estimators=100)
	random_forest.fit(trainingData, results)
	return random_forest.predict(testData)

def normalizeSex ( ):
	for dataset in combine:
		dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

def normalizeAges ( ):
	guess_ages = np.zeros((2,3))
	for dataset in combine:
		for i in range(0, 2):
			for j in range(0, 3):
				guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
				age_guess = guess_df.median()
				guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

		for i in range(0, 2):
			for j in range(0, 3):
				dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),'Age'] = guess_ages[i,j]
	
		dataset['Age'] = dataset['Age'].astype(int)	

def normalizeEmbarked( ):
	freq_port = train_df.Embarked.dropna().mode()[0]

	for dataset in combine:
		dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

	for dataset in combine:
		dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

def normalizeFare():
	global train_df, test_df
	test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
	train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
	train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

	for dataset in combine:
		dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
		dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
		dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
		dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
		dataset['Fare'] = dataset['Fare'].astype(int)

	train_df = train_df.drop(['Parch', 'SibSp'], axis=1)
	test_df = test_df.drop(['Parch', 'SibSp'], axis=1)
	train_df = train_df.drop(['FareBand'], axis=1)


def normalizeAgeClass( ):
	for dataset in combine:
		dataset['Age*Class'] = dataset.Age * dataset.Pclass


def normalizeData( ):
	normalizeSex ( )
	normalizeAges( )
	setAgeBoundaries( )
	normalizeFamily( )
	normalizeEmbarked( )
	normalizeAgeClass( )
	normalizeFare( )



def main ( ):
	global train_df
	global test_df
	global combine

	# Training and Testing Data
	train_df = pd.read_csv('../input/train.csv')
	test_df = pd.read_csv('../input/test.csv')

	# Drop Useless Features
	train_df = train_df.drop(['Ticket', 'Cabin', 'Name'], axis=1)
	test_df = test_df.drop(['Ticket', 'Cabin', 'Name'], axis=1)

	# Normalize both data sets
	combine = [train_df, test_df]
	normalizeData( )
	combine = [train_df, test_df]

	# Setting up data
	X_train = train_df.drop(["Survived","PassengerId","Fare","Age","Pclass"], axis=1)
	Y_train = train_df["Survived"]
	X_test  = test_df.drop(["PassengerId","Fare","Age","Pclass"], axis=1).copy()
	X_train.shape, Y_train.shape, X_test.shape

	print X_train
	# Use predictive model (ML)
	prediction = classifyWithDecisionTree(X_train, Y_train, X_test)

	#Build the answer
	submission = pd.DataFrame({
		"PassengerId": test_df["PassengerId"],
		"Survived": prediction
		})

	# Put it in csv file
	submission.to_csv('../output/submission.csv', index=False)


main( )

# coding: utf-8

# In[1]:

#import Spark and MLlib packages
from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.linalg import Vectors

#import data analysis packages
import numpy as np
import pandas as pd
import sklearn

from pandas import Series, DataFrame
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from numpy import array

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

#import data visualization packages
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic(u'matplotlib inline')

#misc packages
from __future__ import division
from __future__ import print_function


# In[2]:

#I.Load dataset
#The dataset is picked from here https://github.com/mwaskom/seaborn-data
iris = sns.load_dataset("iris")


# In[3]:

#Get the data size and feature size
data_size, feature_size = iris.shape
print("data_size is {}, feature_size is {}".format(data_size, feature_size))


# In[4]:

#II. datacleaning for PySpark
#1.Label Setosa as 0, Versicolor as 1, Virginica as 2
species = iris.species
label_species = []
for i in range(len(species)):
    if species[i] == 'setosa':
        label_species.insert(i, 0)
    elif species[i] == 'versicolor':
        label_species.insert(i, 1)
    else :
        label_species.insert(i, 2)


# In[5]:

#2.Convert list to Series so that it can be load to dataframe
series_species = Series(label_species, index = range(len(species)))


# In[6]:

#3.Reload Series to Species column 
iris['species'] = series_species
print(iris.head())


# In[7]:

#4.Move Species column to the first column
cols = iris.columns.tolist()
cols = cols[-1:] + cols[:-1]
iris = iris[cols]


# In[8]:

iris.head()


# In[9]:

#III. Using Sciki-learn SVC 
#This time, let's analysis the last two features 
#petal_length and petal_width
data_col = cols[-2:]

#Target data
Y = iris['species']

#Traning and testing dataset
X = iris[data_col] 


# In[10]:

print(X.head())


# In[11]:

print(Y.tail())


# In[31]:

# Split the data into Trainging and Testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# In[32]:

X['petal_length'].shape


# In[33]:

X_train[:,1].shape


# In[34]:

# Train the model and Predict
NB_multi = MultinomialNB().fit(X_train, Y_train)
NB_Bernou = BernoulliNB().fit(X_train, Y_train)


# In[39]:

for i , clf in enumerate((NB_multi, NB_Bernou)):
    predicted = clf.predict(X_test)
    expected = Y_test
    
    #compare results
    errors = metrics.accuracy_score(expected, predicted)
    if i == 0:
        print("Multinomail Naive Bayes accuracy is {}".format(errors))
    else:
        print("Bernoulli Naive Bayes accuracy is {}".format(errors))


# In[25]:

#dump the tips to a local file for now
X_data_size = len(X['petal_length'])
Y = np.vstack(Y)

fo = open("/usr/local/spark/examples/src/main/resources/iris_3.data", "w")
for i in range(X_data_size):    
    fo.write("{},{} {}\n".format(float(Y[i][0]), float(X['petal_length'][i]), float(X['petal_width'][i]))) 

fo.close()


# In[41]:

#IV Use MLlib
sc = SparkContext("local", "SVM_LogisticRegression")


# In[42]:

#code parsing function and load data
def parseLine(line):
    parts = line.split(',')
    label = float(parts[0])
    features = Vectors.dense([float(x) for x in parts[1].split(' ')])
    return LabeledPoint(label, features)

data = sc.textFile('/usr/local/spark/examples/src/main/resources/iris_3.data').map(parseLine)


# In[43]:

data.take(10)


# In[48]:

#Split the training set and test set
#from what we used in scikit-learn part, 112/150 = 75%
training, test = data.randomSplit([0.75, 0.25], seed = 0)

#Training model
model = NaiveBayes.train(training, 0.5)


# In[49]:

#predict and test accuracy
predictionAndLabel = test.map(lambda p: (model.predict(p.features), p.label))
accuracy = predictionAndLabel.filter(lambda (x, v): x == v).count()/test.count()

print("PySpark NaiveBayes model accuracy {}".format(accuracy))


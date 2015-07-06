# coding: utf-8
import csv as csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# get_ipython().magic(u'ls ')
# get_ipython().magic(u'cd Documents/')
# get_ipython().magic(u'cd Kaggle/')
# get_ipython().magic(u'ls ')
# get_ipython().magic(u'cd TitanicSurivivalPrediction/')
# get_ipython().magic(u'ls ')
csv_file_object = csv.reader(open('train.csv','rb'))
header = csv_file_object.next()
data = []
for row in csv_file_object:
    data.append(row)
data = np.array(data)
print data



df = pd.read_csv("train.csv",header=0)
df
df.head(3)
type(df)
df.dtypes
df.info()
df.describe()
df['Age'][0:10]
df.Age[0:10]
type(df['Age'])
df['Age'].mean()
df['Age'].median()
df[['Sex', 'Pclass', 'Age']]
df[df['Age'] > 60]

df[df['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']]
df[df['Age'].isnull()][['Sex', 'Pclass', 'Age']]

for i in range(1,4):
    print i, len(df[ (df['Sex'] == 'male') & (df['Pclass'] == i)])
    
# import pylab as P
# df['Age'].hist()
# P.show()
# df['Age'].dropna().hist(bins=16, range=(0,80),alpha = 0.5)
# P.show()
# df['Gender'] = 4
# df.head()
# df['Gender'] = df['Sex'].map(lambda x: x[0].upper())
# df.head()
df['Gender'] = df['Sex'].map( {'female':0, 'male': 1} ).astype(int)
# df.head()
# df['Embarked']
median_ages = np.zeros((2,3))

for i in range(0,2):
    for j in range(0,3):
        median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()

def f(x):
	return median_ages[x.Gender,x.Pclass-1] if x.Age != x.Age else x.Age

df['AgeFill'] = df.apply(lambda x: median_ages[x.Gender,x.Pclass-1] if x.Age != x.Age else x.Age, axis = 1)

# df[df['Age'].isnull()][['Gender', 'Pclass', 'Age', 'AgeFill']].head(10)
# for i in range(0,2):
#     for j in range(0,3):
#         df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]

# df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
# df['FamilySize'] = df['SibSp'] + df['Parch']
# df['Age*Class'] = df.AgeFill * df.Pclass
# df['Age*Class'].hist()
# P.show()
# df.dtypes()
# df.dtypes[df.dtypes.map(lambda x: x=='object')]
# df = df.drop(['Name', 'Sex',  'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1)
# train_data = df.values
# train_data
# get_ipython().magic(u'save ')
# get_ipython().magic(u'pinfo save')
df.loc[df['Embarked'].isnull(), 'Embarked'] = df['Embarked'].dropna().mode().values


Ports = list(enumerate(df['Embarked'].unique()))     #interesting way to create a list of enumerates
Ports_dict = {name:i for i,name in Ports}           # creating a dict so that we can map letters to values
df['Embarked'] = df['Embarked'].map(Ports_dict)

df.drop(['Name','Sex','Ticket','Cabin','PassengerId', 'Age'],inplace=True, axis = 1) # df is left with only useful data now
'''
======= END of Training data cleaning ====
'''

'''
======= Start of Testing data cleaning ====
'''
# Creating Gender feature like before
test_df = pd.read_csv("test.csv", header = 0)
test_df['Gender'] = test_df['Sex'].map({'female':0, 'male':1})

# Creating AgeFill feature like before
test_df["AgeFill"] = test_df.apply(lambda x: median_ages[x.Gender, x.Pclass-1] if x.Age != x.Age else x.Age, axis = 1)

# Fixing the Embarked feature like before
test_df.Embarked = test_df.Embarked.map(Ports_dict)

# this table contains the median_Fare for each class. Will be used to fill in empty Fare values for some data
median_Fare = np.zeros(3)
for i in range(3):
    median_Fare[i] = df.Fare[df.Pclass == i+1].median()
# The following code is selecting the rows in test_df that doesn't have a Fare value, and assign a value to it
# using its class as a criteria and using the median_Fare table that we computed before
test_df.loc[test_df.Fare.isnull(), 'Fare'] = test_df.loc[test_df.Fare.isnull()].apply(lambda x: median_Fare[x.Pclass-1], axis = 1)

# Save the PassengerId for later use... since it is not used as a parameter for our prediction model
test_ids = test_df.PassengerId
# Now we can drop everything in that test_df that we don't use
test_df.drop(['Cabin','Age','PassengerId','Name','Sex','Ticket'], inplace = True, axis = 1)

'''
======= END of cleaning test data ====
'''

'''
======= Start of predicting using RandomForest ====
'''

train_data = df.values
test_data = test_df.values

print "Training..."
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data[::,1::], train_data[::, 0])

print "Predicting..."
output = forest.predict(test_data).astype(int)


prediction_file = open("myPandaPrediction.csv", "wb")
open_file_object = csv.writer(prediction_file)
open_file_object.writerow(["PassengerId", "Survived"])
open_file_object.writerows(zip(test_ids,output))
prediction_file.close()
print 'Done'

# Trying predicting with SVM
from sklearn import svm

clf_svm = svm.SVC()
print "Training..."
clf_svm.fit(train_data[::,1::],train_data[::,0])

print "Predicting..."
outcome = clf_svm.predict(test_data[::]).astype(int)

prediction_file = open("myPandaPrediction.csv", "wb")
open_file_object = csv.writer(prediction_file)
open_file_object.writerow(["PassengerId", "Survived"])
open_file_object.writerows(zip(test_ids,outcome))
prediction_file.close()

# coding: utf-8
import csv as csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

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

# type(df)
# df.dtypes
# df.info()
# df.describe()
# df['Age'][0:10]
# df.Age[0:10]
# type(df['Age'])
# df['Age'].mean()
# df['Age'].median()
# df[['Sex', 'Pclass', 'Age']]
# df[df['Age'] > 60]
# 
# df[df['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']]
# df[df['Age'].isnull()][['Sex', 'Pclass', 'Age']]

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
======= Normalizing stuff
'''

# normalizing stuff
comb_SibSp = pd.concat([df.SibSp, test_df.SibSp])
df.SibSp = (df.SibSp - comb_SibSp.mean())/comb_SibSp.std()
test_df.SibSp = (test_df.SibSp - comb_SibSp.mean())/comb_SibSp.std()
comb_Parch = pd.concat([df.Parch, test_df.Parch])
df.Parch = (df.Parch - comb_Parch.mean())/comb_Parch.std()
test_df.Parch = (test_df.Parch - comb_Parch.mean())/comb_Parch.std()

comb_AgeFill = pd.concat([df.AgeFill,test_df.AgeFill])
df.AgeFill = (df.AgeFill - comb_AgeFill.mean())/comb_AgeFill.std()
test_df.AgeFill = (test_df.AgeFill - comb_AgeFill.mean())/comb_AgeFill.std()






'''
======= Start of predicting using RandomForest ====
'''

train_data = df.values
test_data = test_df.values

train_data = np.random.permutation(train_data[::,::])
temp = np.size(train_data,0)/5
cv_data = train_data[0:temp:,::]
train_data2 = train_data[temp::,::]

from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")
# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 7),
              "min_samples_split": sp_randint(1, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}


print "Training..."
forest = RandomForestClassifier(n_estimators = 100)

# run randomized search
n_iter_search = 30
random_search = RandomizedSearchCV(forest, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(train_data[::,1::], train_data[::,0])
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)




train_data = np.random.permutation(train_data[::,::])
temp = np.size(train_data,0)/5
cv_data = train_data[0:temp:,::]
train_data2 = train_data[temp::,::]
#
# forest = RandomForestClassifier(n_estimators= 100, bootstrap = True, min_samples_leaf = 7, min_samples_split = 7,
#                                  criterion = 'gini', max_features = 3, max_depth= None)
# forest = forest.fit(train_data2[::,1::], train_data2[::, 0])

print "Predicting..."
# output_cv = forest.predict(cv_data[::,1::]).astype(int)
# output_train = forest.predict(train_data2[::,1::]).astype(int)

output_cv = random_search.predict(cv_data[::,1::]).astype(int)
output_train = random_search.predict(train_data2[::,1::]).astype(int)
print "Done..."
if (len(train_data2) != len(output_train)): print "something wrong"
else:
    print "RF Training Accuracy:", \
        len(output_train[train_data2[::,0] == output_train])/float(len(output_train)),
    print "CV Accuracy:", len(output_cv[output_cv == cv_data[::,0]])/float(len(output_cv))

# real test data
test_data = test_df.values

output = random_search.predict(test_data).astype(int)

train_data = df.values

# ============ making the prediction file============
prediction_file = open("July25_RF.csv", "wb")
open_file_object = csv.writer(prediction_file)
open_file_object.writerow(["PassengerId", "Survived"])
open_file_object.writerows(zip(test_ids,output))
prediction_file.close()
print 'Done'

'''
============ Learning curve using SVM or RF with %80 %20 training/CV data =====
'''
from sklearn.grid_search import GridSearchCV
from time import time

train_data = df.values
test_data = test_df.values

train_data = np.random.permutation(train_data[::,::])
temp = np.size(train_data,0)/5
cv_data = train_data[0:temp:,::]
train_data2 = train_data[temp::,::]

#
# clf_svm = svm.SVC()
# clf_svm.fit(train_data2[::,1::],train_data2[::,0])
#
# outcome_train2 = clf_svm.predict(train_data2[::,1::]).astype(int)
# train2_accuracy = len(train_data2[train_data2[::,0] == outcome_train2]) / float(len(outcome_train2))
#
# outcome_cv = clf_svm.predict(cv_data[::,1::]).astype(int)
# cv_accuracy = len(cv_data[cv_data[::,0] == outcome_cv]) / float(len(outcome_cv))

t0 = time()
param_grid = {'C': [0.5, 0.6, 0.4, 0.7, 0.3],
              'gamma': [0.005, 0.001, 0.0005, 0.0001, 0.05, 0.01]
              }
clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='auto'), param_grid, cv = 5)
# clf = svm.SVC(kernel='sigmoid', class_weight='auto', gamma = 0.004, C = 0.5)
print "Fitting data..."
clf = clf.fit(train_data2[::,1::], train_data2[::,0])
print "done"
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

outcome_train2 = clf.predict(train_data2[::,1::]).astype(int)
train2_accuracy = len(train_data2[train_data2[::,0] == outcome_train2]) / float(len(outcome_train2))

outcome_cv = clf.predict(cv_data[::,1::]).astype(int)
cv_accuracy = len(cv_data[cv_data[::,0] == outcome_cv]) / float(len(outcome_cv))

print "SVM training data accuracy:" , train2_accuracy, "  CV data accuracy:", cv_accuracy

'''
============== Trying predicting with SVM ============
'''
train_data = df.values
test_data = test_df.values

clf_svm = svm.SVC()
print "Training..."
t0 = time()
param_grid = {'C': [4.5, 5, 5.6],
              'gamma': [0.05, 0.01]}
clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='auto'), param_grid)
clf = clf.fit(train_data[::,1::], train_data[::,0])
print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)



print "Predicting..."
outcome = clf.predict(test_data[::]).astype(int)


# ============ making the prediction file============
prediction_file = open("July23_SVM(2).csv", "wb")
open_file_object = csv.writer(prediction_file)
open_file_object.writerow(["PassengerId", "Survived"])
open_file_object.writerows(zip(test_ids,outcome))
prediction_file.close()

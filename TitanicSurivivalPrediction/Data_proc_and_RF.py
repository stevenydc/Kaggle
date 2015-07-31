__author__ = 'stevenydc'
import csv as csv
import numpy as np
import pandas as pd
import pylab as P
from sklearn.ensemble import RandomForestClassifier
from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn import svm
import re

StandardDataCleaning = False
df = pd.read_csv("train.csv",header=0)

# Creating Gender variable that encodes sex with 0 for female and 1 for male
df['Gender'] = df['Sex'].map( {'female':0, 'male': 1} ).astype(int)


# Calculating median age for every gender/class group... will be used to fill missing age data
median_ages = np.zeros((2,3))
for i in range(0,2):
    for j in range(0,3):
        median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()

# Creating new variable AgeFill that fills null data points in the original Age variable
df['AgeFill'] = df.apply(lambda x: median_ages[x.Gender,x.Pclass-1] if x.Age != x.Age else x.Age, axis = 1)

# Alternative way of doing the same thing as above (one liner):
# def f(x):
# 	return median_ages[x.Gender,x.Pclass-1] if x.Age != x.Age else x.Age
# for i in range(0,2):
#     for j in range(0,3):
#         df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'AgeFill'] = median_ages[i,j]

# Filling the missing Embarked values with most popular port (the mode of the port variable)
df.loc[df['Embarked'].isnull(), 'Embarked'] = df['Embarked'].dropna().mode().values

# # Changing categorical variable to numerical... which is required for most ML algorithms
# Ports = list(enumerate(df['Embarked'].unique()))     #interesting way to create a list of enumerates
# Ports_dict = {name:i for i,name in Ports}           # creating a dict so that we can map letters to values
# df['Embarked'] = df['Embarked'].map(Ports_dict)

df = pd.concat([df, pd.get_dummies(df.Embarked).rename(columns=lambda x: 'Embarked_' + str(x))], axis=1)

df['Names'] = df['Name'].map(lambda x: len(re.split(' ', x)))
# All titles in data is preceded by a ',' and is followed by a '.'
# The .*? makes the '*' operation non greedy! 
df['Title'] = df['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
# Group low-occuring, related titles together
df['Title'][df.Title == 'Jonkheer'] = 'Master'
df['Title'][df.Title.isin(['Ms','Mlle'])] = 'Miss'
df['Title'][df.Title == 'Mme'] = 'Mrs'
df['Title'][df.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'
df['Title'][df.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'
# Changing Title variable to several binary variables and merge it back to df
df = pd.concat([df,pd.get_dummies(df['Title']).rename(columns=lambda x: 'Title_' + str(x))],axis=1)

# Replace missing values with "U0"
df['Cabin'][df.Cabin.isnull()] = 'U0'
# create feature for the alphabetical part of the cabin number
df['CabinLetter'] = df['Cabin'].map( lambda x : re.compile("([a-zA-Z]+)").findall(x)[0])
# convert the distinct cabin letters with incremental integer values
df['CabinLetter'] = pd.factorize(df['CabinLetter'])[0]

# # Divide all fares into quartiles
# df['Fare_bin'] = pd.qcut(df['Fare'], 4)
# # qcut() creates a new variable that identifies the quartile range, but we can't use the string so either
# # factorize or create dummies from the result
# df['Fare_bin_id'] = pd.factorize(df['Fare_bin'])[0]


# chopping off data that will not be used
df.drop(['Name','Sex','Ticket','Cabin','PassengerId', 'Age', 'Embarked', 'Title'],inplace=True, axis = 1)

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
    median_Fare[i] = test_df.Fare[test_df.Pclass == i+1].median()
# The following code is selecting the rows in test_df that doesn't have a Fare value, and assign a value to it
# using its class as a criteria and using the median_Fare table that we computed before
# test_df.loc[test_df.Fare.isnull(), 'Fare'] = test_df.loc[test_df.Fare.isnull()].apply(lambda x: median_Fare[x.Pclass-1], axis = 1)
test_df.loc[test_df.Fare.isnull(),'Fare'] = test_df[test_df.Fare.isnull()].apply(lambda x: median_Fare[x.Pclass-1], axis = 1)

# Save the PassengerId for later use (generating file)... since it is not used as a parameter for our prediction model
test_ids = test_df.PassengerId

# Transforming Embarked categorical variable to several binary variables
test_df = pd.concat([test_df, pd.get_dummies(test_df.Embarked).rename(columns=lambda x: 'Embarked_' + str(x))], axis=1)

test_df['Names'] = test_df['Name'].map(lambda x: len(re.split(' ', x)))
# All titles in data is preceded by a ',' and is followed by a '.'
# The .*? makes the '*' operation non greedy!
test_df['Title'] = test_df['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
# Group low-occuring, related titles together
test_df['Title'][test_df.Title == 'Jonkheer'] = 'Master'
test_df['Title'][test_df.Title.isin(['Ms','Mlle'])] = 'Miss'
test_df['Title'][test_df.Title == 'Mme'] = 'Mrs'
test_df['Title'][test_df.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'
test_df['Title'][test_df.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'
# Changing Title variable to several binary variables and merge it back to test_df
test_df = pd.concat([test_df,pd.get_dummies(test_df['Title']).rename(columns=lambda x: 'Title_' + str(x))],axis=1)

# Replace missing values with "U0"
test_df['Cabin'][test_df.Cabin.isnull()] = 'U0'
# create feature for the alphabetical part of the cabin number
test_df['CabinLetter'] = test_df['Cabin'].map( lambda x : re.compile("([a-zA-Z]+)").findall(x)[0])
# convert the distinct cabin letters with incremental integer values
test_df['CabinLetter'] = pd.factorize(test_df['CabinLetter'])[0]
# # Divide all fares into quartiles
# test_df['Fare_bin'] = pd.qcut(test_df['Fare'], 4)
# # qcut() creates a new variable that identifies the quartile range, but we can't use the string so either
# # factorize or create dummies from the result
# test_df['Fare_bin_id'] = pd.factorize(test_df['Fare_bin'])[0]

# chopping off data that will not be used
test_df.drop(['Name','Sex','Ticket','Cabin','PassengerId', 'Age', 'Embarked', 'Title'],inplace=True, axis = 1)


'''
======= END of cleaning test data ====
'''



'''
======= Attempting to engineer some new features =======
'''
df['FareTimesPclass'] = df['Fare'] * (4-df['Pclass'])
test_df['FareTimesPclass'] = test_df['Fare'] * (4-test_df['Pclass'])

df['AgeFillTimesRelatives'] = df['AgeFill'] * df['Parch'] * df['SibSp']
test_df['AgeFillTimesRelatives'] = test_df['AgeFill'] * test_df['Parch'] * test_df['SibSp']







'''
======= Normalizing stuff =======
'''
# def normalize(a, b):
#     comb = pd.concat([a,b])
#     a = (a-comb.mean())/comb.std
#     b = (b-comb.mean())/comb.std
# normalizing stuff

# normalize(df.SibSp,test_df.SibSp)
# normalize(df.Parch,test_df.Parch)
# normalize(df.AgeFill,test_df.AgeFill)
# normalize(df.Embarked,test_df.Embarked)
# normalize(df.Fare,test_df.Fare)
# normalize(df.Pclass,test_df.Pclass)
# normalize(df.FareTimesPclass, test_df.FareTimesPclass)
# normalize(df.AgeTimesRelatives, test_df.AgeTimesRelatives)



# Since Gender and Embarked are categorical variables, we can't encode these to numbers arbitrarily
# Use one Hot encoder? from scikit-learn
enc = OneHotEncoder()
enc.fit(df[['Gender','Embarked','Pclass']])
NewGenderEmbarked = enc.transform(df[['Gender','Embarked','Pclass']]).toarray()
NewGenderEmbarked_test = enc.transform(test_df[['Gender','Embarked','Pclass']]).toarray()
df['GenderI'] = NewGenderEmbarked[::,0]
df['GenderII'] = NewGenderEmbarked[::,1]
df['EmbarkedI'] = NewGenderEmbarked[::,2]
df['EmbarkedII'] = NewGenderEmbarked[::,3]
df['EmbarkedIII'] = NewGenderEmbarked[::,4]
df['PclassI'] = NewGenderEmbarked[::,5]
df['PclassII'] = NewGenderEmbarked[::,6]
df['PclassIII'] = NewGenderEmbarked[::,7]

test_df['GenderI'] = NewGenderEmbarked_test[::,0]
test_df['GenderII'] = NewGenderEmbarked_test[::,1]
test_df['EmbarkedI'] = NewGenderEmbarked_test[::,2]
test_df['EmbarkedII'] = NewGenderEmbarked_test[::,3]
test_df['EmbarkedIII'] = NewGenderEmbarked_test[::,4]
test_df['PclassI'] = NewGenderEmbarked_test[::,5]
test_df['PclassII'] = NewGenderEmbarked_test[::,6]
test_df['PclassIII'] = NewGenderEmbarked_test[::,7]

df.drop(['Gender','Embarked','Pclass'],axis = 1, inplace = True)
test_df.drop(['Gender','Embarked','Pclass'],axis = 1, inplace = True)


comb_SibSp = pd.concat([df.SibSp, test_df.SibSp])
df.SibSp = (df.SibSp - comb_SibSp.mean())/comb_SibSp.std()
test_df.SibSp = (test_df.SibSp - comb_SibSp.mean())/comb_SibSp.std()
comb_Parch = pd.concat([df.Parch, test_df.Parch])
df.Parch = (df.Parch - comb_Parch.mean())/comb_Parch.std()
test_df.Parch = (test_df.Parch - comb_Parch.mean())/comb_Parch.std()

comb_AgeFill = pd.concat([df.AgeFill,test_df.AgeFill])
df.AgeFill = (df.AgeFill - comb_AgeFill.mean())/comb_AgeFill.std()
test_df.AgeFill = (test_df.AgeFill - comb_AgeFill.mean())/comb_AgeFill.std()

comb_Fare = pd.concat([df.Fare,test_df.Fare])
df.Fare = (df.Fare - comb_Fare.mean())/comb_Fare.std()
test_df.Fare = (test_df.Fare - comb_Fare.mean())/comb_Fare.std()

comb_1 = pd.concat([df.FareTimesPclass,test_df.FareTimesPclass])
df.FareTimesPclass = (df.FareTimesPclass - comb_1.mean())/comb_1.std()
test_df.FareTimesPclass = (test_df.FareTimesPclass - comb_1.mean())/comb_1.std()

comb_2 = pd.concat([df.AgeFillTimesRelatives,test_df.AgeFillTimesRelatives])
df.AgeFillTimesRelatives = (df.AgeFillTimesRelatives - comb_2.mean())/comb_2.std()
test_df.AgeFillTimesRelatives = (test_df.AgeFillTimesRelatives - comb_2.mean())/comb_2.std()


# comb_Pclass = pd.concat([df.Pclass,test_df.Pclass])
# df.Pclass = (df.Pclass - comb_Pclass.mean())/comb_Pclass.std()
# test_df.Pclass = (test_df.Pclass - comb_Pclass.mean())/comb_Pclass.std()
# comb_Embarked = pd.concat([df.Embarked,test_df.Embarked])
# df.Embarked = (df.Embarked - comb_Embarked.mean())/comb_Embarked.std()
# test_df.Embarked = (test_df.Embarked - comb_Embarked.mean())/comb_Embarked.std()





StandardDataCleaning = True




'''
======= Making first round of prediction using RF =======
'''
# Utility function to report best scores
def report(grid_scores, n_top=1):
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
              "max_features": sp_randint(5, 20),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(3, 11),
              "bootstrap": [True, False],
              "criterion": ["entropy"]}


print "Training..."
train_data = df.values
train_data = np.random.permutation(train_data[::,::])
temp = np.size(train_data,0)/5
cv_data = train_data[0:temp:,::]
train_data2 = train_data[temp::,::]


forest = RandomForestClassifier(n_estimators = 100)
# run randomized search
n_iter_search = 30
random_search = RandomizedSearchCV(forest, param_distributions=param_dist,
                                   n_iter=n_iter_search, cv=10)

start = time()
random_search.fit(train_data2[::,1::], train_data2[::,0])
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)
train_output = random_search.predict(train_data2[::,1::])
cv_output = random_search.predict(cv_data[::,1::])
print "Training set accuracy: %.3f   CV set accuracy: %.3f"\
      %(len(train_data2[train_output == train_data2[::,0]])/float(len(train_data2)),
      (len(cv_data[cv_output == cv_data[::,0]])/float(len(cv_data))))



# Analyzing important features
forest = random_search.best_estimator_
feature_importance = forest.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())

feature_list = df.columns.values



print "Predicting..."
test_data = test_df.values
output = random_search.predict(test_data).astype(int)
print "Done..."

'''
======= Creating new training data =======
'''
train_data2_X = np.concatenate([train_data2[::, 1::],test_data])
train_data2_Y = np.concatenate([train_data2[::, 0], output2])



'''
======= 2nd round of training and predicting =======
'''
print "Training..."
forest = RandomForestClassifier(n_estimators = 100)
# run randomized search
n_iter_search = 30
random_search = RandomizedSearchCV(forest, param_distributions=param_dist,
                                   n_iter=n_iter_search,cv=5)

start = time()
random_search.fit(train_data2_X, train_data2_Y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)

train_output = random_search.predict(train_data2_X)
cv_output = random_search.predict(cv_data[::,1::])
print "Round2: Training set accuracy: %.3f   CV set accuracy: %.3f"\
      %(len(train_data2_X[train_output == train_data2_Y])/float(len(train_data2_X)),
      (len(cv_data[cv_output == cv_data[::,0]])/float(len(cv_data))))




print "Predicting..."
output2 = random_search.predict(test_data).astype(int)
print "the two outputs are %.5f%% similar" % (len(output2[output2 == output])/float(len(output))*100)


'''
======= Making files =======
'''
prediction_file = open("July31_RF.csv", "wb")
open_file_object = csv.writer(prediction_file)
open_file_object.writerow(["PassengerId", "Survived"])
open_file_object.writerows(zip(test_ids,output))
prediction_file.close()

prediction_file = open("July27_RF_2.csv", "wb")
open_file_object = csv.writer(prediction_file)
open_file_object.writerow(["PassengerId", "Survived"])
open_file_object.writerows(zip(test_ids,output2))
prediction_file.close()


print 'Done'


'''
======= Using SVM =======
'''
print "Training..."
train_data = df.values
param_grid = {'C': [4, 5, 6, 10, 25],
              'gamma': [0.005, 0.001, 0.05, 0.01],
              'kernel': ['rbf']
              }
# run randomized search
n_iter_search = 30
random_search = RandomizedSearchCV(svm.SVC(class_weight='auto'), param_distributions = param_grid,
                                   n_iter=n_iter_search)
start = time()
random_search.fit(train_data[::,1::], train_data[::,0])
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)




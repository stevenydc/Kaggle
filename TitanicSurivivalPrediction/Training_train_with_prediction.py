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

# Changing categorical variable to numerical... which is required for most ML algorithms
Ports = list(enumerate(df['Embarked'].unique()))     #interesting way to create a list of enumerates
Ports_dict = {name:i for i,name in Ports}           # creating a dict so that we can map letters to values
df['Embarked'] = df['Embarked'].map(Ports_dict)

# chopping off data that will not be used
df.drop(['Name','Sex','Ticket','Cabin','PassengerId', 'Age'],inplace=True, axis = 1)

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
# test_df.loc[test_df.Fare.isnull(), 'Fare'] = test_df.loc[test_df.Fare.isnull()].apply(lambda x: median_Fare[x.Pclass-1], axis = 1)
test_df.loc[test_df.Fare.isnull(),'Fare'] = test_df[test_df.Fare.isnull()].apply(lambda x: median_Fare[x.Pclass-1], axis = 1)

# Save the PassengerId for later use (generating file)... since it is not used as a parameter for our prediction model
test_ids = test_df.PassengerId
# Now we can drop everything in that test_df that we don't use
test_df.drop(['Cabin','Age','PassengerId','Name','Sex','Ticket'], inplace = True, axis = 1)

'''
======= END of cleaning test data ====
'''


'''
======= Normalizing stuff =======
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

comb_Pclass = pd.concat([df.Pclass,test_df.Pclass])
df.Pclass = (df.Pclass - comb_Pclass.mean())/comb_Pclass.std()
test_df.Pclass = (test_df.Pclass - comb_Pclass.mean())/comb_Pclass.std()

comb_Embarked = pd.concat([df.Embarked,test_df.Embarked])
df.Embarked = (df.Embarked - comb_Embarked.mean())/comb_Embarked.std()
test_df.Embarked = (test_df.Embarked - comb_Embarked.mean())/comb_Embarked.std()


StandardDataCleaning = True
'''
======= Making first round of prediction using RF =======
'''
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
              "max_features": sp_randint(3, 7),
              "min_samples_split": sp_randint(4, 7),
              "min_samples_leaf": sp_randint(3, 8),
              "bootstrap": [True, False],
              "criterion": ["entropy"]}


print "Training..."
train_data = df.values
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

print "Predicting..."
test_data = test_df.values
output = random_search.predict(test_data).astype(int)
print "Done..."

'''
======= Creating new training data =======
'''
train_data2_X = np.concatenate([train_data[::, 1::],test_data])
train_data2_Y = np.concatenate([train_data[::, 0], output])



'''
======= 2nd round of training and predicting =======
'''
print "Training..."
forest = RandomForestClassifier(n_estimators = 100)
# run randomized search
n_iter_search = 30
random_search = RandomizedSearchCV(forest, param_distributions=param_dist,
                                   n_iter=n_iter_search)

start = time()
random_search.fit(train_data2_X, train_data2_Y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)

print "Predicting..."
output2 = random_search.predict(test_data).astype(int)
print "the two outputs are %.5f%% similar" % (len(output2[output2 == output])/float(len(output))*100)



'''
======= Making files =======
'''
prediction_file = open("July27_RF_1.csv", "wb")
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



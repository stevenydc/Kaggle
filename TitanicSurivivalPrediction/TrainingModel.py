__author__ = 'stevenydc'
import csv as csv
import numpy as np


csv_file_object = csv.reader(open('train.csv','rb'))
header = csv_file_object.next()

data = []
for row in csv_file_object:
    data.append(row)
data = np.array(data)


'''
============= Gender Based Model ============
'''
# num_passenger = len(data)
# num_survivor = sum(data[::,1].astype(np.float))
# proportion_survivor = num_survivor/float(num_passenger)
#
#
# women_passenger = data[::,4] == 'female'
# men_passenger = data[::,4] != 'female'
#
# num_women = len(data[women_passenger])        # women_passenger is an array! different from list.
#                                               # so women_passenger is like a mask.... blocks false values
# num_men = len(data[men_passenger])            # data[women_passenger] picks out the rows where women_passenger = True
#
# num_women_survivor = sum(data[women_passenger,1].astype(float))
# num_men_survivor = sum(data[men_passenger,1].astype(float))
#
# proportion_men_survivor = num_men_survivor/num_men
# proportion_women_survivor = num_women_survivor/num_women
#
# print "Proportion of women who survived was %f" % proportion_women_survivor
# print "Proportion of men who survived was %s" % proportion_men_survivor
#
#
# test_file = open('test.csv', 'rb')
# test_file_object = csv.reader(test_file)
# header = test_file_object.next()
#
# prediction_file = open("GenderModelPrediction.csv","wb")
# prediction_file_object = csv.writer(prediction_file)
#
# #Create the header
# prediction_file_object.writerow(["PassengerId","Survived"])
#
#
# for row in test_file_object:
#     if row[3] == "female":
#         prediction_file_object.writerow([row[0],1])
#     else:
#         prediction_file_object.writerow([row[0],0])
#

'''
============= END of Gender Based Model =========
'''







'''
============= Beginning of model with several variables =========
'''
fare_ceiling = 40
data[data[::,9].astype(np.float) >= fare_ceiling, 9] = fare_ceiling-1

fare_bracket_size = 10
number_of_price_brackets = fare_ceiling/fare_bracket_size
number_of_classes = len(np.unique(data[::,2]))

survival_table = np.zeros([2,number_of_classes,number_of_price_brackets])

for i in range(number_of_classes):
    for j in range(number_of_price_brackets):
        women_only_stats = data[(data[::,4] == 'female') &
                                (data[::,2].astype(np.float) == i+1) &
                                (data[::,9].astype(np.float) >= j*fare_bracket_size) &
                                (data[::,9].astype(np.float) <  (j+1)*fare_bracket_size), 1]
        men_only_stats = data[(data[::,4] != 'female') &
                              (data[::,2].astype(np.float) == i+1) &
                              (data[::,9].astype(np.float) >= j*fare_bracket_size) &
                              (data[::,9].astype(np.float) <  (j+1)*fare_bracket_size), 1]
        survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float))
        survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))


survival_table[survival_table != survival_table] = 0
survival_table[survival_table < 0.5] = 0
survival_table[survival_table >= 0.5] = 1

# setting up test, and output file
test_file = open('test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()
prediction_file = open('genderClassFareModel.csv','wb')
prediction_file_object = csv.writer(prediction_file)
prediction_file_object.writerow(["PassengerId", "Survived"])

binfare = 0
for row in test_file_object:
    try:
        row[8] = float(row[8])
    except:
        binfare = number_of_price_brackets-float(row[1])       # the higher (1st being highest) the class,
        row[8] = binfare*fare_bracket_size                     # the higher up the price bracket
    for i in range(number_of_price_brackets):
        if row[8] >= fare_ceiling:
            binfare = number_of_price_brackets-1
            break
        if (row[8] >= i*fare_bracket_size) and (row[8] < (i+1)*fare_bracket_size):
            binfare = i
            break
    if row[3] == 'female':
        prediction_file_object.writerow([row[0], int(survival_table[0,int(row[1])-1,binfare])])
    else:
        prediction_file_object.writerow([row[0], int(survival_table[1,int(row[1])-1,binfare])])

test_file.close()
prediction_file.close()

'''
============= End of model with several variables =========
'''








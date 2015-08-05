train <- read.csv("~/Documents/Kaggle/TitanicSurivivalPrediction/train.csv")
test <- read.csv("~/Documents/Kaggle/TitanicSurivivalPrediction/test.csv")
test$Survived <- rep(0,418)

train$Child <- 0
train$Child[train$Age < 18] <- 1

aggregate(Survived ~ Child+Sex, data=train,FUN=sum)
aggregate(Survived ~ Child+Sex+Pclass, data=train,FUN=length)
aggregate(Survived ~ Child+Sex, data=train,FUN=function(x) {sum(x)/length(x)})

library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)

fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train, 
             method="class", control = rpart.control(minsplit=2,cp=0))
fancyRpartPlot(fit)

test$Survived <- NA
combi <- rbind(train,test)
combi$Name <- as.character(combi$Name)
strsplit(combi$Name[1],split='[,.]')[[1]][2]
combi$Title <- sapply(combi$Name, FUN = function(x) {strsplit(x,split='[,.]')[[1]][2]})
combi$Title <- sub(' ','',combi$Title)

combi$Title[combi$Title %in% c('Mme','Mlle')] <- 'Mlle'
combi$Title[combi$Title %in% c('Capt','Don','Major','Sir')] <- 'Sir'
combi$Title[combi$Title %in% c('Dona','Lady','the Countess','Jonkheer')] <- 'Lady'

combi$Title <- factor(combi$Title)

combi$FamilySize <- combi$SibSp + combi$Parch +1

# Want to find last names... so that we can create familyID
combi$Surname <- sapply(combi$Name, FUN = function(x) {strsplit(x,split='[,.]')[[1]][1]})
combi$FamilyID <- paste(as.character(combi$FamilySize),combi$Surname,sep='')
combi$FamilyID[combi$FamilySize <= 2] <- 'Small'

# There are FamilyID's that only have count of 1 or 2.. which means that
# their family members might have had different last names
famIDs <- data.frame(table(combi$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 2,]
# Now we clean these IDs out from FamilyID
combi$FamilyID[combi$FamilyID %in% famIDs$Var1] <- 'Small'
combi$FamilyID <- factor(combi$FamilyID)

# Now we can split the combined dataset with newly engineered features
train <- combi[1:891,]
test <- combi[892:1309,]

fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked +
             Title + FamilySize + FamilyID, 
             data=train, 
             method="class")

Prediction <- predict(fit,test,type='class')

# Replace missing Age values using a decision tree!
Agefit <- rpart(Age ~ Pclass+Sex+SibSp+Parch+Fare+Embarked+Title+FamilySize,
                data=combi[!is.na(combi$Age),], method='anova')
combi$Age[is.na(combi$Age)] <- predict(Agefit,combi[is.na(combi$Age),])

combi$Embarked[c(which(combi$Embarked==''))] = 'S'
combi$Embarked <- factor(combi$Embarked)

combi$Fare[which(is.na(combi$Fare))] <- median(combi$Fare, na.rm=TRUE)

# reduce the number of levels of the FamilyID factor to less than 32
# becuase RandomForest only takes that many levels
combi$FamilyID2 <- as.character(combi$FamilyID)
combi$FamilyID2[combi$FamilySize <= 3] <- 'Small'
combi$FamilyID2 <- factor(combi$FamilyID2)

library(randomForest)

# to ensure the same randomness everytime
set.seed(415)

fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked +
               Title + FamilySize + FamilyID2, 
             data=train, importance=TRUE, ntree=2000)

varImpPlot(fit)

# this library is a forest of conditional inference trees
library(party)
Prediction <- predict(fit,test)


fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked +
                      Title + FamilySize + FamilyID, 
                    data=train, controls=cforest_unbiased(ntree=2000, mtry=3))

Prediction2 <- predict(fit,test,OOB=TRUE, type='response')

length(Prediction2[Prediction != Prediction2])

submit<- data.frame(PassengerId = test$PassengerId, Survived=Prediction2)
write.csv(submit,file="CForest.csv",row.names=FALSE)



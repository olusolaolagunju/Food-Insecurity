
library(caret)
library(xgboost)
library(tidyverse)
library(readr)
require(Matrix)
require(data.table)
library(ROSE)
if (!require('vcd')) install.packages('vcd')
library("mltools")
library("caret") 
library("MASS")
library("MLmetrics")

Food_insecurity <-  read.csv("C:/Users/DELL/OneDrive/COURSERA COURSES/GOOGLE ANALYTIC/R/Google Data Analytic/Food_security/Food_security/Data/tidy_data/Food_insecurity_data.csv")
Food_insecurity <- Food_insecurity %>%
  dplyr:: select(-c(9:21))


# Food_insecurity <- Food_insecurity %>%
#   dplyr::select(-c(1, 3:4))

set.seed(1789)

# changing the status

Food_insecurity$gender <-  as.factor(Food_insecurity$gender)
Food_insecurity$age <-  as.factor(Food_insecurity$age)
Food_insecurity$education <-  as.factor(Food_insecurity$education)
Food_insecurity$employment_status <-  as.factor(Food_insecurity$employment_status)
Food_insecurity$income <-  as.factor(Food_insecurity$income)
Food_insecurity$household_size <-  as.factor(Food_insecurity$household_size)
Food_insecurity$children <-  as.factor(Food_insecurity$children)
Food_insecurity$lg <-  as.factor(Food_insecurity$lg)
Food_insecurity$status<-  as.factor(Food_insecurity$status)
contrasts(Food_insecurity$status)
Food_insecurity$status <- ifelse(Food_insecurity$status == "Food Insecure", "FI", "FS")


# multiply the high class by 2 (723 *2)
over <- ovun.sample(status~., data = Food_insecurity, method = "over", N = 1446)$data


Food_insecurity_matrix <- sparse.model.matrix(status ~ ., data = Food_insecurity)[,-1]
Food_insecurity_matrix_smote <- sparse.model.matrix(status ~ ., data = over)[,-1] 


# Create the output numeric vector for the LABEL (not as a sparse Matrix):

output_vector = Food_insecurity[,"status"] == "FI"
output_vector_smote = over[,"status"] == "FI"

numberOfTrainingSamples <- round(length(output_vector) * .7)
numberOfTrainingSamples_smote <- round(length(output_vector_smote) * .7)

# training data
train_data <- Food_insecurity_matrix_smote[1:numberOfTrainingSamples_smote,]
train_labels <- output_vector_smote[1:numberOfTrainingSamples_smote] # 279 data points

# testing data
test_data <- Food_insecurity_matrix[-(1:numberOfTrainingSamples),]
test_labels <- output_vector[-(1:numberOfTrainingSamples)] # 110 data points 115


# Convert the cleaned dataframe to a dmatrix
# The very final step is to convert our matrixes into dmatrix objects. This step isn't absolutely necessary, but it will help our model train move more quickly, and you'll need to to this if you ever want to train a model on multiple cores.

# put our testing & training data into two seperates Dmatrixs objects
dtrain <- xgb.DMatrix(data = train_data, label= train_labels)
dtest <- xgb.DMatrix(data = test_data, label= test_labels)


model <- xgboost(data = dtrain, # the data
                 nround = 10, # max number of boosting iterations
                 eta = 1,
                 max_depth = 6,
                 objective = "binary:logistic")
pred <- predict(model, dtest)

# get & print the classification error
err <- mean(as.numeric(pred > 0.5) != test_labels)

print(paste("test-error=", err)) 
pred_label <- ifelse(pred >= 0.5, "TRUE", "FALSE")

confusionMatrix(factor(test_labels),factor(pred_label))


#MCC
MCC_XGBoost_full_over <- mcc(TP= 214,
                             FP=3,
                             TN=49,
                             FN=4)

# MCC_XGBoost_reduced_over <- mcc(TP= 214,
#                              FP=8,
#                              TN=44,
#                              FN=4)


importance_matrix <- xgb.importance(names(Food_insecurity_matrix), model = model)

importance_matrix
importanceClean <- importance_matrix[,`:=`(Cover=NULL, Frequency=NULL)]

head(importanceClean)


# and plot it!

xgb.plot.importance(importance_matrix)





ctab_test <- table(test_labels, pred_label)
ctab_test



Recall <- (ctab_test[2, 2]/sum(ctab_test[2, ]))*100
Recall 


Precision <- (ctab_test[2, 2]/sum(ctab_test[, 2]))*100
Precision 
F_Score <- (2 * Precision * Recall / (Precision + Recall))/100
F_Score

library(pROC)
roc <- roc(test_labels, pred)

auc(roc)

plot(roc) 
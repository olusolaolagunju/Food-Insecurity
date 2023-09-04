# HYPER PARAMTER: https://towardsdatascience.com/a-guide-to-xgboost-hyperparameters-87980c7f44a9

library(caret)

library(xgboost)
library(tidyverse)
library(readr)
require(Matrix)
require(data.table)
if (!require('vcd')) install.packages('vcd')
library("mltools")
library("caret") 
library("MASS")
library("MLmetrics")

Food_insecurity <-  read.csv("C:/Users/DELL/OneDrive/COURSERA COURSES/GOOGLE ANALYTIC/R/Google Data Analytic/Food_security/Food_security/Data/tidy_data/5foodsec_label.csv")

Food_insecurity %>% 
  group_by(status) %>% 
  summarise(st = n()) %>% 
  mutate(status_freq = paste0(round((100 * st/sum(st)), 0), '%'))


Food_insecurity <- Food_insecurity %>% 
 dplyr:: select(-c(9:21))
 Food_insecurity <- Food_insecurity %>%
  dplyr:: select(-c(1, 3:4))
set.seed(1789)
# 
# Food_insecurity <- Food_insecurity[sample(1:nrow(Food_insecurity)), ]
# changing the status
Food_insecurity$status <- ifelse(Food_insecurity$status == "Food Insecure", "FI", "FS")
# view(Food_insecurity)
# contrasts(Food_insecurity$status)
#Food_insecurity$gender <-  as.factor(Food_insecurity$gender)
Food_insecurity$age <-  as.factor(Food_insecurity$age)
Food_insecurity$education <-  as.factor(Food_insecurity$education)
Food_insecurity$employment_status <-  as.factor(Food_insecurity$employment_status)
Food_insecurity$income <-  as.factor(Food_insecurity$income)
Food_insecurity$household_size <-  as.factor(Food_insecurity$household_size)
Food_insecurity$children <-  as.factor(Food_insecurity$children)
Food_insecurity$lg <-  as.factor(Food_insecurity$lg)
Food_insecurity$status<-  as.factor(Food_insecurity$status)

Food_insecurity_matrix <- sparse.model.matrix(status ~ ., data = Food_insecurity)[,-1]


# Create the output numeric vector for the LABEL (not as a sparse Matrix):

output_vector = Food_insecurity[,"status"] == "FI"


numberOfTrainingSamples <- round(length(output_vector) * .7)

# training data

train_data <- Food_insecurity_matrix[1:numberOfTrainingSamples,]
train_labels <- output_vector[1:numberOfTrainingSamples] # 279 data points

# testing data
test_data <- Food_insecurity_matrix[-(1:numberOfTrainingSamples),]
test_labels <- output_vector[-(1:numberOfTrainingSamples)] # 110 data points 115


# Convert the cleaned dataframe to a dmatrix
# The very final step is to convert our matrixes into dmatrix objects. This step isn't absolutely necessary, but it will help our model train move more quickly, and you'll need to to this if you ever want to train a model on multiple cores.

# put our testing & training data into two seperates Dmatrixs objects
dtrain <- xgb.DMatrix(data = train_data, label= train_labels)
dtest <- xgb.DMatrix(data = test_data, label= test_labels)


# PART2 : TRAINING OUR MODEL
#
# train a model using our training data
# model <- xgboost(data = dtrain,max.depth = 3,
#                  eta = 1, nthread = 3, # the data
#                  nround = 5, # max number of boosting iterations
#                  objective = "binary:logistic")  # the objective function
# # train a model using our training data


model <- xgboost(data = dtrain, # the data
                 nround = 10, # max number of boosting iterations
                 eta = 1,
                 #booster = "gblinear",
                 max_depth = 6,
                 #nthread = 2,
                 objective = "binary:logistic")


pred <- predict(model, dtest)
# pred_train <- predict(model, dtrain)

# get & print the classification error
err <- mean(as.numeric(pred > 0.5) != test_labels)
# err_train  <- mean(as.numeric(pred_train > 0.5) != train_labels)

print(paste("test-error=", err)) #"test-error= 0.107246376811594
pred_label <- ifelse(pred >= 0.5, "TRUE", "FALSE")
# pred_train <- ifelse(pred_train >= 0.5, "TRUE", "FALSE")
confusionMatrix(factor(test_labels),factor(pred_label))

MCC_XGBoost_full <- mcc(TP= 216,
                           FP=11,
                           TN=41,
                           FN=2)# MCC


MCC_XGBoost_reduced <- mcc(TP= 217,
                                FP=16,
                                TN=36,
                                FN=1)

#Sensitivity = POS PRED VALUE

# confusionMatrix(factor(train_labels),factor(pred_train))
## # get the number of negative & positive cases in our data
negative_cases <- sum(train_labels == FALSE)
postive_cases <- sum(train_labels == TRUE)


# model_tuned <- xgboost(data = dtrain, # the data           
#                        max.depth = 3, # the maximum depth of each decision tree
#                        nround = 10, # number of boosting rounds
#                        early_stopping_rounds = 3, # if we dont see an improvement in this many rounds, stop
#                        objective = "binary:logistic", # the objective function
#                        scale_pos_weight = negative_cases/postive_cases) # control for imbalanced classes
# 
# # generate predictions for our held-out testing data
# pred <- predict(model_tuned, dtest) # or use model_tuned
# 
# # get & print the classification error
# err <- mean(as.numeric(pred > 0.5) != test_labels)
# print(paste("test-error=", err)) #"test-error= 0.219512195121951"


# There are a couple things to notice here.
# 
# First, our error in the first round was actually higher than it was for earlier models (0.016... vs 0.014...). This is because we've penalized failing to capture very rare events.
# 
# Then, as we add more training rounds, our error drops a little bit and actually ends up lower than it was with our earlier model. This is because adding additional training rounds adds additional complexity to our model that better allows it to capture the variation in our training data.
# 
# After a while, though, our error starts to actually go up. This is probably due to over-fitting: we end up at the point where adding more complexity to the model is actually hurting it. We've talked about avoiding over-fitting above, but another technique that can help avoid over-fitting is adding a regularization term, gamma. Gamma is a measure of how much an additional split will need to reduce loss in order to be added to the ensemble. If a proposed model does not reduce loss by at least whatever-you-set-gamma-to, it won't be included. Here, I'll set it to one, which is fairly high. (By default gamma is zero.)
# 


# train a model using our training data
# model_tuned <- xgboost(data = dtrain, # the data           
#                        max.depth = 3, # the maximum depth of each decision tree
#                        nround = 10, # number of boosting rounds
#                        early_stopping_rounds = 3, # if we dont see an improvement in this many rounds, stop
#                        objective = "binary:logistic", # the objective function
#                        scale_pos_weight = negative_cases/postive_cases, # control for imbalanced classes
#                        gamma = 1) # add a regularization term
# 
# # generate predictions for our held-out testing data
# x_pred <- predict(model_tuned, dtest)
# 
# # get & print the classification error
# err <- mean(as.numeric(x_pred > 0.5) != test_labels)
# print(paste("test-error=", err)) # 0.219512195121951"

# Adding a regularization terms makes our model more conservative, so it doesn't end up adding the models which were reducing our accuracy.
# 
# We've done quite a bit of parameter turning at this point, but you may have noticed that it didn't actually help our accuracy on the test set! The sensible defaults for xgboost are doing a pretty good job on their own. If you have a larger and more complex dataset you'll probably get more utility out of parameter tuning, but for this problem it looks like the simple model is actually just as useful as the more complex one.


# Examining our model

# So far, we've:
# 
# cleaned & prepared our data
# trained our model
# tuned our model (not strictly necessary in this case, but generally it will help!)
# Now we can spend some time examing and interpreting our model. One of the really nice things about xgboost is that is has a lot of built-in functions to help us figure out why our model is making the distictions it's making.
# 
# One way that we can examine our model is by looking at a representation of the combination of all the decision trees in our model. Since all the trees have the same depth (remember that we set that with a parameter!) we can stack them all on top of one another and pick the things that show up most often in each node.


# plot them features! what's contributing most to our model?
#' xgb.plot.multi.trees(feature_names = names(Food_insecurity_matrix), 
#'                       model = model)
#' 
#' xgb.plot.multi.trees(feature_names = names(Food_insecurity_matrix), 
#' 
#'                      features.keep = 3, model = model)
#' #' The top of the tree is on the left and the bottom of the tree is on the right.
#'  For features, the number next to it is "quality", which helps indicate how important
#'  this feature was across all the trees. Higher quality means a feature was more 
#'  important. So we can tell that is_domestic was by far the most important feature
#'  across all of our trees, both because it's higher in the tree and also because 
#'  it's quality score is very high.
# 
# For the nodes with "Leaf", the number next to the "Leaf" is the average value the m
#' model returned across all trees if a a certain observation ended up in that leaf. 
#' Because we're using a logistic model here, it's telling us the log-odds rather than 
#' the probability. We can pretty easily convert the log odds to probability, though.

#' # convert log odds to probability
#' odds_to_probs <- function(odds){
#'   return(exp(odds)/ (1 + exp(odds)))
#' }
#' 
#' 
#' # probability of leaf above countryPortugul
#' odds_to_probs(0.57635)
#' #' So, in the trees where an observation ended up in that leaf, on average the 
#' probability that a human would be sick in that instance was 35%. Since that was below 
#' the threshold of 50% we used for our decision rule, we'd say that these instance 
#' usually wouldn't result in a human getting sick.
# 
# What if we want a quick way to see which features are most important? We can do that 
# using by creating and then plotting the importance matrix, like so:

# get information on how important each feature is

importance_matrix <- xgb.importance(names(Food_insecurity_matrix), model = model)

importance_matrix
importanceClean <- importance_matrix[,`:=`(Cover=NULL, Frequency=NULL)]

head(importanceClean)


# and plot it!

xgb.plot.importance(importance_matrix)

#' Here, each bar is a different feature, and the x-axis is plotting the weighted gain. 
#' ("Weighted" just means that if you add together the gain for every feature, you'll get
#'  1.) Basically this plot tells us how informative each feature was when we look at
#'  every tree in our ensemble. So features with a lot of gain were very important 
#'  to our model while features with less gain were less helpful.
# 
pred_label <- ifelse(pred >= 0.5, "TRUE", "FALSE")

confusionMatrix(factor(test_labels),factor(pred_label))
# CONFUSION MATRIX
#' 1. RECALL/SENSITIVITY = NEG PRED VALUE
#' 2. SPECIFICITY = POS PRED VALUE
#' 3. PRECISION = SPECIFICITY

ctab_test <- table(test_labels, pred_label)
ctab_test


#' Training dataset converting from probability to class values


#Our logistics model is able to classify 80.54% of all the observations correctly in the test dataset.
# Accuracy in Test dataset. This shows that our model is performing good
accuracy_test <- sum(diag(ctab_test))/sum(ctab_test)*100

accuracy_test  #86.55462 # 85.71429 #0.887


#Misclassification Rate = (FP+FN)/(TN + FP + FN + TP)
#Recall Or TPR = TP/(FN + TP)
# Recall in Train dataset

#RECALL/SENSITIVITY =NED PRED VALUE
Recall <- (ctab_test[2, 2]/sum(ctab_test[2, ]))*100
Recall #  97.9798  Neg Pred Value #96970

# True Negative Rate: Pos Pred Value
# TNR indicates how often does our model predicts actual nonevents from the overall 
#nonevents.

#SPECIFICITY =Pos Pred Value

TNR <- (ctab_test[1, 1]/sum(ctab_test[1, ]))*100
TNR # 30


# Precision
# Precision indicates how often does your predicted TRUE values are actually TRUE.


# Precision = Specificity
Precision <- (ctab_test[2, 2]/sum(ctab_test[, 2]))*100
Precision # 0.87387 #87.27273

# Calculating F-Score
# F-Score is a harmonic mean of recall and precision. The score value lies between 0 and 1. The value of 1 represents perfect precision & recall. The value 0 represents the worst case.

F_Score <- (2 * Precision * Recall / (Precision + Recall))/100
F_Score # 0.9238095

# ROC Curve
# The area under the curve(AUC) is the measure that represents ROC(Receiver Operating Characteristic) curve. This ROC curve is a line plot that is drawn between the Sensitivity and (1 – Specificity) Or between TPR and TNR. This graph is then used to generate the AUC value. An AUC value of greater than .70 indicates a good model.
library(pROC)
roc <- roc(test_labels, pred)

auc(roc)

plot(roc) # 0.7677


# CONFUSION MATRIX
#' 1. RECALL = NEG PRED VALUE
#' 2. SPECIFICITY = POS PRED VALUE
#' 3. PRECISION = SPECIFICITY
# TESTING PREDICTION AND CLASSIFICATION ACCURACY
#measure prediction accuracy
mean((test_labels - pred)^2) #mse  0.1619212
caret::MAE(test_labels, pred) #mae #  0.3829583

caret::RMSE(test_labels, pred) #rmse  0.4023943

#Model accuaracy
mean(test_labels == pred_label)
# model evaluation. original outcome - predicted outcome
residuals = test_labels - pred

RMSE = sqrt(mean(residuals^2)) # 0.4023943

cat('The root mean square error of the test data is ', round(RMSE,3),'\n')

y_test_mean = mean(test_labels)

# Calculate total sum of squares
tss =  sum((test_labels - y_test_mean)^2 )

# Calculate residual sum of squares
rss =  sum(residuals^2)

# Calculate R-squared
rsq  =  1 - (rss/tss)
cat('The R-square of the test data is ', round(rsq,3), '\n') # -0.102 




# Important featuress


#library(forcats) # for fct_rorder
Feature <-  c("Income", "Household size", "Children", "Lg", "Age", "Employemnt status", "Gender", "Education")
Gain <- c(0.4301,0.1146, 0.0643, 0.1995,0.0679, 0.0371, 0.0400 , 0.0465)
Impt_feat <- tibble(Feature, Gain)

dev.off() # to reset ggplot
ggplot(Impt_feat, aes(x = fct_reorder(Feature, Gain), y = Gain, fill = Feature))+
  geom_bar(stat='identity')+
  coord_flip()+
  scale_fill_manual(values=c("lightslategrey", "lightslategrey","lightslategrey", "lightslategrey","lightslategrey","lightslategrey", "lightblue", "lightslategrey"))+
  geom_text(aes(label = round((Gain),2)))+
  labs(x= "Features",
       y = "Gain",
       title = "Important XGBoost model features",
       subtitle= "Features with more than 0.1 'Gain', contribute the most to the model accuracy")+
  theme_classic()+
  theme(legend.position="none")+
  theme(panel.spacing = unit(1, "lines"))







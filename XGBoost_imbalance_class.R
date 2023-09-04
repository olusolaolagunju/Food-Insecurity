Food_insecurity <-  read.csv("../Food_insecurity_data.csv")
Food_insecurity <- Food_insecurity %>%
  dplyr:: select(-c(9:21))

set.seed(1789)

#convert variables to factors
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


# Change the dependnet variable into a sparse matrix
Food_insecurity_matrix <- sparse.model.matrix(status ~ ., data = Food_insecurity)[,-1]
output_vector = Food_insecurity[,"status"] == "FI"
numberOfTrainingSamples <- round(length(output_vector) * .7)
train_data <- Food_insecurity_matrix[1:numberOfTrainingSamples,]
train_labels <- output_vector[1:numberOfTrainingSamples] # 279 data points
# testing data
test_data <- Food_insecurity_matrix[-(1:numberOfTrainingSamples),]
test_labels <- output_vector[-(1:numberOfTrainingSamples)] # 110 data points 115
# put our testing & training data into two seperates Dmatrixs objects
dtrain <- xgb.DMatrix(data = train_data, label= train_labels)
dtest <- xgb.DMatrix(data = test_data, label= test_labels)
model <- xgboost(data = dtrain, # the data
                 nround = 10, # max number of boosting iterations
                 eta = 1,
                 max_depth = 6,
                 objective = "binary:logistic")
pred <- predict(model, dtest)
err <- mean(as.numeric(pred > 0.5) != test_labels)
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
# confusionMatrix(factor(train_labels),factor(pred_train))

importance_matrix <- xgb.importance(names(Food_insecurity_matrix), model = model)
importance_matrix
importanceClean <- importance_matrix[,`:=`(Cover=NULL, Frequency=NULL)]
head(importanceClean)


# and plot it!

xgb.plot.importance(importance_matrix)
 
pred_label <- ifelse(pred >= 0.5, "TRUE", "FALSE")

confusionMatrix(factor(test_labels),factor(pred_label))

ctab_test <- table(test_labels, pred_label)

Recall <- (ctab_test[2, 2]/sum(ctab_test[2, ]))*100
Recall #  97.9798  Neg Pred Value #96970


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






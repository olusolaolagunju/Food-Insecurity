library("mltools")
library("caret") 
library("MASS")
library("MLmetrics")
library(sjPlot) # models
library(sjmisc)# regression model
library(sjlabelled) # regression model
sessionInfo()
library(tidyverse)

Food_insecurity <-  read.csv("../Food_insecurity_data.csv")

# # food security percentage 
 
Food_insecurity %>%
  group_by(status) %>%
  summarise(st = n()) %>%
  mutate(status_freq = paste0(round((100 * st/sum(st)), 0), '%'))

# # select important columns 
# colnames(Food_insecurity)

Food_insecurity <- Food_insecurity %>% 
  dplyr::select(-c(9:21))

# Food_insecurity <- Food_insecurity %>%
#   dplyr::select(-c(1,3:4))


# Change all the columns to factors 
Food_insecurity$gender  <-  as.factor(Food_insecurity$gender)
Food_insecurity$age  <-  as.factor(Food_insecurity$age)
Food_insecurity$education   <-  as.factor(Food_insecurity$education)
Food_insecurity$employment_status <-  as.factor(Food_insecurity$employment_status)
Food_insecurity$income  <-  as.factor(Food_insecurity$income)
Food_insecurity$household_size <-  as.factor(Food_insecurity$household_size)
Food_insecurity$children  <-  as.factor(Food_insecurity$children)
Food_insecurity$lg  <-  as.factor(Food_insecurity$lg)
Food_insecurity$status <-  as.factor(Food_insecurity$status)

# checking the contrast of status 

contrasts(Food_insecurity$status)
Food_insecurity$status <-  relevel(Food_insecurity$status, ref = "Food secure")
set.seed(1234)


# Splitting the data into train and test
index2 <- createDataPartition(Food_insecurity$status, p = .70, list = FALSE)
train2 <- Food_insecurity[index2, ] # 70% DATA
food_test <- Food_insecurity[-index2, ] # 30 % DATA #119
colnames(food_test)
train2%>%
  group_by(status) %>%
  summarise(st = n()) %>%
  mutate(status_freq = paste0(round((100 * st/sum(st)), 0), '%'))
     

# Training the model to include all the predictors
food_model <- glm(status ~ ., family = binomial(), train2)

summary(food_model)
tab_model(food_model) # ODD RATIO

pred_prob <- predict(food_model, food_test, type = "response")

## Converting from probability to actual output. that is, convert the outcome as either Food secure or insecure (0, 1)
# Converting from probability to actual output
food_test$pred_class <- ifelse(pred_prob >= 0.5, "Food_secure ", "Food_insecure ")


# Generating the classification table
ctab_test <- table(food_test$status, food_test$pred_class)
ctab_test
accuracy_test <- sum(diag(ctab_test))/sum(ctab_test)*100
accuracy_test

Recall <- (ctab_test[2, 2]/sum(ctab_test[2, ]))*100
Recall

TNR <- (ctab_test[1, 1]/sum(ctab_test[1, ]))*100
TNR 

Precision <- (ctab_test[2, 2]/sum(ctab_test[, 2]))*100
Precision

# Calculating F-Score
# F-Score is a harmonic mean of recall and precision. The score value lies between 0 and 1. The value of 1 represents perfect precision & recall. The value 0 represents the worst case.

F_Score <- (2 * Precision * Recall / (Precision + Recall))/100
F_Score 

# ROC Curve
# The area under the curve(AUC) is the measure that represents ROC(Receiver Operating Characteristic) curve. This ROC curve is a line plot that is drawn between the Sensitivity and (1 – Specificity) Or between TPR and TNR. This graph is then used to generate the AUC value. An AUC value of greater than .70 indicates a good model.
library(pROC)
roc <- roc(train2$status, food_model$fitted.values)
auc(roc)
plot(roc) 



#MCC
MCC_Logistic_full <- mcc(TP= 206,
                        FP=29,
                        TN=24,
                        FN=10)

# MCC_Logistic_reduced <- mcc(TP= 202,
#                          FP=28,
#                          TN=25,
#                          FN=14)



#library(forcats) # for fct_rorder
Feature <-  c("Income", "Lg", "Children", "Employemnt status")
Walds<- c(95.8, 14.9, 5.5,  4.8 )
Impt_feat <- tibble(Feature, Walds)
str(Impt_feat)
dev.off() # to reset ggplot
ggplot(Impt_feat, aes(x = fct_reorder(Feature, Walds), y = Walds, fill = Feature))+
  geom_bar(stat='identity')+
  coord_flip()+
  scale_fill_manual(values=c("lightslategrey", "lightslategrey","lightblue", "lightslategrey"))+
  geom_text(aes(label = Walds))+
  labs(x= "Features",
       y = "Walds χ2",
       title = "Important Logistic Regression model features")+
  theme_classic()+
  theme(legend.position="none")+
  theme(panel.spacing = unit(1, "lines"))


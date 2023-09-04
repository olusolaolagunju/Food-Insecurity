# https://www.r-bloggers.com/2021/05/class-imbalance-handling-imbalanced-data-in-r/
#install.packages("randomForest")
#install.packages("ROSE")
# install.packages("e1071")
# install.packages("pROC")
library(randomForest)
library(ROSE)
library(randomForest)
library(caret)
library(e1071)
library(pROC)
library(ROCR)
library("mltools")
library("caret") 
library("MASS")
library("MLmetrics")


Food_insecurity <-  read.csv("C:/Users/DELL/OneDrive/COURSERA COURSES/GOOGLE ANALYTIC/R/Google Data Analytic/Food_security/Food_security/Data/tidy_data/Food_insecurity_data.csv")

Food_insecurity %>% 
  group_by(status) %>% 
  summarise(st = n()) %>% 
  mutate(status_freq = paste0(round((100 * st/sum(st)), 0), '%'))

# New <- Food_insecurity
# 
# 
# barplot(prop.table(table(Food_insecurity$status)),
#         col = grey.colors(2),
#         ylim = c(0, 0.8),
#         main = "Class Distribution")
# 
# barplot(prop.table(table(over$status)),
#         col = grey.colors(2),
#         ylim = c(0, 0.8),
#         main = "Class Distribution after oversampling") 
Food_insecurity <- Food_insecurity %>% 
  dplyr::select(-c(9:21))



# Food_insecurity <- Food_insecurity %>%
#   dplyr::select(-c(1,3:4))

set.seed(1789)

# # # Change all the columns to factors 
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
 # Food_insecurity$status <-  relevel(Food_insecurity$status, ref = "Food secure")

table(Food_insecurity$status )

#IMBALANACE CLASS
# Splitting the data into train and test
index2 <- createDataPartition(Food_insecurity$status, p = .70, list = FALSE)
train2 <- Food_insecurity[index2, ] # 70% DATA
food_test <- Food_insecurity[-index2, ] # 30 % DATA #119


rftrain <- randomForest(status~., data = train2)
confusionMatrix(predict(rftrain, food_test), food_test$status, positive = "Food Insecure")
y_pred = predict(rftrain, newdata = food_test[-9])
table(y_pred, food_test$status)

# MCC
MCC_rftrain_full <- mcc(TP= 213,
                           FP=3,
                           TN=33,
                           FN=20)


# MCC_rftrain_reduced <- mcc(TP= 213,
#               FP=3,
#               TN=34,
#               FN=19)
# AUC
predictions <- as.numeric(predict(rftrain, food_test[-9], type = "response"))

pred <- ROCR::prediction(predictions, food_test$status)
Per <- ROCR::performance(pred, measure = "tpr", x.measure = "fpr")
 plot(Per, col= rainbow(10))
AUC <- ROCR::performance(pred, "auc")@y.values[[1]]
print(AUC)


#F SCORE
Recall <- 91.89
Precision <- 64.15
F_Score <- (2 * Precision * Recall / (Precision + Recall))/100
F_Score

# OVER SAMPLING
over <- ovun.sample(status~., data = train2, method = "over", N = 1014)$data
rfover <- randomForest(status~., data = over)
confusionMatrix(predict(rfover, food_test), food_test$status, positive = "Food Insecure")
y_pred = predict(rfover, newdata = food_test[-6])
table(y_pred, food_test$status)
importance(rfover)

# MCC
MCC_rftrain_full_over <- mcc(TP= 202,
                           FP=14,
                           TN=50,
                           FN=3)


MCC_rftrain_reduced_over <- mcc(TP= 201,
                           FP=15,
                           TN=53,
                           FN=0)
# AUC
contrasts(food_test$status)
predictions <- as.numeric(predict(rfover, food_test[-9], type = "response"))
pred <- ROCR::prediction(predictions, food_test$status)
Per <- ROCR::performance(pred, measure = "tpr", x.measure = "fpr")
plot(Per, col= rainbow(10))
AUC <- ROCR::performance(pred, "auc")@y.values[[1]]
print(AUC)


#F SCORE

Recall <- 78.12
Precision <- 94.34
F_Score <- (2 * Precision * Recall / (Precision + Recall))/100
F_Score

# MCC
MCC_rf <- mcc(TP= 201,
              FP=15,
              TN=53,
              FN=0)
MCC_rf 

mcc(TP= 100,
    FP=5000,
    TN=94900,
    FN=1)
# Plotting model
plot(rfover)

# Importance plot
importance(rfover)
# Variable importance plot
varImpPlot(rfover)
Feature <-  c("Income", "Household size", "Children", "Lg", "Age", "Employemnt status", "Gender", "Education")
MeanDecreaseGini <- c(167, 42, 16, 72,25, 10, 15 , 25)
Impt_feat <- tibble(Feature, MeanDecreaseGini)

dev.off() # to reset ggplot
ggplot(Impt_feat, aes(x = fct_reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini, fill = Feature))+
  geom_bar(stat='identity')+
  coord_flip()+
  scale_fill_manual(values=c("lightslategrey", "lightslategrey","lightslategrey", "lightslategrey","lightslategrey","lightslategrey", "lightblue", "lightslategrey"))+
  geom_text(aes(label = round((MeanDecreaseGini),2)))+
  labs(x= "Features",
       y = "MeanDecreaseGini",
       title = "Important Random Forest model features")+
theme_classic()+
  theme(legend.position="none")+
  theme(panel.spacing = unit(1, "lines"))

# UNDER SAMPLING 
table(train2$status)
under <- ovun.sample(status~., data = train2, method = "under", N = 248)$data
table(under$status)
#relevel the class
under$status <-  relevel(under$status, ref = "Food secure")
rfunder <- randomForest(status~., data = under)
confusionMatrix(predict(rfunder, food_test), food_test$status, positive = "Food Insecure")



# BOTH 
table(train2$status)
both <- ovun.sample(status~., data=train2, method = "both",
                    p = 0.5,
                    seed = 631,
                    N = )$data
table(both$status)
# relevel the reference class
both$status <-  relevel(both$status, ref = "Food secure")
rfboth <-randomForest(status~., data=both)
confusionMatrix(predict(rfboth, food_test), food_test$status, positive = "Food Insecure")

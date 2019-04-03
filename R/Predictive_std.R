#' Script to make prediction modelling based on processed STANDARDIZED data
#' Group 7

#' Clear working environment
rm(list = ls())

#' Setting directory
setwd("C:\\Users\\danie\\Documents\\Daniel Gil\\KULeuven\\Stage 2\\Term 2\\Advanced Analytics\\Assignments\\ChurnRate")
# Set your own directory:


#' Loading packages
library(ggplot2)
library(caret)
library(dplyr)
library(factoextra) # PCA
library(pROC)
library(glmnet) # Lasso and Ridge logistic

#' Reading data
training_std <- read.table("Data/train_sub_std_up.csv", sep = ",", dec = ".", 
                           header = T)
testing_std <- read.table("Data/test_sub_std.csv", sep = ",", dec = ".", 
                          header = T)
test_std <- read.table("Data/test_preprocessed_std.csv", sep = ",", dec = ".",
                       header = T)

#' Transforming categorical variables to factor
#' CHURN
training_std$CHURN <- as.factor(training_std$CHURN)
testing_std$CHURN <- as.factor(testing_std$CHURN)

#' PREPAID
training_std$PREPAID <- as.factor(training_std$PREPAID)
testing_std$PREPAID <- as.factor(testing_std$PREPAID)
test_std$PREPAID <- as.factor(test_std$PREPAID)

#' FINSTATE
training_std$FIN_STATE <- as.factor(training_std$FIN_STATE)
testing_std$FIN_STATE <- as.factor(testing_std$FIN_STATE)
test_std$FIN_STATE <- as.factor(test_std$FIN_STATE)


#----
#' *Logistic regression*
#' Model with only original variables
# names(training_std)
# names(training_std[,-c(35, 36, 38)])
logit_std <- glm(CHURN ~ ., data = training_std[,-c(35, 36, 38)], 
             family = binomial)
summary(logit_std)

#' Prediction on testing dataset
logit_std_pred <- predict(logit_std, testing_std, type = "response")  # predicted scores

# ROC 
logit_std_roc <- roc(testing_std$CHURN, logit_std_pred)
plot(logit_std_roc)
pROC::auc(logit_std_roc)

# Confusion matric
confusionMatrix(data = as.factor(as.numeric(logit_std_pred > 0.5)), 
                reference = testing$CHURN)

#' Lasso logistic
#' Transform from data.frame to matrix
x_std <- model.matrix(CHURN ~ ., data = training_std[,-c(35, 36, 38)])[,-1]
#' Transform response to numeric
y_std <- ifelse(training_std$CHURN == 0, 0, 1)

#' Cross-validation to find the best lambda
set.seed(12345)
lambda_std_l1 <- cv.glmnet(x_std, y_std, alpha = 1, family = "binomial")
logit_std_l1 <- glmnet(x_std, y_std, alpha = 1, family = "binomial", 
                   lambda = lambda_std_l1$lambda.min)
coef(logit_std_l1)

#' Prediction on testing dataset
testing_std <- testing_std[, names(training_std)]
# identical(names(training_std[,-c(35, 36, 38)]),
#           names(testing_std[,-c(35, 36, 38)])) # OK
x_testing_std <- model.matrix(CHURN ~ ., 
                          data = testing_std[,-c(35, 36, 38)])[,-1]
logit_std_l1_pred <- predict(logit_std_l1, x_testing_std, type = "response", 
                         s = lambda_std_l1$lambda.min)
# ROC 
logit_std_l1_roc <- roc(testing_std$CHURN, as.numeric(logit_std_l1_pred))
plot(logit_std_l1_roc)
pROC::auc(logit_std_l1_roc)

# Confusion matric
confusionMatrix(data = as.factor(as.numeric(logit_std_l1_pred > 0.5)), 
                reference = testing$CHURN)

#' Ridge logistic
#' Cross-validation to find the best lambda
set.seed(12345)
lambda_std_l2 <- cv.glmnet(x_std, y_std, alpha = 0, family = "binomial")
logit_std_l2 <- glmnet(x_std, y_std, alpha = 0, family = "binomial", 
                   lambda = lambda_std_l2$lambda.min)
coef(logit_std_l2)

#' Prediction on testing dataset
logit_std_l2_pred <- predict(logit_std_l2, x_testing_std, type = "response", 
                         s = lambda_std_l2$lambda.min)
# ROC 
logit_std_l2_roc <- roc(testing_std$CHURN, as.numeric(logit_std_l2_pred))
plot(logit_std_l2_roc)
pROC::auc(logit_std_l2_roc)

# Confusion matric
confusionMatrix(data = as.factor(as.numeric(logit_std_l2_pred > 0.5)), 
                reference = testing$CHURN)

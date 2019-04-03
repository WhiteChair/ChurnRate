#' Script to make prediction modelling based on processed data
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
training <- read.table("Data/train_sub_up.csv", sep = ",", dec = ".", header = T)
testing <- read.table("Data/test_sub.csv", sep = ",", dec = ".", header = T)
test <- read.table("Data/test_preprocessed.csv", sep = ",", dec = ".", header = T)

#' Transforming categorical variables to factor
#' CHURN
training$CHURN <- as.factor(training$CHURN)
testing$CHURN <- as.factor(testing$CHURN)

#' PREPAID
training$PREPAID <- as.factor(training$PREPAID)
testing$PREPAID <- as.factor(testing$PREPAID)
test$PREPAID <- as.factor(test$PREPAID)

#' FINSTATE
training$FIN_STATE <- as.factor(training$FIN_STATE)
testing$FIN_STATE <- as.factor(testing$FIN_STATE)
test$FIN_STATE <- as.factor(test$FIN_STATE)


#----
#' *Logistic regression*
#' Model with only original variables
# names(training)
# names(training[,-c(1, 2, 4, 38:42, 44:45)])
logit <- glm(CHURN ~ ., data = training[,-c(1, 2, 4, 38:42, 44:45)], 
             family = binomial)
summary(logit)

#' Prediction on testing dataset
logit_pred <- predict(logit, testing, type = "response")  # predicted scores

# ROC 
logit_roc <- roc(testing$CHURN, logit_pred)
plot(logit_roc)
pROC::auc(logit_roc)

# Confusion matric
confusionMatrix(data = as.factor(as.numeric(logit_pred > 0.5)), 
                reference = testing$CHURN)

####
#' Lasso logistic
#' Transform from data.frame to matrix
x <- model.matrix(CHURN ~ ., data = training[,-c(1, 2, 4, 38:42, 44:45)])[,-1]
#' Transform response to numeric
y <- ifelse(training$CHURN == 0, 0, 1)

#' Cross-validation to find the best lambda
set.seed(12345)
lambda_l1 <- cv.glmnet(x, y, alpha = 1, family = "binomial")
logit_l1 <- glmnet(x, y, alpha = 1, family = "binomial", 
                lambda = lambda_l1$lambda.min)
coef(logit_l1)

#' Prediction on testing dataset
testing <- testing[, names(training)]
# identical(names(training[,-c(1, 2, 4, 38:42, 44:45)]),
#           names(testing[,-c(1, 2, 4, 38:42, 44:45)])) # OK
x_testing <- model.matrix(CHURN ~ ., 
                          data = testing[,-c(1, 2, 4, 38:42, 44:45)])[,-1]
logit_l1_pred <- predict(logit_l1, x_testing, type = "response", 
                         s = lambda_l1$lambda.min)
# ROC 
logit_l1_roc <- roc(testing$CHURN, as.numeric(logit_l1_pred))
plot(logit_l1_roc)
pROC::auc(logit_l1_roc)

# Confusion matric
confusionMatrix(data = as.factor(as.numeric(logit_l1_pred > 0.5)), 
                reference = testing$CHURN)

#' Ridge logistic
#' Cross-validation to find the best lambda
set.seed(12345)
lambda_l2 <- cv.glmnet(x, y, alpha = 0, family = "binomial")
logit_l2 <- glmnet(x, y, alpha = 0, family = "binomial", 
                   lambda = lambda_l2$lambda.min)
coef(logit_l2)

#' Prediction on testing dataset
logit_l2_pred <- predict(logit_l2, x_testing, type = "response", 
                         s = lambda_l2$lambda.min)
# ROC 
logit_l2_roc <- roc(testing$CHURN, as.numeric(logit_l2_pred))
plot(logit_l2_roc)
pROC::auc(logit_l2_roc)

# Confusion matric
confusionMatrix(data = as.factor(as.numeric(logit_l2_pred > 0.5)), 
                reference = testing$CHURN)

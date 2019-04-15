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

#' Since there is only one category in both variables, there is no need to
#' transform to factor variables
#' #' PREPAID
#' table(training$PREPAID)
#' training$PREPAID <- as.factor(training$PREPAID)
#' testing$PREPAID <- as.factor(testing$PREPAID)
#' test$PREPAID <- as.factor(test$PREPAID)
#' 
#' #' FINSTATE
#' table(training$FIN_STATE, useNA = "always")
#' training$FIN_STATE <- as.factor(training$FIN_STATE)
#' testing$FIN_STATE <- as.factor(testing$FIN_STATE)
#' test$FIN_STATE <- as.factor(test$FIN_STATE)

# Store X and Y for later use. Only original variables
x = training[, -c(1, 2, 4, 38:42, 44:45)]
y = training$CHURN

#----
#' *Logistic regression*
#' Model with only original variables
# names(training)
# names(training[,-c(1, 2, 4, 38:42, 44:45)])
logit <- glm(CHURN ~ ., data = training[,-c(1, 2, 4, 38:42, 44:45)], 
             family = binomial)
# Tried with variables created by Donald. There's a warning because those new 
# variables a linear combination of the original variables. There is perfect
# colinearity
# https://stackoverflow.com/questions/26558631/predict-lm-in-a-loop-warning-prediction-from-a-rank-deficient-fit-may-be-mis
# logit <- glm(CHURN ~ ., data = training[,-c(1, 2, 4)], 
#              family = binomial)
summary(logit)

#' Prediction on testing dataset
logit_pred <- predict(logit, testing, type = "response")  # predicted scores

# ROC 
(logit_roc <- roc(testing$CHURN, logit_pred))
plot(logit_roc)
#plot(logit_roc, print.thres = "best")
pROC::auc(logit_roc)

# Confusion matric
confusionMatrix(data = as.factor(as.numeric(logit_pred > 0.5)), 
                reference = testing$CHURN, mode = "everything", positive = "1")

?confusionMatrix

####
#' Lasso logistic
#' Lasso and Ridge give the same results that in the script "predictive_Std"
#' because glmnet internally standardize variables
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
(logit_l1_roc <- roc(testing$CHURN, as.numeric(logit_l1_pred)))
plot(logit_l1_roc)
pROC::auc(logit_l1_roc)

# Confusion matric 
confusionMatrix(data = as.factor(as.numeric(logit_l1_pred > 0.5)), 
                reference = testing$CHURN, mode = "everything", positive = "1")

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
(logit_l2_roc <- roc(testing$CHURN, as.numeric(logit_l2_pred)))
plot(logit_l2_roc)
pROC::auc(logit_l2_roc)

# Confusion matric
confusionMatrix(data = as.factor(as.numeric(logit_l2_pred > 0.5)), 
                reference = testing$CHURN, mode = "everything", positive = "1")

#' In terms of AUC, lasso is the best model because it highest value (although
#' equal to the model without any regularization), but with less parameters. 
#' Ridge also provides good predictions. It has better accuracy than lasso


#----
#' *Decision trees*
#' Model with only original variables
library(tree)
?tree
tree1 <- tree(CHURN ~ ., data = training[,-c(1, 2, 4, 38:42, 44:45)], 
              split = "deviance")
summary(tree1)
plot(tree1)
text(tree1, pretty = 0)

tree1_pred <- predict(tree1, testing, type = "vector")[, 2]  # predicted scores

# ROC 
(tree1_roc <- roc(testing$CHURN, as.numeric(tree1_pred)))
#tree1_roc$thresholds
plot(tree1_roc)
pROC::auc(tree1_roc)

# Confusion matric
confusionMatrix(data = as.factor(as.numeric(tree1_pred > 0.5)), 
                reference = testing$CHURN, mode = "everything", positive = "1")


# Using gini
# Leads to better classification on training set but bad results on testing.
# It needs prunning
tree2 <- tree(CHURN ~ ., data = training[,-c(1, 2, 4, 38:42, 44:45)], 
              split = "gini") 
summary(tree2)
plot(tree2)
text(tree2, pretty = 0)

predict(train)

tree2_pred <- predict(tree2, testing, type = "vector")[, 2]  # predicted scores

# ROC 
(tree2_roc <- roc(testing$CHURN, as.numeric(tree2_pred)))
plot(tree2_roc)
#plot(tree2_roc, add = T)
pROC::auc(tree2_roc)

# Confusion matric
confusionMatrix(data = as.factor(as.numeric(tree2_pred > 0.5)), 
                reference = testing$CHURN, mode = "everything", positive = "1")

# Using cross-validation to prune the tree
# tree1 (deviance) using misclassification error
tree1_cv_mis <- cv.tree(tree1, FUN = prune.misclass, K = 10)
plot(tree1_cv_mis)
tree1_prune_mis <- prune.misclass(tree1, best = 7)
plot(tree1_prune_mis)
text(tree1_prune_mis, pretty = 0)

tree1_prune_mis_pred <- predict(tree1_prune_mis, testing, type = "vector")[, 2]  # predicted scores

# ROC 
(tree1_prune_mis_roc <- roc(testing$CHURN, as.numeric(tree1_prune_mis_pred)))
plot(tree1_prune_mis_roc)
#plot(tree1_prune_mis_roc, add = T)

# Confusion matric
confusionMatrix(data = as.factor(as.numeric(tree1_prune_mis_pred > 0.5)), 
                reference = testing$CHURN, mode = "everything", positive = "1")

# tree1 (deviance) using deviance
tree1_cv_dev <- cv.tree(tree1, FUN = prune.tree, K = 10)
plot(tree1_cv_dev)
tree1_prune_dev <- prune.misclass(tree1, best = 8)
plot(tree1_prune_dev)
text(tree1_prune_dev, pretty = 0)

tree1_prune_dev_pred <- predict(tree1_prune_dev, testing, type = "vector")[, 2]  # predicted scores

# ROC 
(tree1_prune_dev_roc <- roc(testing$CHURN, as.numeric(tree1_prune_dev_pred)))
plot(tree1_prune_dev_roc)
#plot(tree1_prune_dev_roc, add = T)

# Confusion matric
confusionMatrix(data = as.factor(as.numeric(tree1_prune_dev_pred > 0.5)), 
                reference = testing$CHURN, mode = "everything", positive = "1")

###
# tree2 (gini) using misclassification error
tree2_cv_mis <- cv.tree(tree2, FUN = prune.misclass, K = 10)
plot(tree2_cv_mis)
tree2_prune_mis <- prune.misclass(tree2, best = 100)
plot(tree2_prune_mis)
text(tree2_prune_mis, pretty = 0)

tree2_prune_mis_pred <- predict(tree2_prune_mis, testing, type = "vector")[, 2]  # predicted scores

# ROC 
(tree2_prune_mis_roc <- roc(testing$CHURN, as.numeric(tree2_prune_mis_pred)))
plot(tree2_prune_mis_roc, print.thres = "best")
#plot(tree2_prune_mis_roc, add = T)

# Confusion matric
confusionMatrix(data = as.factor(as.numeric(tree2_prune_mis_pred > 0.5)), 
                reference = testing$CHURN, mode = "everything", positive = "1")

# tree2 (gini) using deviance
tree2_cv_dev <- cv.tree(tree2, FUN = prune.tree, K = 10)
plot(tree2_cv_dev)
tree2_prune_dev <- prune.misclass(tree2, best = 10)
plot(tree2_prune_dev)
text(tree2_prune_dev, pretty = 0)

tree2_prune_dev_pred <- predict(tree2_prune_dev, testing, type = "vector")[, 2]  # predicted scores

# ROC 
(tree2_prune_dev_roc <- roc(testing$CHURN, as.numeric(tree2_prune_dev_pred)))
plot(tree2_prune_dev_roc)
#plot(tree2_prune_dev_roc, add = T)

# Confusion matric
confusionMatrix(data = as.factor(as.numeric(tree2_prune_dev_pred > 0.5)), 
                reference = testing$CHURN, mode = "everything", positive = "1")

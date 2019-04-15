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

#' Since there is only one category in both variables, there is no need to
#' transform to factor variables
#' #' PREPAID
#' training_std$PREPAID <- as.factor(training_std$PREPAID)
#' testing_std$PREPAID <- as.factor(testing_std$PREPAID)
#' test_std$PREPAID <- as.factor(test_std$PREPAID)
#' 
#' #' FINSTATE
#' training_std$FIN_STATE <- as.factor(training_std$FIN_STATE)
#' testing_std$FIN_STATE <- as.factor(testing_std$FIN_STATE)
#' test_std$FIN_STATE <- as.factor(test_std$FIN_STATE)

# Store X and Y for later use. Removing Start_Date, ID and Fin_State
x = training_std[, -c(35, 36, 38, 39)]
y = training_std$CHURN

#----
#' Feature importance
#' Boxplots
featurePlot(x = training_std[, -c(35, 36, 38, 39)], y = training_std$CHURN, 
            plot = "box",
            strip = strip.custom(par.strip.text = list(cex = .7)),
            scales = list(x = list(relation = "free"), 
                          y = list(relation = "free")))
#' Densities
featurePlot(x = training_std[, -c(35, 36, 38, 39)], y = training_std$CHURN, 
            plot = "density",
            strip = strip.custom(par.strip.text = list(cex = .7)),
            scales = list(x = list(relation = "free"), 
                          y = list(relation = "free")))

#' Feature selection using recursive feature elimination
# set.seed(12345)
# #options(warn = -1)
# subsets <- c(1:35)
# 
# # Using random forest
# ?rfeControl
# ctrl <- rfeControl(functions = rfFuncs,
#                    method = "repeatedcv",
#                    repeats = 5,
#                    verbose = FALSE)
# 
# (lmProfile <- rfe(x = training_std[, -c(35, 36, 38, 39)], y = training_std$CHURN,
#                  sizes = subsets,
#                  rfeControl = ctrl))
# lmProfile

#----
#' *Train Control*
#' 10-fold Crossvalidation
# fitControl <- trainControl(
#   method = 'cv',                   # k-fold cross validation
#   number = 5,                      # number of folds
#   savePredictions = 'final',       # saves predictions for optimal tuning parameter
#   classProbs = T,                  # should class probabilities be returned
#   summaryFunction = twoClassSummary  # results summary function
# ) 

#' 10-fold Crossvalidation repeated 3 times
fitControl <- trainControl(
  method = 'repeatedcv',                   # k-fold cross validation
  number = 10,                      # number of folds
  repeats = 3,
  savePredictions = 'final',       # saves predictions for optimal tuning parameter
  classProbs = T,                  # should class probabilities be returned
  summaryFunction = twoClassSummary  # results summary function
)

#' *Logistic regression*
#' Replacing levels of CHURN to be used in CARET functions
levels(training_std$CHURN) <- make.names(levels(factor(training_std$CHURN)))
levels(testing_std$CHURN) <- make.names(levels(factor(testing_std$CHURN)))

#' Model with only original variables
modelLookup("glm")
set.seed(12345)
logit <- train(CHURN ~ ., data = training_std[,-c(35, 36, 38)], 
               method = "glm", trControl = fitControl, metric = "ROC", 
               family = "binomial")
summary(logit)

#' Variable importance
logit_varimp <- varImp(logit)
plot(logit_varimp, main = "Variable Importance - Logit")

#' Prediction on testing dataset
logit_pred_prob <- predict(logit, testing_std, type = "prob")  # predicted scores
logit_pred <- predict(logit, testing_std)  # predicted class

# Confusion matrix
confusionMatrix(data = logit_pred, reference = testing_std$CHURN, 
                mode = "everything", positive = "X1")

# ROC 
(logit_roc <- roc(testing_std$CHURN, logit_pred_prob$X1))
plot(logit_roc)


#' *Lasso logistic*
#' http://www.sthda.com/english/articles/37-model-selection-essentials-in-r/153-penalized-regression-essentials-ridge-lasso-elastic-net/#ridge-regression
getModelInfo("glmnet")
modelLookup("glmnet")
set.seed(12345)
lasso <- train(CHURN ~ ., data = training_std[,-c(35, 36, 38)], 
               method = "glmnet", trControl = fitControl, metric = "ROC",
               family = "binomial",
               tuneGrid = expand.grid(.alpha = 1, 
                                      .lambda = seq(0, 2, by = 0.05)))
lasso
lasso$bestTune

# Coefficient of the final model. You need
# to specify the best lambda
coef(lasso$finalModel, lasso$bestTune$lambda)

#' Variable importance
lasso_varimp <- varImp(lasso)
plot(lasso_varimp, main = "Variable Importance - Lasso")

#' Prediction on testing dataset
lasso_pred_prob <- predict(lasso, testing_std, type = "prob")  # predicted scores
lasso_pred <- predict(lasso, testing_std)  # predicted class

# Confusion matrix
confusionMatrix(data = lasso_pred, reference = testing_std$CHURN, 
                mode = "everything", positive = "X1")

# ROC 
(lasso_roc <- roc(testing_std$CHURN, lasso_pred_prob$X1))
plot(lasso_roc)


#' *Ridge logistic*
set.seed(12345)
ridge <- train(CHURN ~ ., data = training_std[,-c(35, 36, 38)], 
               method = "glmnet", trControl = fitControl, metric = "ROC",
               family = "binomial",
               tuneGrid = expand.grid(.alpha = 0, 
                                      .lambda = seq(0, 2, by = 0.05)))
ridge

# Coefficient of the final model. You need
# to specify the best lambda
coef(ridge$finalModel, ridge$bestTune$lambda)

#' Variable importance
ridge_varimp <- varImp(ridge)
plot(ridge_varimp, main = "Variable Importance - Ridge")

#' Prediction on testing dataset
ridge_pred_prob <- predict(ridge, testing_std, type = "prob")  # predicted scores
ridge_pred <- predict(ridge, testing_std)  # predicted class

# Confusion matrix
confusionMatrix(data = ridge_pred, reference = testing_std$CHURN, 
                mode = "everything", positive = "X1")

# ROC 
(ridge_roc <- roc(testing_std$CHURN, ridge_pred_prob$X1))
plot(ridge_roc)

#' *Elastic net logistic*
set.seed(12345)
elastic <- train(CHURN ~ ., data = training_std[,-c(35, 36, 38)], 
               method = "glmnet", trControl = fitControl, metric = "ROC",
               family = "binomial",
               tuneLength = 10)
elastic

# Coefficient of the final model. You need
# to specify the best lambda
coef(elastic$finalModel, elastic$bestTune$lambda)

#' Variable importance
elastic_varimp <- varImp(elastic)
plot(elastic_varimp, main = "Variable Importance - Elastic")

#' Prediction on testing dataset
elastic_pred_prob <- predict(elastic, testing_std, type = "prob")  # predicted scores
elastic_pred <- predict(elastic, testing_std)  # predicted class

# Confusion matrix
confusionMatrix(data = elastic_pred, reference = testing_std$CHURN, 
                mode = "everything", positive = "X1")

# ROC 
(elastic_roc <- roc(testing_std$CHURN, elastic_pred_prob$X1))
plot(elastic_roc)


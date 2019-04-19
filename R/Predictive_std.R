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
library(doParallel)

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
head(cbind(logit_pred_prob,logit_pred), n = 60)
# ROC 
(logit_roc <- roc(testing_std$CHURN, logit_pred_prob$X1))
par(pty = "s")
plot(logit_roc, print.thres = "best")

# Confusion matrix
confusionMatrix(data = logit_pred, reference = testing_std$CHURN, 
                mode = "everything", positive = "X1")

# aux = as.factor(as.numeric(logit_pred_prob$X1 > 0.448))
#   levels(aux) <- make.names(levels(factor(aux)))
#   length(aux)
#   length(testing_std$CHURN)
# confusionMatrix(data = aux, 
#                 reference = testing_std$CHURN, mode = "everything", 
#                 positive = "X1")

# Best Threshold
confusionMatrix(data = as.factor(as.numeric(logit_pred_prob$X1 > 0.448)), 
                reference = as.factor(ifelse(testing_std$CHURN == "X0", 0, 1)),
                mode = "everything",
                positive = "1")

?confusionMatrix


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

#' *Decision tree with complexity as tuning parameter*
getModelInfo("rpart")
modelLookup("rpart")
set.seed(12345)
tree <- train(CHURN ~ ., data = training_std[,-c(35, 36, 38)], 
                 method = "rpart", trControl = fitControl, metric = "ROC",
              tuneLength = 50)
tree

# ROC value under different pruning values (complexity parameter) 
trellis.par.set(caretTheme())
plot(tree) 

# plot the model
plot(tree$finalModel, uniform = TRUE, main = "Decision Tree (cp)")
text(tree$finalModel, use.n. = TRUE, all = TRUE, cex = .8)

#' Variable importance
tree_varimp <- varImp(tree)
plot(tree_varimp, main = "Variable Importance - Decision Tree (cp)")

#' Prediction on testing dataset
tree_pred_prob <- predict(tree, testing_std, type = "prob")  # predicted scores
tree_pred <- predict(tree, testing_std)  # predicted class

# Confusion matrix
confusionMatrix(data = tree_pred, reference = testing_std$CHURN, 
                mode = "everything", positive = "X1")

# ROC 
(tree_roc <- roc(testing_std$CHURN, tree_pred_prob$X1))
plot(tree_roc)


#' *Decision tree with max depth as tuning parameter*
modelLookup("rpart2")
set.seed(12345)
tree2 <- train(CHURN ~ ., data = training_std[,-c(35, 36, 38)], 
              method = "rpart2", trControl = fitControl, metric = "ROC",
              tuneLength = 50)
tree2

# ROC value under different pruning values (complexity parameter) 
trellis.par.set(caretTheme())
plot(tree2) 

# plot the model
plot(tree2$finalModel, uniform = TRUE, main = "Decision Tree (max_depth)")
text(tree2$finalModel, use.n. = TRUE, all = TRUE, cex = .8)

#' Variable importance
tree2_varimp <- varImp(tree2)
plot(tree2_varimp, main = "Variable Importance - Decision Tree (max_depth)")

#' Prediction on testing dataset
tree2_pred_prob <- predict(tree2, testing_std, type = "prob")  # predicted scores
tree2_pred <- predict(tree2, testing_std)  # predicted class

# Confusion matrix
confusionMatrix(data = tree2_pred, reference = testing_std$CHURN, 
                mode = "everything", positive = "X1")

# ROC 
(tree2_roc <- roc(testing_std$CHURN, tree2_pred_prob$X1))
plot(tree2_roc)

#' *Random Forest*
modelLookup("rf")

cl <- makePSOCKcluster(10)
registerDoParallel(cl)
set.seed(12345)
rf <- train(CHURN ~ ., data = training_std[,-c(35, 36, 38)], 
               method = "rf", trControl = fitControl, metric = "ROC",
               tuneLength = 10)
stopCluster(cl)
rf

# ROC value under different tuning paramters
trellis.par.set(caretTheme())
plot(rf) 

#' Variable importance
rf_varimp <- varImp(rf)
plot(rf_varimp, main = "Variable Importance - Random Forest")

#' Prediction on testing dataset
rf_pred_prob <- predict(rf, testing_std, type = "prob")  # predicted scores
rf_pred <- predict(rf, testing_std)  # predicted class

# Confusion matrix
confusionMatrix(data = rf_pred, reference = testing_std$CHURN, 
                mode = "everything", positive = "X1")

# ROC 
(rf_roc <- roc(testing_std$CHURN, rf_pred_prob$X1))
plot(rf_roc)

# Prediction on test set
rf_pred_test <- predict(rf, test_std, type = "prob")  # predicted scores
rf_pred_test_out <- cbind(ID = test_std$ID, Churn = rf_pred_test$X1)

write.csv(rf_pred_test_out, "Output/rf.csv", row.names = F)

#' *Random Forest specifying number of variables in each tree*
modelLookup("rf")

#' Size of subset features
mtry <- sqrt(ncol(x))
tunegrid <- expand.grid(.mtry = mtry)

cl <- makePSOCKcluster(10)
registerDoParallel(cl)
set.seed(12345)
rf2 <- train(CHURN ~ ., data = training_std[,-c(35, 36, 38)], 
            method = "rf", trControl = fitControl, metric = "ROC",
            tuneGrid = tunegrid)
stopCluster(cl)
rf2

# ROC value under different tuning paramters
#trellis.par.set(caretTheme())
#plot(rf2) 

#' Variable importance
rf2_varimp <- varImp(rf2)
plot(rf2_varimp, main = "Variable Importance - Random Forest (size_var)")

#' Prediction on testing dataset
rf2_pred_prob <- predict(rf2, testing_std, type = "prob")  # predicted scores
rf2_pred <- predict(rf2, testing_std)  # predicted class

# Confusion matrix
confusionMatrix(data = rf2_pred, reference = testing_std$CHURN, 
                mode = "everything", positive = "X1")

# ROC 
(rf2_roc <- roc(testing_std$CHURN, rf2_pred_prob$X1))
plot(rf2_roc)

# Prediction on test set
rf2_pred_test <- predict(rf2, test_std, type = "prob")  # predicted scores
rf2_pred_test_out <- cbind(ID = test_std$ID, Churn = rf2_pred_test$X1)

write.csv(rf2_pred_test_out, "Output/rf2.csv", row.names = F)


#' *Adaboost*
modelLookup("adaboost")
cl <- makePSOCKcluster(10)
registerDoParallel(cl)
set.seed(12345)
ada <- train(CHURN ~ ., data = training_std[,-c(35, 36, 38)], 
             method = "adaboost", trControl = fitControl, metric = "ROC",
             tuneLength = 10)
stopCluster(cl)
ada

# ROC value under different tuning paramters
trellis.par.set(caretTheme())
plot(ada) 

#' Variable importance
ada_varimp <- varImp(ada)
plot(ada_varimp, main = "Variable Importance - Adaboost")

#' Prediction on testing dataset
ada_pred_prob <- predict(ada, testing_std, type = "prob")  # predicted scores
ada_pred <- predict(ada, testing_std)  # predicted class

# Confusion matrix
confusionMatrix(data = ada_pred, reference = testing_std$CHURN, 
                mode = "everything", positive = "X1")

# ROC 
(ada_roc <- roc(testing_std$CHURN, ada_pred_prob$X1))
plot(ada_roc)

# Prediction on test set
ada_pred_test <- predict(ada, test_std, type = "prob")  # predicted scores
ada_pred_test_out <- cbind(ID = test_std$ID, Churn = ada_pred_test$X1)

write.csv(ada_pred_test_out, "Output/adaboost.csv", row.names = F)

#' Saving environment
save.image(file = 'Training.RData')
#load('Training.RData')

#' *Extreme Gradient Boosting*
modelLookup("xgbDART")
cl <- makePSOCKcluster(10)
registerDoParallel(cl)
set.seed(12345)
xgb <- train(CHURN ~ ., data = training_std[,-c(35, 36, 38)], 
             method = "xgbDART", trControl = fitControl, metric = "ROC",
             tuneLength = 5, verbose = F)
stopCluster(cl)
xgb

# ROC value under different tuning paramters
trellis.par.set(caretTheme())
plot(xgb) 

#' Variable importance
xgb_varimp <- varImp(xgb)
plot(xgb_varimp, main = "Variable Importance - XGB")

#' Prediction on testing dataset
xgb_pred_prob <- predict(xgb, testing_std, type = "prob")  # predicted scores
xgb_pred <- predict(xgb, testing_std)  # predicted class

# Confusion matrix
confusionMatrix(data = xgb_pred, reference = testing_std$CHURN, 
                mode = "everything", positive = "X1")

# ROC 
(xgb_roc <- roc(testing_std$CHURN, xgb_pred_prob$X1))
plot(xgb_roc)

# Prediction on test set
xgb_pred_test <- predict(xgb, test_std, type = "prob")  # predicted scores
xgb_pred_test_out <- cbind(ID = test_std$ID, Churn = xgb_pred_test$X1)

write.csv(xgb_pred_test_out, "Output/XGB.csv", row.names = F)

#' *Neural Net*
modelLookup("nnet")
cl <- makePSOCKcluster(10)
registerDoParallel(cl)
set.seed(12345)
bnn <- train(CHURN ~ ., data = training_std[,-c(35, 36, 38)], 
             method = "nnet", trControl = fitControl, metric = "ROC",
             tuneLength = 50)
stopCluster(cl)
bnn

# ROC value under different tuning paramters
trellis.par.set(caretTheme())
plot(bnn) 

#' Variable importance
bnn_varimp <- varImp(bnn)
plot(bnn_varimp, main = "Variable Importance - Neural Net")

#' Prediction on testing dataset
bnn_pred_prob <- predict(bnn, testing_std, type = "prob")  # predicted scores
bnn_pred <- predict(bnn, testing_std)  # predicted class

# Confusion matrix
confusionMatrix(data = bnn_pred, reference = testing_std$CHURN, 
                mode = "everything", positive = "X1")

# ROC 
(bnn_roc <- roc(testing_std$CHURN, bnn_pred_prob$X1))
plot(bnn_roc)

# Prediction on test set
bnn_pred_test <- predict(bnn, test_std, type = "prob")  # predicted scores
bnn_pred_test_out <- cbind(ID = test_std$ID, Churn = bnn_pred_test$X1)

write.csv(bnn_pred_test_out, "Output/NN.csv", row.names = F)

#' Saving environment
#save.image(file = 'Training.RData')
load('Training.RData')

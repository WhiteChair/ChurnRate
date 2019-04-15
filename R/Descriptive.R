#' Script to make descriptive statistics on training data
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
library(skimr)

#' Reading data
training <- read.table("Data/train_sub.csv", sep = ",", dec = ".", header = T)
# training <- read.table("Data/train_sub_up.csv", sep = ",", dec = ".", header = T)
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
#' Descriptive statistics
#' Univariate
summary(training) # It is important to standardize continuous variables
# Using skimr package
View(skimmed <- skim_to_wide(training))
View(skimmed[, c(1:5, 9:11, 13, 15:16)])

#' Multivariate
#' Correlation matrix
pairs(training[,5:15], col = training$CHURN) # In some variables there's a clear difference

#' PCA on only original continuous variables
pca <- prcomp(as.matrix(training[,-c(1:4,38:42, 44:46)]), scale. = T)
summary(pca) # 20 PCs retain 90% of the variability
fviz_eig(pca)
fviz_pca_var(pca, col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)

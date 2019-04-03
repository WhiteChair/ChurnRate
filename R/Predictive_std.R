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

#' Reading data
training_std <- read.table("Data/train_sub_std_up.csv", sep = ",", dec = ".", 
                           header = T)
testing_std <- read.table("Data/test_sub_std.csv", sep = ",", dec = ".", 
                          header = T)
test_std <- read.table("Data/test_preprocessed_std.csv", sep = ",", dec = ".",
                       header = T)
#' Script to preprocess the data
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
library(unbalanced)
library(ROSE)

#' Reading data
train <- read.table("Data/telco_train.csv", sep = ",", dec = ".", header = T)
#head(train)
test <- read.table("Data/telco_test.csv", sep = ",", dec = ".", header = T)
#head(test)
#names(train)

#' Partition of training data in training and test datasets
#' 80% training, 20% testing
n_train <- round(nrow(train) * .8)
set.seed(12345)
id_train <- sample(1:nrow(train), n_train, replace = F)
train_sub <- train[id_train,] # sub stands for subset
test_sub <- train[-id_train,]

#' Counting number of NA's in each variable
apply(train_sub,2,function(x){sum(is.na(x))})
apply(test_sub,2,function(x){sum(is.na(x))})

#' Transforming categorical variables to factor
#' CHURN
train_sub$CHURN <- as.factor(train_sub$CHURN)
test_sub$CHURN <- as.factor(test_sub$CHURN)

#' PREPAID
train_sub$PREPAID <- as.factor(train_sub$PREPAID)
test_sub$PREPAID <- as.factor(test_sub$PREPAID)
test$PREPAID <- as.factor(test$PREPAID)

#' FINSTATE
train_sub$FIN_STATE <- as.factor(train_sub$FIN_STATE)
test_sub$FIN_STATE <- as.factor(test_sub$FIN_STATE)
test$FIN_STATE <- as.factor(test$FIN_STATE)

#----
#' New variables
#' Payment_Delay_level
#' Training dataset
train_sub$Payment_Delay_level <- ifelse(train_sub$COUNT_PAYMENT_DELAYS_1YEAR <= 4, 1,
                              ifelse(train_sub$COUNT_PAYMENT_DELAYS_1YEAR %in% 5:8, 2, 
                                ifelse(train_sub$COUNT_PAYMENT_DELAYS_1YEAR >= 8, 3, 0
                                       )))
#' Testing dataset
test_sub$Payment_Delay_level <- ifelse(test_sub$COUNT_PAYMENT_DELAYS_1YEAR <= 4, 1,
                                        ifelse(test_sub$COUNT_PAYMENT_DELAYS_1YEAR %in% 5:8, 2, 
                                               ifelse(test_sub$COUNT_PAYMENT_DELAYS_1YEAR >= 8, 3, 0
                                               )))

#' Test dataset
test$Payment_Delay_level <- ifelse(test$COUNT_PAYMENT_DELAYS_1YEAR <= 4, 1,
                                       ifelse(test$COUNT_PAYMENT_DELAYS_1YEAR %in% 5:8, 2, 
                                              ifelse(test$COUNT_PAYMENT_DELAYS_1YEAR >= 8, 3, 0
                                              )))

#--
#' CLV level
#' Training dataset
train_sub$CLV_level <- ifelse(train_sub$CLV < 11398, 1,
                          ifelse(train_sub$CLV %in% 11398:16891, 2,
                                 ifelse(train_sub$CLV >= 16892, 3 ,0 )))

#' Testing dataset
test_sub$CLV_level <- ifelse(test_sub$CLV < 11398, 1,
                              ifelse(test_sub$CLV %in% 11398:16891, 2,
                                     ifelse(test_sub$CLV >= 16892, 3 ,0 )))

#' Test dataset
test$CLV_level <- ifelse(test$CLV < 11398, 1,
                             ifelse(test$CLV %in% 11398:16891, 2,
                                    ifelse(test$CLV >= 16892, 3 ,0 )))

#--
#' Delta Complaints
#' Training dataset
train_sub$delta_m3_m6 <- train_sub$COMPLAINT_3MONTHS - train_sub$COMPLAINT_6MONTHS
train_sub$delta_m1_m3 <- train_sub$COMPLAINT_1MONTH - train_sub$COMPLAINT_3MONTHS

#' Testing dataset
test_sub$delta_m3_m6 <- test_sub$COMPLAINT_3MONTHS - test_sub$COMPLAINT_6MONTHS
test_sub$delta_m1_m3 <- test_sub$COMPLAINT_1MONTH - test_sub$COMPLAINT_3MONTHS

#' Test dataset
test$delta_m3_m6 <- test$COMPLAINT_3MONTHS - test$COMPLAINT_6MONTHS
test$delta_m1_m3 <- test$COMPLAINT_1MONTH - test$COMPLAINT_3MONTHS

#--
#' START_DATE
#' Compute the number of days since the customer joined the telco provider
#' Here I changed the reference date that Donald had according to the wording
#' of the problem
#' Training dataset
end_date <- as.Date("2013-10-31", format = "%Y-%m-%d")
train_sub$START_DATE <- as.Date(train_sub$START_DATE, format = "%Y-%m-%d")
train_sub$Time_Customer <- as.numeric(difftime(end_date, train_sub$START_DATE,
                                               units = "weeks"))
train_sub$Time_Customer_days <- as.numeric(end_date - train_sub$START_DATE)

#' Testing dataset
test_sub$START_DATE <- as.Date(test_sub$START_DATE, format = "%Y-%m-%d")
test_sub$Time_Customer <- as.numeric(difftime(end_date, test_sub$START_DATE,
                                               units = "weeks"))
test_sub$Time_Customer_days <- as.numeric(end_date - test_sub$START_DATE)

#' Test dataset
test$START_DATE <- as.Date(test$START_DATE, format = "%Y-%m-%d")
test$Time_Customer <- as.numeric(difftime(end_date, test$START_DATE,
                                              units = "weeks"))
test$Time_Customer_days <- as.numeric(end_date - test$START_DATE)

#--
#' Heavy weekend user
#' Training dataset
train_sub$OFFNET_weekend_talker <- ifelse(train_sub$MINUTES_INC_OFFNET_WKD_1MONTH 
                                          <= 16.542, 1,
                          ifelse(train_sub$MINUTES_INC_OFFNET_WKD_1MONTH %in% 
                                   16.542:37.268, 2,
                          ifelse(train_sub$MINUTES_INC_OFFNET_WKD_1MONTH > 37.268, 
                                 3, 0)))

#' Testing dataset
test_sub$OFFNET_weekend_talker <- ifelse(test_sub$MINUTES_INC_OFFNET_WKD_1MONTH 
                                          <= 16.542, 1,
                          ifelse(test_sub$MINUTES_INC_OFFNET_WKD_1MONTH %in% 
                                                   16.542:37.268, 2,
                          ifelse(test_sub$MINUTES_INC_OFFNET_WKD_1MONTH > 37.268, 
                                                        3, 0)))

#' Test dataset
test$OFFNET_weekend_talker <- ifelse(test$MINUTES_INC_OFFNET_WKD_1MONTH 
                                          <= 16.542, 1,
                              ifelse(test$MINUTES_INC_OFFNET_WKD_1MONTH %in% 
                                                   16.542:37.268, 2,
                              ifelse(test$MINUTES_INC_OFFNET_WKD_1MONTH > 37.268, 
                                                        3, 0)))

#--
#' Outcaller, mostly talked to people from other provider
#' Training dataset
train_sub$Outcaller <- ifelse(train_sub$AVG_MINUTES_OUT_OFFNET_1MONTH > 
                                train_sub$AVG_MINUTES_INC_OFFNET_1MONTH, 1,0)

#' Testing dataset
test_sub$Outcaller <- ifelse(test_sub$AVG_MINUTES_OUT_OFFNET_1MONTH > 
                               test_sub$AVG_MINUTES_INC_OFFNET_1MONTH, 1,0)

#' Test dataset
test$Outcaller <- ifelse(test$AVG_MINUTES_OUT_OFFNET_1MONTH > 
                           test$AVG_MINUTES_INC_OFFNET_1MONTH, 1,0)

#' Recoding NA's
#' It does not make sense to impute usage data to people who do not have data
#' subscription
#' COUNT_CONNECTIONS_3MONTH
#' Training dataset
train_sub$COUNT_CONNECTIONS_3MONTH <- 
  ifelse(is.na(train_sub$COUNT_CONNECTIONS_3MONTH), 
       0, train_sub$COUNT_CONNECTIONS_3MONTH)

#' Testing dataset
test_sub$COUNT_CONNECTIONS_3MONTH <- 
  ifelse(is.na(test_sub$COUNT_CONNECTIONS_3MONTH), 
         0, test_sub$COUNT_CONNECTIONS_3MONTH)

#' Test dataset
test$COUNT_CONNECTIONS_3MONTH <- 
  ifelse(is.na(test$COUNT_CONNECTIONS_3MONTH), 
         0, test$COUNT_CONNECTIONS_3MONTH)

#' AVG_DATA_1MONTH
#' Training dataset
train_sub$AVG_DATA_1MONTH <-  ifelse(is.na(train_sub$AVG_DATA_1MONTH), 
                                     0, train_sub$AVG_DATA_1MONTH)

#' Testing dataset
test_sub$AVG_DATA_1MONTH <- ifelse(is.na(test_sub$AVG_DATA_1MONTH), 
                                   0, test_sub$AVG_DATA_1MONTH)

#' Test dataset
test$AVG_DATA_1MONTH <- ifelse(is.na(test$AVG_DATA_1MONTH), 
                               0, test$AVG_DATA_1MONTH)

#' AVG_DATA_3MONTH
#' Training dataset
train_sub$AVG_DATA_3MONTH <- ifelse(is.na(train_sub$AVG_DATA_3MONTH), 
                                    0, train_sub$AVG_DATA_3MONTH)

#' Testing dataset
test_sub$AVG_DATA_3MONTH <- ifelse(is.na(test_sub$AVG_DATA_3MONTH), 
                                    0, test_sub$AVG_DATA_3MONTH)

#' Test dataset
test$AVG_DATA_3MONTH <- ifelse(is.na(test$AVG_DATA_3MONTH), 
                                   0, test$AVG_DATA_3MONTH)


#---- 
#' Continous variables standardized
# names(train_sub)
# names(test_sub)
# names(test)
# Compute mean and sd of each column
preproc <- preProcess(train_sub[,-c(1:5, 39:43, 45:46)], 
                            method = c("center", "scale"))

# Standardize variables
train_sub_std <- predict(preproc, train_sub[,-c(1:5, 39:43, 45:46)])
test_sub_std <- predict(preproc, test_sub[,-c(1:5, 39:43, 45:46)])
test_std <- predict(preproc, test[,-c(1:4, 38:42, 44:45)])
# identical(names(train_sub[,-c(1:5, 39:43, 45:46)]),
#           names(test_sub[,-c(1:5, 39:43, 45:46)])) # OK
# identical(names(train_sub[,-c(1:5, 39:43, 45:46)]),
#           names(test[,-c(1:4, 38:42, 44:45)])) # OK


# Paste original categorical variables
train_sub_std <- cbind(train_sub_std, train_sub[,1:5])
test_sub_std <- cbind(test_sub_std, test_sub[,1:5])
test_std <- cbind(test_std, test[,1:4])

# identical(names(train_sub[,-c(1:5, 39:43, 45:46)]),
#           names(test_sub[,-c(1:5, 39:43, 45:46)]))
# identical(names(train_sub[,-c(1:5, 39:43, 45:46)]),
#           names(test[,-c(1:4, 38:42, 44:45)]))
 
#' Upsampling (only in training dataset)
#' If upsampling is done randomly (instead of SMOTE), it does not matter if the
#' standardization is before or after (although not proven yet)
#' In this cases all preprocessing is done before upsampling
#' https://stats.stackexchange.com/questions/363312/normalization-standardization-should-one-do-this-before-oversampling-undersampl
#' Using unbalanced package
set.seed(12345)
data_over <- ubBalance(X = train_sub[,-2], Y = train_sub$CHURN, 
                       type = "ubOver", k = 0)
train_sub_up1 <- data.frame(data_over$X, CHURN = data_over$Y)
# setdiff(names(train_sub), names(train_sub_up1)) #OK

# Standardized dataset
set.seed(12345)
data_over_std <- ubBalance(X = train_sub_std[,-36], Y = train_sub_std$CHURN, 
                       type = "ubOver", k = 0)
train_sub_std_up1 <- data.frame(data_over_std$X, CHURN = data_over_std$Y)
setdiff(names(train_sub_std), names(train_sub_std_up1)) #OK

# Check upsampling is the same in both datasets
identical(rownames(train_sub_up1), rownames(train_sub_std_up1)) # OK
# View(skim_to_wide(train_sub_std_up1))
# head(overData)
# table(train_sub_up1$CHURN)
# table(train_sub_std_up1$CHURN)

#' Using caret package
# set.seed(12345)
# train_sub_up2 <- upSample(x = train_sub[,-2], y = train_sub$CHURN, list = F, 
#                          yname = "CHURN")

#' Exporting dataset
#' Training dataset
write.csv(train_sub, "Data/train_sub.csv", row.names = F)
write.csv(train_sub_std, "Data/train_sub_std.csv", row.names = F)
write.csv(train_sub_up1, "Data/train_sub_up.csv", row.names = F)
write.csv(train_sub_std_up1, "Data/train_sub_std_up.csv", row.names = F)

#' Testing dataset
write.csv(test_sub, "Data/test_sub.csv", row.names = F)
write.csv(test_sub_std, "Data/test_sub_std.csv", row.names = F)

#' Test dataset
write.csv(test, "Data/test_preprocessed.csv", row.names = F)
write.csv(test_std, "Data/test_preprocessed_std.csv", row.names = F)

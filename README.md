# Advanced Analytics in Business [D0S07a] Assignment
## ChurnRate

The goal is to construct a predictive model to predict whether a customer will churn or not. 

_We are scored on best AUC score._ 

## Uploading a result

we are group 7 
our pw: featspaw 

## Data
The data set describes a churn setting for telco customers of a Latin-American telco provider. The label ("CHURN") describes a binary target established as follows:

Customer still at the company by end of December 2013: CHURN = 0
Customer left the company in December 2013: CHURN = 1
The state of the features was extracted based on the situation at the end of October 2013. The prediction is hence set up to predict churn with a 1 month "action period". Customers which churned during the course of November were filtered out from the data set
The data set was then randomly split into a train and test split.

### Features:

ID: customer identifier
CHURN: binary target
START_DATE: data when customer joined the telco provided
PREPAID: whether this customer has ever utilized prepaid mobile cards in the past
FIN_STATE: financial state parameter
COUNT_PAYMENT_DELAYS_CURRENT: number of payment arrears at the moment of extracting the features (cumulative over multiple products)
COUNT_PAYMENT_DELAYS_1YEAR: number of payment arrears over the past year
DAYS_PAYMENT_DELAYS_CURRENT: number of days the customer paid late based on the moment of extracting the features; note that this is a cumulative count over multiple products); counts go negative based on number of days
DAYS_PAYMENT_DELAYS_1YEAR: number of days the customer paid late over the past year
COMPLAINT_1WEEK, COMPLAINT_2WEEKS, COMPLAINT_1MONTH, COMPLAINT_3MONTHS, COMPLAINT_6MONTHS: number of complaints received from the customer (past 1 week, 2 weeks, 1 month, 3 months, and 6 months
CLV: customer lifetime value as estimated by a different (simple) model at the time of extracting the features
COUNT_OFFNET_CALLS_1WEEK, COUNT_ONNET_CALLS_1WEEK: number of on (with customers of same telco) and offnet calls over the past week
COUNT_CONNECTIONS_3MONTH, AVG_DATA_1MONTH, AVG_DATA_3MONTH: number of data connections over the past 3 months, average bytes of data used per connection over the past 1 and 3 months; not applicable if customer doesn't have a data subscription
COUNT_SMS_INC_ONNET_6MONTH, COUNT_SMS_OUT_OFFNET_6MONTH, COUNT_SMS_INC_OFFNET_1MONTH, COUNT_SMS_INC_OFFNET_WKD_1MONTH, COUNT_SMS_INC_ONNET_1MONTH, COUNT_SMS_INC_ONNET_WKD_1MONTH, COUNT_SMS_OUT_OFFNET_1MONTH, COUNT_SMS_OUT_OFFNET_WKD_1MONTH, COUNT_SMS_OUT_ONNET_1MONTH, COUNT_SMS_OUT_ONNET_WKD_1MONTH: counts of SMS messages sent (OUT) and received (INC) over different time frames; again, ONNET indicates that the counterparty was also a subscriber of the same telco, OFFNET indicates they were not; WKD indicates that the aggregation was only done over the weekends (Saturday and Sunday)
AVG_MINUTES_INC_OFFNET_1MONTH, AVG_MINUTES_INC_ONNET_1MONTH, MINUTES_INC_OFFNET_WKD_1MONTH, MINUTES_INC_ONNET_WKD_1MONTH AVG_MINUTES_OUT_OFFNET_1MONTH, AVG_MINUTES_OUT_ONNET_1MONTH, MINUTES_OUT_OFFNET_WKD_1MONTH, MINUTES_OUT_ONNET_WKD_1MONTH: information concerning the durations of the calls; AVG indicates average per call, otherwise the value indicates a cumulative amount of minutes; INC and OUT indicate incoming and outgoing calls; ONNET indicates that the counterparty was also a subscriber of the same telco, OFFNET indicates they were not; WKD indicates that the aggregation or cumulation was only done over the weekends (Saturday and Sunday)

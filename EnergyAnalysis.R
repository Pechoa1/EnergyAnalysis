setwd("D:/Data/IN527Final")

library(purrr)
library(dplyr)
library(plyr)
library(readr)
library(psych)
library(tidyverse)
library(ggplot2)
library(reshape2)

# Import 11 data sets, join and organize

data_join <- list.files(path = "D:/Data/IN527Final", pattern = "*.csv", full.names = TRUE) %>%
  lapply(read_csv) %>%
  reduce(full_join, by = "Datetime")

# Remove duplicate columns 
df <- select(data_join, -c(10, 11, 12,13,14,15,16,17,18,19,20,21,22))

# import dayton's information since we will use that for the model later
dfday <- read_csv("DAYTON_hourly.csv")

# view data
describe(df)
str(df)

# melt data to a 3 variable matrix
dfmelt <- melt(df, id.vars = 'Datetime', variable.name = 'series')

# plot power consumption for all 11 locations
ggplot(dfmelt, aes(x = Datetime ,value)) + geom_line(aes(colour = series)) + 
  ggtitle("MW usage for all cities") + ylab("Megawatts") + xlab("Date")

  
# From the graph we can see PJME is the highest power consumer out of the picks.
  # Also, columns have variable time frames

df2018 <- dfmelt[dfmelt$Datetime >= "2018-01-01" & dfmelt$Datetime <= "2018-01-08", ]

# one week line graph for all areas
ggplot(df2018, aes(x = Datetime ,value)) + geom_line(aes(colour = series))

# Looking at PJME under 1 week
dfpjme <- select(df, c(Datetime, PJME_MW))
dfpjme <- dfpjme[dfpjme$Datetime >= "2018-01-01" & dfpjme$Datetime <= "2018-01-08", ]
dfpjme$day <- format(dfpjme$Datetime, format = "%d")
dfpjme$hour <- format(dfpjme$Datetime, format = "%H")

# graph pjme for hour by day
ggplot(dfpjme, aes(hour, PJME_MW)) + geom_line(aes(group = day,)) + 
  ggtitle("PJME one week") + facet_grid(. ~day) + theme(axis.text.x = element_blank())


  # Looking at the data for Dayton only
view(dfday)
ggplot(dfday, aes(Datetime,DAYTON_MW)) + geom_line()

  # format to make columns in year and month
dfday$Year <- format(dfday$Datetime, format = "%Y")
dfday$Month <- format(dfday$Datetime, format = "%m")
dfday$day <- format(dfday$Datetime, format = "%d")
dfday$hour <- format(dfday$Datetime, format = "%H")
dfday$Quarter <- lubridate::quarter(dfday$Datetime)


 # box plot of days grouped by month
ggplot(dfday, aes(day, DAYTON_MW)) + geom_boxplot(aes(group = Month,))  +
  facet_grid(. ~Month) + theme(axis.text.x = element_blank()) + ggtitle('Dayton day/month')
  # box plot of month grouped by year
ggplot(dfday, aes(Month, DAYTON_MW)) + geom_boxplot(aes(group = Year,))+
  facet_grid(. ~Year) + theme(axis.text.x = element_blank()) + ggtitle('Dayton month/year')

ggplot(dfday, aes(Datetime, DAYTON_MW)) + geom_line() + 
  geom_ma(ma_fun = SMA, n=3000, color = "BLUE") + geom_ma(ma_fun = SMA, n=10000, color = "RED")

# Insights
 # In months we see an increase in power usage from 06 to 08 
  # only difference in year is more power on 2007

# we now split data into a two year test df
dftrain <- subset(dfday, format(as.Date(Datetime), "%Y")<2016)
dftest <- subset(dfday, format(as.Date(Datetime), "%Y")>=2016)

# we will now use XGBOOST to predict our test
library(xgboost)

# split x into time, y into MW
x_train <-dftrain
x_train <- select(x_train, -c(DAYTON_MW,Datetime))
x_train <- mutate_all(x_train,function(x) as.numeric(as.character(x)))
y_train <- dftrain$DAYTON_MW

x_test <- dftest
x_test <- select(x_test, -c(DAYTON_MW,Datetime))
x_test <- mutate_all(x_test,function(x) as.numeric(as.character(x)))
y_test <- dftest$DAYTON_MW

# now we build the model
# Still learning what the next three chunks of code does.
xgb_trcontrol <- caret::trainControl(
  method = "cv", 
  number = 5,
  allowParallel = TRUE, 
  verboseIter = FALSE, 
  returnData = FALSE
)

xgb_grid <- base::expand.grid(
  list(
    nrounds = c(100, 200),
    max_depth = c(10, 15, 20), # maximum depth of a tree
    colsample_bytree = seq(0.5), # subsample ratio of columns when construction each tree
    eta = 0.1, # learning rate
    gamma = 0, # minimum loss reduction
    min_child_weight = 1,  # minimum sum of instance weight (hessian) needed ina child
    subsample = 1 # subsample ratio of the training instances
))


xgb_model <- caret::train(
  x_train, y_train,
  trControl = xgb_trcontrol,
  tuneGrid = xgb_grid,
  method = "xgbTree",
  nthread = 1
)


 # Check best values
xgb_model$bestTune

 # perform forecast

xgb_pred <- xgb_model %>% stats::predict(x_test)

# compare forecast with graph and correlation
plot(xgb_pred)
plot(dftest)
xgb_pred <- as.data.frame(xgb_pred)
xgb_pred$Datetime <- dftest$Datetime

ggplot(dftest, aes(x=Datetime, y=DAYTON_MW)) + geom_line() +
  geom_point(data = xgb_pred, aes(x=Datetime, y=xgb_pred), color = 'darkblue', alpha = 0.3)

# 0.51 correlation :(
cor(dftest$DAYTON_MW, xgb_pred$xgb_pred)

# 1 week prediction
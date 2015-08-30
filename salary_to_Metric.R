install.packages(c("e1071", "igraph", "mclust", "gbm", "randomForest", 
                   "ipred", "nnet", "kernlab"))

#loading libraries
library(caret)
library(gbm)
library(randomForest)
library(ipred)
library(nnet)
library(kernlab)
library(parallel)
library(doParallel)

#set working directory
setwd("C:/Users/Vladislav/Desktop/R_code")

#loading file
coreData = read.csv("sample.csv", na.strings = c("", "None"))

# Removing nzv variables
# nzv = nearZeroVar(coreData[sapply(coreData, is.numeric)], saveMetrics = TRUE)
# coreData = coreData[, nzv[, 'nzv'] == FALSE]


# Remove highly correlated variables
# corrM <- cor(coreData[sapply(coreData, is.numeric)])
# highCorr <- findCorrelation(corrM, cutoff = .90, verbose = FALSE)
# coreItog <- coreData[, -highCorr]
# delta1 <- dim(coreData)[2] - dim(coreItog)[2]        

#finding NAs
# sapply(coreData, function(x) sum(is.na(x)))

# preprocessing
row.names(coreData) <- coreData$id
coreData <- coreData[, -1]

for (i in c(1, 5:7, 9:19)) {
  coreData[, i] <- as.factor(coreData[, i])
}

# deleting variables with no variation - region_id, area_id, employer_id

coreData <- coreData[, -c(6, 7, 10)]

# 2nd model-----salary_to--------------------------------------------
# splitting data
set.seed(2301)
coreData[, 18] <- as.numeric(coreData[, 18])
core.sto <- coreData[is.na(coreData[, 18])== FALSE,]

core.sto <- core.sto[, -17] #deleting salary_from variable
core.sto <- na.omit(core.sto)

inTrain <- createDataPartition(core.sto$salary_to, p=0.8, list=FALSE)
train <- core.sto[inTrain,]
valid <- core.sto[-inTrain,]

# 2nd model evaluation---salary_to---------------------------------------------
# set seed
set.seed(2301)

# Setting clusters
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Creating metrics

mymetSummary <- function (data, lev = NULL, model = NULL) {
  
  pred <- ifelse(data$pred < 0, 0, data$pred)
  est <- data.frame(pred, obs = data$obs)
  metric_data <- c(NULL)
  for (i in 1:dim(est)[1]) {
    metric <- abs(min(est$obs[i], est$pred[i])/max(est$obs[i], est$pred[i])-1) 
    metric <- metric^2
    metric_data <- rbind(metric_data, metric)
  }
  
  out <- sqrt(sum(metric_data)/length(metric_data))
  names(out) <- "myMetric"
  out
}

# Setting control parameters 
mControl <- trainControl(method = "repeatedcv", number = 10, 
                         summaryFunction = mymetSummary, allowParallel=TRUE)


# (1) gbm
gbm.m <- train(train$salary_to ~ ., method = "gbm", verbose = FALSE,
               data = train, metric = "myMetric", maximize = FALSE, 
               trControl = mControl)

quality <- min(gbm.m$results$myMetric)

temp <- predict(gbm.m, valid)
pred <- ifelse(temp < 0, 0, temp)

est <- data.frame(pred, obs = valid$salary_to)

metric_data <- c(NULL)

for (i in 1:dim(est)[1]) {
  metric <- abs(min(est$obs[i], est$pred[i])/max(est$obs[i], est$pred[i])-1) 
  metric <- metric^2
  metric_data <- rbind(metric_data, metric)
}
metricItog <- sqrt(sum(metric_data)/length(metric_data))

qualityItog <- data.frame(Metric.train = quality, Metric = metricItog)

# (2) rf
rf.m <- train(train$salary_to ~ ., method = "rf", verbose = FALSE,
              data = train, metric = "myMetric", maximize = FALSE, 
              trControl = mControl)

quality <- min(rf.m$results$myMetric)

temp <- predict(rf.m, valid)
pred <- ifelse(temp < 0, 0, temp)

est <- data.frame(pred, obs = valid$salary_to)

for (i in 1:dim(est)[1]) {
  metric <- abs(min(est$obs[i], est$pred[i])/max(est$obs[i], est$pred[i])-1) 
  metric <- metric^2
  metric_data <- rbind(metric_data, metric)
}
metricItog <- sqrt(sum(metric_data)/length(metric_data))

qualityItog.new <- data.frame(Metric.train = quality, Metric = metricItog)

qualityItog <- rbind(qualityItog, qualityItog.new)

# (3) treebag
treebag.m <- train(train$salary_to ~ ., method = "treebag", 
                   verbose = FALSE, data = train, metric = "myMetric", 
                   maximize = FALSE, trControl = mControl)

quality <- min(treebag.m$results$myMetric)

temp <- predict(treebag.m, valid)
pred <- ifelse(temp < 0, 0, temp)

est <- data.frame(pred, obs = valid$salary_to)

for (i in 1:dim(est)[1]) {
  metric <- abs(min(est$obs[i], est$pred[i])/max(est$obs[i], est$pred[i])-1) 
  metric <- metric^2
  metric_data <- rbind(metric_data, metric)
}
metricItog <- sqrt(sum(metric_data)/length(metric_data))

qualityItog.new <- data.frame(Metric.train = quality, Metric = metricItog)

qualityItog <- rbind(qualityItog, qualityItog.new)

# (4) nnet
nnet.m <- train(train$salary_to ~ ., method = "nnet", 
                verbose = FALSE, data = train, metric = "myMetric", 
                maximize = FALSE, trControl = mControl)

quality <- min(nnet.m$results$myMetric)

temp <- predict(nnet.m, valid)
pred <- ifelse(temp < 0, 0, temp)

est <- data.frame(pred, obs = valid$salary_to)

for (i in 1:dim(est)[1]) {
  metric <- abs(min(est$obs[i], est$pred[i])/max(est$obs[i], est$pred[i])-1) 
  metric <- metric^2
  metric_data <- rbind(metric_data, metric)
}
metricItog <- sqrt(sum(metric_data)/length(metric_data))

qualityItog.new <- data.frame(Metric.train = quality, Metric = metricItog)

qualityItog <- rbind(qualityItog, qualityItog.new)

# (5) svmRadial
svmRadial.m <- train(train$salary_to ~ ., method = "svmRadial", 
                     verbose = FALSE, data = train, metric = "myMetric", 
                     maximize = FALSE, trControl = mControl)

quality <- min(svmRadial.m$results$myMetric)

temp <- predict(svmRadial.m, valid)
pred <- ifelse(temp < 0, 0, temp)

est <- data.frame(pred, obs = valid$salary_to)

for (i in 1:dim(est)[1]) {
  metric <- abs(min(est$obs[i], est$pred[i])/max(est$obs[i], est$pred[i])-1) 
  metric <- metric^2
  metric_data <- rbind(metric_data, metric)
}
metricItog <- sqrt(sum(metric_data)/length(metric_data))

qualityItog.new <- data.frame(Metric.train = quality, Metric = metricItog)

qualityItog <- rbind(qualityItog, qualityItog.new)

# Stop clusters
stopCluster(cl)

# Naming datasets
methods.m <- c("gbm", "rf", "treebag", "nnet", "svmRadial")

qualityItog$name <- methods.m


# table
qualityItog

itog.mod.name <- qualityItog$name[qualityItog$Metric == min(qualityItog$Metric)]

itog.mod.name

# itog predict for best model

# itog.m <- train(core.sto$salary_to ~ ., method = itog.mod.name, 
#                verbose = FALSE, data = core.sto)

# itog_salary_to <- data.frame(pred = predict(itog.m, core.sto), obs = core.sto$salary_to)


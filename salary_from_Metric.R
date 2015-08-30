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

# 1st model-----salary_from--------------------------------------------
# splitting data
set.seed(2301)
coreData[, 17] <- as.numeric(coreData[, 17])
core.sfrom <- coreData[is.na(coreData[, 17])== FALSE,]

core.sfrom <- core.sfrom[, -18] #deleting salary_to variable
core.sfrom <- na.omit(core.sfrom)

inTrain <- createDataPartition(core.sfrom$salary_from, p=0.8, list=FALSE)
train <- core.sfrom[inTrain,]
valid <- core.sfrom[-inTrain,]

# 1st model evaluation---salary_from---------------------------------------------
# set seed
set.seed(2301)

# Setting clusters
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Creating metrics

mymetSummary <- function (data, lev = NULL, model = NULL) {
est <- data.frame(pred = data$pred, obs = data$obs)
metric_data <- c(NULL)
for (i in 1:dim(est)[1]) {
  metric <- abs(min(c(est$obs[i], est$pred[i]))/max(c(est$obs[i], est$pred[i]))-1) 
  metric <- metric^2
  metric_data <- rbind(metric_data, metric)
}

out <- sum(metric_data)/length(metric_data)
names(out) <- "myMetric"
out
}

# Setting control parameters 
mControl <- trainControl(method = "repeatedcv", number = 10, 
                         summaryFunction = mymetSummary, allowParallel=TRUE)


# (1) gbm
gbm.m <- train(train$salary_from ~ ., method = "gbm", verbose = FALSE,
               data = train, metric = "myMetric", maximize = FALSE, 
               trControl = mControl)

quality <- min(gbm.m$results$myMetric)

est <- data.frame(pred = predict(gbm.m, valid), obs = valid$salary_from)

metric_data <- c(NULL)

for (i in 1:dim(est)[1]) {
  metric <- abs(min(c(est$obs[i], est$pred[i]))/max(c(est$obs[i], est$pred[i]))-1) 
  metric <- metric^2
  metric_data <- rbind(metric_data, metric)
}
metricItog <- sum(metric_data)/length(metric_data)

qualityItog <- data.frame(Metric.train = quality, Metric = metricItog)

# (2) rf
rf.m <- train(train$salary_from ~ ., method = "rf", verbose = FALSE,
              data = train, metric = "myMetric", maximize = FALSE, 
              trControl = mControl)

quality <- min(rf.m$results$myMetric)

est <- data.frame(pred = predict(rf.m, valid), obs = valid$salary_from)

for (i in 1:dim(est)[1]) {
  metric <- abs(min(c(est$obs[i], est$pred[i]))/max(c(est$obs[i], est$pred[i]))-1) 
  metric <- metric^2
  metric_data <- rbind(metric_data, metric)
}
metricItog <- sum(metric_data)/length(metric_data)

qualityItog.new <- data.frame(Metric.train = quality, Metric = metricItog)

qualityItog <- rbind(qualityItog, qualityItog.new)

# (3) treebag
treebag.m <- train(train$salary_from ~ ., method = "treebag", 
                   verbose = FALSE, data = train, metric = "myMetric", 
                   maximize = FALSE, trControl = mControl)

quality <- min(treebag.m$results$myMetric)

est <- data.frame(pred = predict(treebag.m, valid), obs = valid$salary_from)

for (i in 1:dim(est)[1]) {
  metric <- abs(min(c(est$obs[i], est$pred[i]))/max(c(est$obs[i], est$pred[i]))-1) 
  metric <- metric^2
  metric_data <- rbind(metric_data, metric)
}
metricItog <- sum(metric_data)/length(metric_data)

qualityItog.new <- data.frame(Metric.train = quality, Metric = metricItog)

qualityItog <- rbind(qualityItog, qualityItog.new)

# (4) nnet
nnet.m <- train(train$salary_from ~ ., method = "nnet", 
                verbose = FALSE, data = train, metric = "myMetric", 
                maximize = FALSE, trControl = mControl)

quality <- min(nnet.m$results$myMetric)

est <- data.frame(pred = predict(nnet.m, valid), obs = valid$salary_from)

for (i in 1:dim(est)[1]) {
  metric <- abs(min(c(est$obs[i], est$pred[i]))/max(c(est$obs[i], est$pred[i]))-1) 
  metric <- metric^2
  metric_data <- rbind(metric_data, metric)
}
metricItog <- sum(metric_data)/length(metric_data)

qualityItog.new <- data.frame(Metric.train = quality, Metric = metricItog)

qualityItog <- rbind(qualityItog, qualityItog.new)

# (5) svmRadial
svmRadial.m <- train(train$salary_from ~ ., method = "svmRadial", 
                     verbose = FALSE, data = train, metric = "myMetric", 
                     maximize = FALSE, trControl = mControl)

quality <- min(svmRadial.m$results$myMetric)

est <- data.frame(pred = predict(svmRadial.m, valid), obs = valid$salary_from)

for (i in 1:dim(est)[1]) {
  metric <- abs(min(c(est$obs[i], est$pred[i]))/max(c(est$obs[i], est$pred[i]))-1) 
  metric <- metric^2
  metric_data <- rbind(metric_data, metric)
}
metricItog <- sum(metric_data)/length(metric_data)

qualityItog.new <- data.frame(Metric.train = quality, Metric = metricItog)

qualityItog <- rbind(qualityItog, qualityItog.new)

# (6) rvmRadial
rvmRadial.m <- train(train$salary_from ~ ., method = "rvmRadial", 
                     verbose = FALSE, data = train, metric = "myMetric", 
                     maximize = FALSE, trControl = mControl)

quality <- min(rvmRadial.m$results$myMetric)

est <- data.frame(pred = predict(rvmRadial.m, valid), obs = valid$salary_from)

for (i in 1:dim(est)[1]) {
  metric <- abs(min(c(est$obs[i], est$pred[i]))/max(c(est$obs[i], est$pred[i]))-1) 
  metric <- metric^2
  metric_data <- rbind(metric_data, metric)
}
metricItog <- sum(metric_data)/length(metric_data)

qualityItog.new <- data.frame(Metric.train = quality, Metric = metricItog)

qualityItog <- rbind(qualityItog, qualityItog.new)

# Stop clusters
stopCluster(cl)

# Naming datasets
methods.m <- c("gbm", "rf", "treebag", "nnet", "svmRadial", "rvmRadial")

qualityItog$name <- methods.m


# table
qualityItog

itog.mod.name <- qualityItog$name[qualityItog$Metric == min(qualityItog$Metric)]

itog.mod.name

# itog predict for best model

# itog.m <- train(core.sfrom$salary_from ~ ., method = itog.mod.name, 
#                verbose = FALSE, data = core.sfrom)

# itog_salary_from <- data.frame(pred = predict(itog.m, core.sfrom), obs = core.sfrom$salary_from)


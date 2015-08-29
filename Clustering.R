install.packages(c('NbClust', 'stats', 'dplyr'))

#loading libraries
library(NbClust)
library(stats)
library(dplyr)
library(parallel)
library(doParallel)
library(Matrix)
library(skmeans)
library(cluster)

#set working directory
setwd("C:/Users/Vlad/Desktop/R_code")

#loading file
train = read.csv("features_strip.txt.txt")

# deleting ids
trainset <- train[, -1]

# clustering in 30 clusters
trainset.red <- Matrix(as.matrix(trainset), sparse = TRUE)

clust_sk <- skmeans(trainset.red, 30, method='pclust', control=list(verbose=TRUE))
summary(silhouette(clust_sk))

clusters_df <- data.frame(clust_id = clust_sk$cluster, vacancy_id = train[, 1])

head(clusters_df)



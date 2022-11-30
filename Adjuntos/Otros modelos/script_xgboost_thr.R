rm(list=ls())
  
#setwd("D:/data/SYNC/Dropbox/MASTER_2020/Semestre 01/FI-7244-Técnicas de  machine learning/Obligatorio/xgboost")

library(xgboost)
library(caretEnsemble)

source('../utils/utils_oblig.R')

set.seed(117)

script.name <- 'xgboost_thr'

#script.date <- date()
script.date <- "V1.1"

script.start <- Sys.time()

print('Start')

# leer el archivo dataset.csv de la carpeta

dataset <- read.csv('../data/dataset.csv')

# ver la estructura del dataset

# str(dataset)

# asignar el nombre del jugador como nombre de la fila

rownames(dataset) <- dataset$CustomerID

df <- na.omit(dataset[,-1])

df$ServiceArea <- NULL

print('** Distribucion a-priori de la variable a predecir')

print(prop.table(table(df$Churn)))

df.part <- train_dev_partition(df, p = 0.8)

df.thr_vec <- seq(0.1, 0.9, 0.05)

df.fn_summary <- function(data, lev = NULL, model = NULL) {
  fn_summaryUtilityThr(data, df.thr_vec)
}

df.metric <- 'utility'

df.form <- Churn ~ .

print('** XGBOOST')

df.xgboost.ctrl <- trainControl(method = 'cv',
                            number = 10,
                            verboseIter = TRUE,
                            classProbs = TRUE,
                            search = 'random',
                            summaryFunction = df.fn_summary)

df.xgboost.grid <-  expand.grid( nrounds = c(150,200,250,300,500), 
                                max_depth = c(1,2,3),
                                eta = c(0.2,0.4,0.6),
                                gamma = 0,
                                colsample_bytree = 0.8,
                                min_child_weight = 1,
                                subsample = 1
                               )


df.xgboost.grid <-  expand.grid( nrounds = c(1000),
                                 max_depth = c(2,3),
                                 eta = c(0.3),
                                 gamma = 0,
                                 colsample_bytree = c(0.6,0.8),
                                 min_child_weight = 1,
                                 subsample = c(0.75,1)
)
df.xgboost <- train(form = df.form, 
                data = df.part$train, 
                #method = 'xgLinear',
                method = 'xgbTree',
                #method = 'svmLinear',
                trControl = df.xgboost.ctrl,
                tuneGrid = df.xgboost.grid,
              #early_stopping_rounds = 20,
              #tunelength = 30,
                verbose = TRUE,
                metric = df.metric)

print(df.xgboost)

plot(df.xgboost)

df.xgboost.model <- df.xgboost$finalModel

df.xgboost.results <- fn_results(df.xgboost)

print('Umbral')

print(df.xgboost.results$prob_thr)

print('Utilidad en train')

print(df.xgboost.results$utility)

print('Utilidad en dev')

df.xgboost.dev.prob <- predict(df.xgboost, newdata = df.part$dev, type = 'prob')
df.xgboost.dev.pred <- fn_pred(df.xgboost.dev.prob, thr = df.xgboost.results$prob_thr)

df.xgboost.dev.utility <- fn_utility(df.xgboost.dev.pred, df.part$dev$Churn)

print(df.xgboost.dev.utility)

print('Matriz de confusion en dev')

df.xgboost.dev.cm <- conf_matrix(df.xgboost.dev.pred, df.part$dev$Churn)

print(df.xgboost.dev.cm)

print('** Generacion de la prediccion sobre test sample')

test_sample <- read.csv('../data/test_sample.csv')
rownames(test_sample) <- test_sample$CustomerID
test_sample$CustomerID <- NULL
test_sample$ServiceArea <- NULL

file_id <- paste0(c(script.name, script.date), collapse = ' ')

gen_prediction(df.xgboost, test_sample, prob_thr = df.xgboost.results$prob_thr, id = file_id)

print('Done')

script.done <- Sys.time()

print(script.done - script.start)


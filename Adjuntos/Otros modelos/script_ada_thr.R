rm(list=ls())
  
setwd("D:/data/SYNC/Dropbox/MASTER_2020/Semestre 01/FI-7244-Técnicas de  machine learning/Obligatorio/gbm")

library(gbm)

source('../utils/utils_oblig.R')

set.seed(117)

script.name <- 'gbm_thr'

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

df.part <- train_dev_partition(df, p = 0.9)

df.thr_vec <- seq(0.1, 0.9, 0.05)

df.fn_summary <- function(data, lev = NULL, model = NULL) {
  fn_summaryUtilityThr(data, df.thr_vec)
}

df.metric <- 'utility'

df.form <- Churn ~ .

print('** GBM')

df.gbm.ctrl <- trainControl(method = 'cv',
                            number = 5,
                            verboseIter = TRUE,
                            classProbs = TRUE,
                            search = 'random',
                            summaryFunction = df.fn_summary)

df.gbm.grid <-  expand.grid(maxdepth=30, 
                            nu=2,
                            iter=500)

df.gbm <- train(form = df.form, 
                data = df.part$train, 
                method = 'ada', 
                trControl = df.gbm.ctrl,
                tuneGrid = df.gbm.grid,
                verbose = TRUE,
                metric = df.metric)

print(df.gbm)

#plot(df.gbm)

df.gbm.model <- df.gbm$finalModel

df.gbm.results <- fn_results(df.gbm)

print('Umbral')

print(df.gbm.results$prob_thr)

print('Utilidad en train')

print(df.gbm.results$utility)

print('Utilidad en dev')

df.gbm.dev.prob <- predict(df.gbm, newdata = df.part$dev, type = 'prob')
df.gbm.dev.pred <- fn_pred(df.gbm.dev.prob, thr = df.gbm.results$prob_thr)

df.gbm.dev.utility <- fn_utility(df.gbm.dev.pred, df.part$dev$Churn)

print(df.gbm.dev.utility)

print('Matriz de confusion en dev')

df.gbm.dev.cm <- conf_matrix(df.gbm.dev.pred, df.part$dev$Churn)

print(df.gbm.dev.cm)

print('** Generacion de la prediccion sobre test sample')

test_sample <- read.csv('../data/test_sample.csv')
rownames(test_sample) <- test_sample$CustomerID
test_sample$CustomerID <- NULL
test_sample$ServiceArea <- NULL

file_id <- paste0(c(script.name, script.date), collapse = ' ')

gen_prediction(df.gbm, test_sample, prob_thr = df.gbm.results$prob_thr, id = file_id)

print('Done')

script.done <- Sys.time()

print(script.done - script.start)


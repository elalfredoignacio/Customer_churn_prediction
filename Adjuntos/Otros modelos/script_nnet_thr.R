rm(list=ls())
  
#setwd("D:/data/SYNC/Dropbox/MASTER_2020/Semestre 01/FI-7244-Técnicas de  machine learning/Obligatorio/nnet")

library(nnet)
library(caretEnsemble)

source('../utils/utils_oblig.R')

set.seed(117)

script.name <- 'nnet_thr'

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

print('** nnet')

df.nnet.ctrl <- trainControl(method = 'cv',
                            number = 3,
                            verboseIter = TRUE,
                            classProbs = TRUE,
                            search = 'grid',
                            summaryFunction = df.fn_summary)

df.nnet.grid <-  expand.grid( decay = seq(3, 4, .1),
                              size = c(18:20)
                              )
                                #max_depth = c(1,2,3),
             #                   eta = c(0.2,0.4,0.6),
        #                        gamma = 0,
         #                       colsample_bytree = 0.8,
          #                      min_child_weight = 1,
           #                     subsample = 1
            #                   )

df.nnet <- train(form = df.form, 
                data = df.part$train, 
                method = 'nnet',
                trControl = df.nnet.ctrl,
                tuneGrid = df.nnet.grid,
                MaxNWts = 20000,
                maxit = 1000,
                verbose = TRUE,
                metric = df.metric)

print(df.nnet)

plot(df.nnet)

df.nnet.model <- df.nnet$finalModel

df.nnet.results <- fn_results(df.nnet)

print('Umbral')

print(df.nnet.results$prob_thr)

print('Utilidad en train')

print(df.nnet.results$utility)

print('Utilidad en dev')

df.nnet.dev.prob <- predict(df.nnet, newdata = df.part$dev, type = 'prob')
df.nnet.dev.pred <- fn_pred(df.nnet.dev.prob, thr = df.nnet.results$prob_thr)

df.nnet.dev.utility <- fn_utility(df.nnet.dev.pred, df.part$dev$Churn)

print(df.nnet.dev.utility)

print('Matriz de confusion en dev')

df.nnet.dev.cm <- conf_matrix(df.nnet.dev.pred, df.part$dev$Churn)

print(df.nnet.dev.cm)

print('** Generacion de la prediccion sobre test sample')

test_sample <- read.csv('../data/test_sample.csv')
rownames(test_sample) <- test_sample$CustomerID
test_sample$CustomerID <- NULL
test_sample$ServiceArea <- NULL

file_id <- paste0(c(script.name, script.date), collapse = ' ')

gen_prediction(df.nnet, test_sample, prob_thr = df.nnet.results$prob_thr, id = file_id)

print('Done')

script.done <- Sys.time()

print(script.done - script.start)


#rm(list=ls())

#setwd("D:/data/SYNC/Dropbox/MASTER_2020/Semestre 01/FI-7244-Técnicas de  machine learning/Obligatorio/xgboost")

#library(xgboost)
library(caretEnsemble)

source('../utils/utils_oblig.R')

set.seed(117)

script.name <- 'stack'

script.date <-  format(Sys.time(), "%Y%m%d_%H%M")

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

print('** STACK')


algorithmList <- c("gbm", "rf" ) 
control <- trainControl(method = "cv", 
                        number = 3, 
                        #repeats = 3, 
                        savePredictions = "final" , 
                        df.metric,
                        classProbs = TRUE,
                        search = 'random',
                        summaryFunction = df.fn_summary
                        )
models <- caretList(Churn ~ ., 
                          data = df.part$train, 
                          trControl = control,
                          metric = "utility", 
                          methodList = algorithmList
                          )

results <- resamples(models)

summary(results)

df.gbm <- caretStack(models, method="gbm", metric = "utility", trControl = control)

print(df.gbm)

plot(df.gbm)

df.gbm.model <- df.gbm$finalModel


df.gbm.results <- fn_results(df.gbm)


print('Umbral')
print(df.gbm.results$prob_thr)
v_umbral <- df.gbm.results$prob_thr

print('Utilidad en train')

print(df.gbm.results$utility)

v_util_train <- df.gbm.results$utility

print('Utilidad en dev')

df.gbm.dev.prob <- predict(df.gbm, newdata = df.part$dev, type = 'prob')
df.gbm.dev.pred <- fn_pred(df.gbm.dev.prob, thr = df.gbm.results$prob_thr)
df.gbm.dev.utility <- fn_utility(df.gbm.dev.pred, df.part$dev$Churn)

print(df.gbm.dev.utility)

v_util_dev <- df.gbm.dev.utility

print('Matriz de confusion en dev')
df.gbm.dev.cm <- conf_matrix(df.gbm.dev.pred, df.part$dev$Churn)
print(df.gbm.dev.cm)

#print('** Generacion de la prediccion sobre test sample')

test_sample <- read.csv('../data/test_sample.csv')
rownames(test_sample) <- test_sample$CustomerID
test_sample$CustomerID <- NULL
test_sample$ServiceArea <- NULL

file_id <- paste0(c(script.name, v_contador, script.date,v_cv_qty,v_trees,v_shrinkage,v_minobsinnode,v_interaction_depth), collapse = '_')

gen_prediction(df.gbm, test_sample, prob_thr = df.gbm.results$prob_thr, id = file_id)
rm(list=ls())
  
#setwd("D:/data/SYNC/Dropbox/MASTER_2020/Semestre 01/FI-7244-Técnicas de  machine learning/Obligatorio/svm")
#setwd("Z:/dropbox/MASTER_2020/Semestre 01/FI-7244-Técnicas de  machine learning/Obligatorio/others")

source('../utils/utils_oblig.R')

set.seed(117)

script.name <- 'svm_thr'

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

df.part <- train_dev_partition(df, p = 0.01)

df.thr_vec <- seq(0.1, 0.9, 0.05)

df.fn_summary <- function(data, lev = NULL, model = NULL) {
  fn_summaryUtilityThr(data, df.thr_vec)
}

df.metric <- 'utility'

df.form <- Churn ~ .

print('** SVM')

df.svm.ctrl <- trainControl(method = 'cv',
                            number = 3,
                            verboseIter = TRUE,
                            classProbs = TRUE,
                            search = 'random',
                            summaryFunction = df.fn_summary)

df.svm.grid <-  expand.grid(c("center","scale"))

df.svm <- train(form = df.form, 
                data = df.part$train, 
                method = 'svmLinear', 
                trControl = df.svm.ctrl,
                #tuneGrid = df.svm.grid,
                preProcess = c("center","scale"),
                verbose = TRUE,
                metric = df.metric)

print(df.svm)

#plot(df.svm)

df.svm.model <- df.svm$finalModel

df.svm.results <- fn_results(df.svm)

print('Umbral')

print(df.svm.results$prob_thr)

print('Utilidad en train')

print(df.svm.results$utility)

print('Utilidad en dev')

df.svm.dev.prob <- predict(df.svm, newdata = df.part$dev, type = 'prob')
df.svm.dev.pred <- fn_pred(df.svm.dev.prob, thr = df.svm.results$prob_thr)

df.svm.dev.utility <- fn_utility(df.svm.dev.pred, df.part$dev$Churn)

print(df.svm.dev.utility)

print('Matriz de confusion en dev')

df.svm.dev.cm <- conf_matrix(df.svm.dev.pred, df.part$dev$Churn)

print(df.svm.dev.cm)

print('** Generacion de la prediccion sobre test sample')

test_sample <- read.csv('../data/test_sample.csv')
rownames(test_sample) <- test_sample$CustomerID
test_sample$CustomerID <- NULL
test_sample$ServiceArea <- NULL

file_id <- paste0(c(script.name, script.date), collapse = ' ')

gen_prediction(df.svm, test_sample, prob_thr = df.svm.results$prob_thr, id = file_id)

print('Done')

script.done <- Sys.time()

print(script.done - script.start)
  

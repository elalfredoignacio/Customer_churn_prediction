library(ranger)
library(randomForest)

source('../utils/utils_oblig.R')

set.seed(117)

script.name <- 'ranger_thr'

script.date <- 'random'

script.start <- Sys.time()

print('Start')

# leer el archivo dataset.csv de la carpeta

dataset <- read.csv('../data/dataset.csv')

# ver la estructura del dataset

# str(dataset)

# asignar el nombre del jugador como nombre de la fila

rownames(dataset) <- dataset$CustomerID

df <- dataset[,-1]

df$ServiceArea <- NULL

df <- na.roughfix(df)

print('** Distribucion a-priori de la variable a predecir')

print(prop.table(table(df$Churn)))

df.part <- train_dev_partition(df, p = 0.8)

df.thr_vec <- seq(0.2, 0.4, 0.025)

df.fn_summary <- function(data, lev = NULL, model = NULL) {
  fn_summaryUtilityThr(data, df.thr_vec)
}

df.metric <- 'utility'

df.form <- Churn ~ .

print('** GBM')

df.ranger.ctrl <- trainControl(method = 'cv',
                            number = 5,
                            verboseIter = TRUE,
                            classProbs = TRUE,
                            search = 'random',
                            summaryFunction = df.fn_summary)

#df.ranger.grid <-  expand.grid(mtry = c(7, 8, 9, 10), 
                          #     min.node.size = c(1, 4, 6, 10),
                           #    splitrule = 'gini')
  
df.ranger <- train(form = df.form, 
                data = df.part$train, 
                method = 'ranger', 
                trControl = df.ranger.ctrl,
                #tuneGrid = df.ranger.grid,
                verbose = FALSE,
                tuneLength = 10,
                metric = df.metric)

print(df.ranger)
df.ranger.results <- fn_results(df.ranger)

#######################
# variables para graficar la utilidad de CV en función de un solo parámetro

valores_mtry <- df.ranger$results$mtry
valores_mnodesz <- df.ranger$results$min.node.size
valores_utilidad_cv <- df.ranger$results$utility
valores_umbral <- df.ranger$results$prob_thr

ranks <- order(valores_mtry)
plot(valores_mtry[ranks], valores_utilidad_cv[ranks], type = "l", 
     main = "utilidad CV ranger", xlab = "mtry", 
     ylab = "utilidad CV", col = "red" )
abline(v=df.ranger$bestTune$mtry)

ranks_mns <- order(valores_mnodesz)
plot(valores_mnodesz[ranks_mns], valores_utilidad_cv[ranks_mns], type = "l", 
     main = "utilidad CV ranger", xlab = "min node size", 
     ylab = "utilidad CV", col = "blue" )
abline(v=df.ranger$bestTune$min.node.size)

ranks_th <- order(valores_umbral)
plot(valores_umbral[ranks_th], valores_utilidad_cv[ranks_th], type = "l", 
     main = "utilidad CV ranger", xlab = "umbral", 
     ylab = "utilidad CV", col = "green" )
abline=(v=df.ranger.results$prob_thr)


plot(valores_mtry[ranks], valores_umbral[ranks], type = "l", 
     main = "Umbral vs mtry ranger", xlab = "mtry", 
     ylab = "Umbral", col = "red" )
abline(v=df.ranger$bestTune$mtry)

plot(valores_mnodesz[ranks_mns], valores_umbral[ranks_mns], type = "l", 
     main = "Umbral vs min node size ranger", xlab = "min node size", 
     ylab = "Umbral", col = "red" )
abline(v=df.ranger$bestTune$min.node.size)

####################

plot(df.ranger)
ggplot(df.ranger)
df.ranger.model <- df.ranger$finalModel

print('Umbral')

print(df.ranger.results$prob_thr)

print('Utilidad en CV')

print(df.ranger.results$utility)

print('Utilidad en dev')

df.ranger.dev.prob <- predict(df.ranger, newdata = df.part$dev, type = 'prob')
df.ranger.dev.pred <- fn_pred(df.ranger.dev.prob, thr = df.ranger.results$prob_thr)

df.ranger.dev.utility <- fn_utility(df.ranger.dev.pred, df.part$dev$Churn)

print(df.ranger.dev.utility)

print('Utilidad en train')

df.ranger.train.prob <- predict(df.ranger, newdata = df.part$train, type = 'prob')
df.ranger.train.pred <- fn_pred(df.ranger.train.prob, thr = df.ranger.results$prob_thr)

df.ranger.train.utility <- fn_utility(df.ranger.train.pred, df.part$train$Churn)

print(df.ranger.train.utility)

#################
# ploteo de las utilidades de CV y train vs el umbral

df.ranger.train.prob <- predict(df.ranger, newdata = df.part$train, type = 'prob')

df.ranger.dev.utility_vec <- fn_utility_thr(y_prob = df.ranger.dev.prob, 
                                           y = df.part$dev$Churn, 
                                           thr_vec = df.thr_vec)


df.ranger.train.utility_vec <- fn_utility_thr(y_prob = df.ranger.train.prob, 
                                             y = df.part$train$Churn, 
                                             thr_vec = df.thr_vec)

print('Utilidad por umbral')
print(df.ranger.dev.utility_vec)
print(df.ranger.train.utility_vec)

#par(mfrow=c(1,1))
plot_thr_utility(df.ranger.dev.utility_vec, df.thr_vec, 'ranger - utilidad dev')

abline(v = df.ranger.results$prob_thr)
par(new=TRUE)

plot_thr_utility_train(df.ranger.train.utility_vec, df.thr_vec, 'ranger - utilidad train')
abline(v = df.ranger.results$prob_thr, col="blue")
#########################################

print('Matriz de confusion en dev')

df.ranger.dev.cm <- conf_matrix(df.ranger.dev.pred, df.part$dev$Churn)

print(df.ranger.dev.cm)

print('** Generacion de la prediccion sobre test sample')

test_sample <- read.csv('../data/test_sample.csv')
rownames(test_sample) <- test_sample$CustomerID
test_sample$CustomerID <- NULL
test_sample$ServiceArea <- NULL
test_sample <- na.roughfix(test_sample)

file_id <- paste0(c(script.name, script.date), collapse = ' ')

gen_prediction(df.ranger, test_sample, prob_thr = df.ranger.results$prob_thr, id = file_id)

print('Done')

script.done <- Sys.time()

print(script.done - script.start)


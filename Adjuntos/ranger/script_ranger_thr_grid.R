library(ranger)
library(randomForest)

source('../utils/utils_oblig.R')

set.seed(117)

script.name <- 'ranger_thr'

script.date <- 'grid'

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

df.thr_vec <- seq(0.2, 0.4, 0.001)

df.fn_summary <- function(data, lev = NULL, model = NULL) {
  fn_summaryUtilityThr(data, df.thr_vec)
}

# defino la función para poder calcular la utilidad en función del umbral

fn_utility_thr <- function(y_prob, y, thr_vec) {
  utility <- c()
  for(thr in thr_vec){
    pred_thr <- fn_pred(y_prob, thr)
    uty <- fn_utility(yhat = pred_thr, y)
    utility <- c(utility, uty)
  }
  c(utility = utility)
}

df.metric <- 'utility'

df.form <- Churn ~ .

print('** RANGER')

df.ranger.ctrl <- trainControl(method = 'cv',
                            number = 5,
                            verboseIter = TRUE,
                            classProbs = TRUE,
                            search = 'grid',
                            summaryFunction = df.fn_summary)

df.ranger.grid <-  expand.grid(mtry = seq(3, 7, 1), 
                             min.node.size = seq(1, 15, 3),
                             splitrule = 'gini')
  
df.ranger <- train(form = df.form, 
                data = df.part$train, 
                method = 'ranger', 
                trControl = df.ranger.ctrl,
                tuneGrid = df.ranger.grid,
                verbose = FALSE,
                #tuneLength = 10,
                metric = df.metric)

print(df.ranger)
df.ranger.results <- fn_results(df.ranger)



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

#abline(v = df.ranger.results$prob_thr)
#par(new=TRUE)

plot_thr_utility_train(df.ranger.train.utility_vec, df.thr_vec, 'ranger - utilidad train')
#abline(v = df.ranger.results$prob_thr, col="blue")
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


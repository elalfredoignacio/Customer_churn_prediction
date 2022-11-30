library(ranger)
library(randomForest)

source('../utils/utils_oblig.R')

set.seed(117)

script.name <- 'ranger_thr_ROC'

script.date <- 'v1'

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

df.fn_summary <- twoClassSummary

df.metric <- 'ROC'

df.form <- Churn ~ .

print('** RANGER')

df.ranger.ctrl <- trainControl(method = 'cv',
                            number = 2,
                            verboseIter = TRUE,
                            classProbs = TRUE,
                            search = 'grid',
                            summaryFunction = df.fn_summary)

df.ranger.grid <-  expand.grid(mtry = c(7, 8, 9, 10), 
                               min.node.size = c(1, 4, 6, 10),
                               splitrule = 'gini')
  
df.ranger <- train(form = df.form, 
                data = df.part$train, 
                method = 'ranger', 
                trControl = df.ranger.ctrl,
                tuneGrid = df.ranger.grid,
                verbose = FALSE,
                metric = df.metric)



print(df.ranger)

plot(df.ranger)
ggplot(df.ranger)
df.ranger.model <- df.ranger$finalModel

#df.ranger.results <- fn_results(df.ranger)

print('Umbral')

print(df.ranger.results$prob_thr)

print('Utilidad en train')

print(df.ranger.results$ROC)

print('Utilidad en dev')

df.ranger.dev.prob <- predict(df.ranger, newdata = df.part$dev, type = 'prob')
df.ranger.dev.pred <- fn_pred(df.ranger.dev.prob, thr = df.ranger.results$prob_thr)

df.ranger.dev.utility <- fn_utility(df.ranger.dev.pred, df.part$dev$Churn)

print(df.ranger.dev.utility)

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


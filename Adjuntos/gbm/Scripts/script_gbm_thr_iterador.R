rm(list=ls())
  
setwd("D:/data/SYNC/Dropbox/MASTER_2020/Semestre 01/FI-7244-Técnicas de  machine learning/Obligatorio/gbm")

library(gbm)

source('../utils/utils_oblig.R')

script.name <- 'gbm_thr_iter'

script.date <-  format(Sys.time(), "%Y%m%d_%H%M")
script.start <- Sys.time()


# leer el archivo dataset.csv de la carpeta
dataset <- read.csv('../data/dataset.csv')
rownames(dataset) <- dataset$CustomerID


#------------------------------------------------------------------------------------------------------------
# DEFINICION DE PARÁMETROS DE GBM
#------------------------------------------------------------------------------------------------------------
v_cv_qty <- 5

v_trees = c(730:770)
v_shrinkage = c(0.08)
v_minobsinnode = c(10)
v_interaction_depth = c(2)
v_contador = 1

v_total <- length(v_trees) * length(v_shrinkage) * length(v_minobsinnode) * length(v_interaction_depth)

v_resultados <- paste0('resultados_',script.date,'.csv',collapse='_')


df.resultados <- data.frame(Modelo=double(),
                              Umbral=double(),
                              prob_cv = double(),
                              prob_train = double(),
                              prob_test = double(),
                              time_exec = double(),
                              cv = double(),
                              trees = double(),
                              shrinkage = double(),
                              minobsinnode = double(),
                              interaction_depth = double()
                             ) 

#-------------------------------------------------------------
# Loop para hacer las llamadas en cada caso
#-------------------------------------------------------------
for (i in v_trees) {
  for (j in v_shrinkage) {
    for (k in v_minobsinnode) {
      for (l in v_interaction_depth) {
        
        set.seed(117)
        
        #Elimina los NA
        df <- na.omit(dataset[,-1])
        
        #Elimina la columna ServiceArea
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
        
        
        v_modelo <- paste("Modelo [",v_contador,"de",v_total,"]","trees=",i,"shrinkage=",j,"minobsinnode=",k,"interaction_depth=",l)
        model.start <- Sys.time()
        print("-----------------------------------------")
        print("INICIO")
        print(v_modelo)
        #print("-----------------------------------------")
        
        df.gbm.ctrl <- trainControl(method = 'cv',
                                    number = v_cv_qty,
                                    verboseIter = TRUE,
                                    classProbs = TRUE,
                                    search = 'grid',
                                    summaryFunction = df.fn_summary)        

        df.gbm.grid <-  expand.grid(n.trees = i, 
                                    shrinkage = j,
                                    n.minobsinnode = k,
                                    interaction.depth = l)
        
                              
        
        df.gbm <- train(form = df.form,
                        data = df.part$train,
                        method = 'gbm',
                        trControl = df.gbm.ctrl,
                        tuneGrid = df.gbm.grid,
                        verbose = FALSE,
                        metric = df.metric
                      )

        print(df.gbm)

        #plot(df.gbm)

        df.gbm.model <- df.gbm$finalModel

        df.gbm.results <- fn_results(df.gbm)

        print('Umbral')
        print(df.gbm.results$prob_thr)
        v_umbral <- df.gbm.results$prob_thr

        print('Utilidad en cv')

        v_util_cv <- df.gbm.results$utility
        print(v_util_cv)
        
        print('Utilidad en Train')
        df.gbm.train.prob <- predict(df.gbm, newdata = df.part$train, type = 'prob')
        df.gbm.train.pred <- fn_pred(df.gbm.train.prob, thr = df.gbm.results$prob_thr)
        df.gbm.train.utility <- fn_utility(df.gbm.train.pred, df.part$train$Churn)
        
        
        v_util_train <- df.gbm.train.utility
        print(v_util_train)

        print('Utilidad en dev')

        df.gbm.dev.prob <- predict(df.gbm, newdata = df.part$dev, type = 'prob')
        df.gbm.dev.pred <- fn_pred(df.gbm.dev.prob, thr = df.gbm.results$prob_thr)
        df.gbm.dev.utility <- fn_utility(df.gbm.dev.pred, df.part$dev$Churn)

        v_util_dev <- df.gbm.dev.utility
        print(v_util_dev)

        print('Matriz de confusion en dev')
        df.gbm.dev.cm <- conf_matrix(df.gbm.dev.pred, df.part$dev$Churn)
        print(df.gbm.dev.cm)

        #print('** Generacion de la prediccion sobre test sample')

        test_sample <- read.csv('../data/test_sample.csv')
        rownames(test_sample) <- test_sample$CustomerID
        test_sample$CustomerID <- NULL
        test_sample$ServiceArea <- NULL

#        file_id <- paste0(c(script.name, v_contador, script.date,v_cv_qty,v_trees,v_shrinkage,v_minobsinnode,v_interaction_depth), collapse = '_')
         file_id <- paste0(c(script.name, v_contador, script.date,v_cv_qty,i,j,k,l), collapse = '_')

        gen_prediction(df.gbm, test_sample, prob_thr = df.gbm.results$prob_thr, id = file_id)

      
      
      model.done <- Sys.time()
      v_timeexec <- model.done - model.start
      v_timeexec_secs <- as.double(difftime(model.done,model.start,units="secs"))
      df.resultados[nrow(df.resultados)+1,] <- c(v_contador,v_umbral,v_util_cv,v_util_train,v_util_dev,v_timeexec_secs,v_cv_qty,i,j,k,l)
      write.csv(df.resultados,v_resultados)
      #print("-----------------------------------------")
      print(paste("Utilidad en Train: ",v_util_train, "Utilidad en dev:",v_util_dev))
      print(v_modelo)
      print(v_timeexec)
      print("-----------------------------------------")
      v_contador <- v_contador + 1  
      }
    }
  }
}
print('Done')
script.done <- Sys.time()
print(script.done - script.start)

df.resultados





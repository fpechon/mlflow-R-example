## This scripts runs a GBM model with the variables:
##  'CarAge', 'DriverAge', 'Density', 'Power', 'Brand','Gas', 'Region'

library(arrow)
library(mlflow)
library(ggplot2)
library(caret)
library(gbm)
library(pdp)
library(carrier)
library(reshape2)
library(parallel)

deviance_poisson = function(NClaims, lambda){
  2*(sum(dpois(x =NClaims, lambda = NClaims,log=TRUE))-
       sum(dpois(x = NClaims, lambda = lambda,
                 log=TRUE)))
}


## ------------------------------------------------------------------------------

mlflow_set_tracking_uri(Sys.getenv('mlflow_tracking_uri'))
mlflow_set_experiment("Claim Frequencies")

dataset = read_parquet("../data/dataset.parquet")



with(mlflow_start_run(), {

  n.trees = mlflow_param("n_trees", 20, "numeric")
  interaction.depth = mlflow_param("interaction_depth", 8, "numeric")
  n.minobsinnode = mlflow_param("n_minobsinnode", 200, "numeric")
  shrinkage = mlflow_param("shrinkage", 0.1, "numeric")
  bag.fraction = mlflow_param("bag_fraction", 0.5, "numeric")
  train.fraction = mlflow_param("train_fraction", 1, "numeric")
  objective = mlflow_param("objective", "poisson", "string")
  
  
  # mlflow_log_param("n.trees", n.trees)
  # mlflow_log_param("interaction.depth", interaction.depth)
  # mlflow_log_param("n.minobsinnode", n.minobsinnode)
  # mlflow_log_param("shrinkage", shrinkage)
  # mlflow_log_param("bag.fraction", bag.fraction)
  # mlflow_log_param("train.fraction", train.fraction)
  # mlflow_log_param("objective", objective)
  
  ## ------------------------------------------------------------------------------
  
  set.seed(21)
  in_training = createDataPartition(dataset$ClaimNb, times = 1, p = 0.8, list = FALSE)
  training_set = dataset[in_training, ]
  testing_set = dataset[-in_training, ]
  
  ## ------------------------------------------------------------------------------
  
  
  cols = c('CarAge', 'DriverAge', 'Power', 'Brand', 'Gas', 'Region', 'Density')
  set.seed(89)
  m0_gbm = gbm(ClaimNb ~ offset(log(Exposure)) + CarAge + DriverAge + Power + Brand + Gas + Region + Density,
                data = training_set,
                distribution = objective,
                n.trees = n.trees,
                interaction.depth = interaction.depth,
                n.minobsinnode = n.minobsinnode,
                shrinkage = shrinkage,
                bag.fraction = bag.fraction,
                train.fraction = train.fraction,
                verbose = TRUE,
                keep.data = TRUE,
                cv.folds = 5,
                n.cores = parallel::detectCores()-1) #Parallel computing
  
  # svg(filename = "temp/n_iter.svg", height=6, width=6)
  # gbm::gbm.perf(m0_gbm)
  # dev.off()
  # mlflow_log_artifact("temp/n_iter.svg")
  
  ## ------------------------------------------------------------------------------
  
  best_iter = gbm.perf(m0_gbm, method = "cv")
  mlflow_log_param("best_iter", best_iter)
  
  
  ## ------------------------------------------------------------------------------
  
  res = matrix(NA, 7, 7)
  for (i in 1:6) {
    for (j in (i + 1):7) {
      res[i, j] = 
        interact.gbm(m0_gbm, data = training_set, i.var = c(i,j), best_iter)
    }
  }
  diag(res) = 0
  row.names(res) = cols
  colnames(res) = row.names(res)
  interact_melt <- melt(res, na.rm = TRUE)
  plot = ggplot(data = interact_melt, aes(x = Var1, y = Var2, fill = value)) + geom_tile(color = "white") +
    scale_fill_gradient2(low = "white", mid = "gray", high = "blue", name = "Friedman's\nH-statistic") +
    theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 12, hjust = 1)) +
    coord_fixed()
  #ggsave("temp/H_statistic.svg", device="svg")
  #mlflow_log_artifact("temp/H_statistic.svg")
  
  htmlwidgets::saveWidget(
    widget = plotly::ggplotly(plot), #the plotly object
    file = paste0("temp/H_statistic.html"), #the path & file name
    selfcontained = TRUE #creates a single html file
  )
  mlflow_log_artifact(paste0("temp/H_statistic.html"))
  
  
  ## ------------------------------------------------------------------------------
  
  ## Add testing set error as metric
  lambda = predict(m0_gbm, 
                   testing_set, 
                   n.trees = best_iter, 
                   type="response") * testing_set$Exposure
  mlflow_log_metric("test_set_poisson_log_loss", 
                    deviance_poisson(testing_set$ClaimNb, lambda)
  )
  
  ## Add training set error as metric
  lambda = predict(m0_gbm, 
                   training_set, 
                   n.trees = best_iter, 
                   type="response") * training_set$Exposure
  mlflow_log_metric("train_poisson_log_loss", 
                    deviance_poisson(training_set$ClaimNb, lambda)
  )
  
  ## ------------------------------------------------------------------------------

  pdp_data = list()
  
  for (variable in cols){
    pdp_data[[variable]] = pdp::partial(m0_gbm, 
                                        pred.var=variable, 
                                        train = training_set, 
                                        offset = log(training_set$Exposure),
                                        n.trees = best_iter,
                                        type="regression")
    pdp_data[[variable]]['yhat'] = exp(pdp_data[[variable]]['yhat'])
  }
  
  ymin = do.call("min", lapply(pdp_data, function(x){min(x["yhat"])}))
  ymax = do.call("max", lapply(pdp_data, function(x){max(x["yhat"])}))
  
  
  for (variable in cols){
    plot = ggplot(pdp_data[[variable]], aes(x = .data[[variable]], y = .data[["yhat"]])) + 
      geom_point() + 
      geom_line() + 
      theme_bw()+
      scale_y_continuous(limits = c(0, ymax), name="Claim Frequency", 
                         labels = scales::percent_format(), 
                         breaks = seq(0,ymax,0.0025))
    # ggsave(paste0("temp/pdp_", variable, ".svg"), plot,  device="svg")
    # mlflow_log_artifact(paste0("temp/pdp_", variable, ".svg"))
    
    htmlwidgets::saveWidget(
      widget = plotly::ggplotly(plot), #the plotly object
      file = paste0("temp/pdp_", variable, ".html"), #the path & file name
      selfcontained = TRUE #creates a single html file
    )
    mlflow_log_artifact(paste0("temp/pdp_", variable, ".html"))
  }
  
  ## ---------------------------------------------------------------------------
  ## Add model to artifacts
  
  crate_model = carrier::crate( ~ gbm::predict.gbm(m0_gbm,.x, type="response") * .x['Exposure'],
                                m0_gbm = m0_gbm)
  mlflow_log_model(crate_model, "gbm")
  
})


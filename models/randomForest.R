# Sys.setenv(RENV_DOWNLOAD_METHOD = "libcurl") 
# renv::restore()


library(rfCountData)
library(arrow)
library(pdp)
library(ggplot2)
library(plotly)
library(mlflow)
library(caret)


deviance_poisson = function(NClaims, lambda){
  2*(sum(dpois(x =NClaims, lambda = NClaims,log=TRUE))-
       sum(dpois(x = NClaims, lambda = lambda,
                  log=TRUE)))
}


mlflow_set_tracking_uri(Sys.getenv('mlflow_tracking_uri'))
mlflow_set_experiment("Claim Frequencies")

dataset = read_parquet("data/dataset.parquet")


## ------------------------------------------------------------------------------
set.seed(21)
in_training = createDataPartition(dataset$ClaimNb, times = 1, p = 0.8, list = FALSE)
training_set = dataset[in_training, ]
testing_set = dataset[-in_training, ]


## ------------------------------------------------------------------------------


with(mlflow_start_run(), {
  
  ## ------------------------------------------------------------------------------

  
  
  ntree = mlflow_param("ntree", 10, type="numeric")
  mtry = mlflow_param("mtry", 3, type="numeric")
  nodesize = mlflow_param("nodesize", 10000, type="numeric")
  maxnodes = mlflow_param("maxnodes", 10, type="numeric")
    
    
  mlflow_log_param("ntree", ntree)
  mlflow_log_param("mtry", mtry)
  mlflow_log_param("nodesize", nodesize)
  mlflow_log_param("maxnodes", maxnodes)
  
  ## ------------------------------------------------------------------------------
  
  
  cols = c("Power", "CarAge", "DriverAge", "Brand", "Gas", "Region", "Density")
 
  ## ------------------------------------------------------------------------------
  
  
  set.seed(5) # For reproducibility
  m0_rf = rfPoisson(x = training_set[cols],
                    offset = log(training_set$Exposure),
                    y = training_set$ClaimNb,
                    ntree = ntree, # Number of trees in the forest
                    nodesize = nodesize, # Minimum number of observations in each leaf
                    mtry = mtry, # Number of variables drawn at each node
                    maxnodes = maxnodes, # Max number of nodes
                    importance=TRUE,
                    keep.inbag = TRUE,
                    do.trace=TRUE)
  
  ## ------------------------------------------------------------------------------
  
  ## Add testing set error as metric
  lambda = predict(m0_rf, testing_set[cols], log(testing_set$Exposure))
  mlflow_log_metric("test_set_poisson_log_loss", 
                    deviance_poisson(testing_set$ClaimNb, lambda)
                    )
  
  ## Add OOB error as metric
  mlflow_log_metric("OOB_poisson_log_loss", 
                    m0_rf$dev[ntree]
                    )
  
  ## Add training set error as metric
  lambda = predict(m0_rf, training_set[cols], log(training_set$Exposure))
  mlflow_log_metric("train_poisson_log_loss", 
                    deviance_poisson(training_set$ClaimNb, lambda)
                    )
              
  
  
  
  
  ## ------------------------------------------------------------------------------
  
  svg("temp/variable_importance.svg", width = 6, height=6)
  varImpPlot(m0_rf)
  dev.off()
  
  mlflow_log_artifact("temp/variable_importance.svg")
  
  ## ------------------------------------------------------------------------------
  
  OOB_error = plot(m0_rf)
  ggsave("temp/OOB_error.svg", device="svg")
  mlflow_log_artifact("temp/OOB_error.svg")
  
  
  ## ---------------------------------------------------------------------------
  
  pdp_data = list()
  
  for (variable in cols){
    pdp_data[[variable]] = pdp::partial(m0_rf, 
                                        pred.var=variable, 
                                        train = training_set, 
                                        offset = log(training_set$Exposure),
                                        type="regression")
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

  crate_model = carrier::crate( ~ utils::getFromNamespace("predict.rfCountData", 
                                                          "rfCountData")(m0_rf,
                                                                         .x[!names(.x) == 'Exposure'],
                                                                         log(.x$Exposure)),
                                m0_rf = m0_rf)
  mlflow_log_model(crate_model, "randomForest")
  
  }
)

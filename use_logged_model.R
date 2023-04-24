library(mlflow)
library(arrow)

# Set Uri of Mlflow server
mlflow_set_tracking_uri(Sys.getenv('mlflow_tracking_uri'))


# Load model
logged_model = 'runs:/65610e1b22914504bc5577a1d3954f90/glmnet'
loaded_model = mlflow_load_model(logged_model)


dataset = read_parquet("data/dataset.parquet")
loaded_model(dataset)



logged_model = 'runs:/356b38f0e64e441bacbf5a6bbc3fe173/randomForest'
logged_model = 'runs:/cbff73ce27764652b3b1cc1a5372d6b7/gbm'
# Load model
loaded_model = mlflow_load_model(logged_model)
loaded_model(head(dataset, 10))

cbind(loaded_model(dataset), predict(m0_rf, dataset, offset = log(dataset$Exposure)))



mlflow_rfunc_serve(logged_model)


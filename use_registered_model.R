library(mlflow)

# Set Uri of Mlflow server
mlflow_set_tracking_uri(Sys.getenv('mlflow_tracking_uri'))

# Get model currently in Production

i=1
while (mlflow_get_latest_versions("Claim Frequency Model")[[i]]$current_stage != "Production"){
  i= i+1
}

# Get info of model in prod
in_prod_model_version = mlflow_get_latest_versions("Claim Frequency Model")[[i]]

# Load model
model = mlflow_load_model(in_prod_model_version$source)
                          
# Load some data
dataset = arrow::read_parquet("data/dataset.parquet")
model(head(dataset))

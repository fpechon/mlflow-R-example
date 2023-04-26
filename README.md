# mlflow-R-example

# How to install and use MLFLOW with a AWS account ?

1. Create a AWS account. There are some free fonctionnalities (free tier). Create a new user using IAM. Note your access key and secret key somewhere and do not loose them. Connect to the AWS account using that user.
2. Follow the guide on the following page: https://allcloud.io/blog/organise-your-ml-experiments-with-mlflow-on-aws/. Do not forget to create **Roles** in IAM, so that the EC2 has access to S3 and the database.
3. a ) If you want the latest version of MLflow (currently 2.3), you will need to install Python 3.10 rather than 3.7 as on the page above. In this case follow the following guide as well when your instances are ready and that you have not yet installed mlflow https://techviewleo.com/how-to-install-python-on-amazon-linux-2/
3. b) Create an environment with Python 3.10 (as in the last steps of the guide in 3.a). Install **mlflow** on this environment, as well as **boto3**
4. Launch Mlflow with a command such as:
```
mlflow server --backend-store-uri postgresql://postgres:YOURPASSWORD@YOUR-DATABASE-ENDPOINT:5432 --default-artifact-root s3://YOURORGANISATION.MLFLOW.BUCKET --host 0.0.0.0
```

You can then access the mlflow server with your browser by connecting to the **Public IPv4 DNS** of the EC2 instance on the port 5000

```
http://<IPV4 DNS>:5000
```

5. You need to have access to S3 from your laptop/PC. For that, you will need to install AWS CLI (https://aws.amazon.com/fr/cli/).
6. when installed, you can configure by going into cmd and typing

```
aws configure
```

Use your user's credential (access and secret key) and you should be good to go.
7. You need to have python installed on your laptop, and have mlflow installed as well (python package).
8. In R, you also need to install mlflow.
9. Set the environment variable "MLFLOW_TRACKING_URI" to ```http://<IPV4 DNS>:5000```, e.g., in R you can run 
```
Sys.setenv(MLFLOW_TRACKING_URI="http://<IPV4 DNS>:5000")
```
10. Create an experiment in mlflow ui.
11. Enjoy.


PS: If may need to create a "temp" folder to store the temp files (like partial dependencies) before they are sent to s3.
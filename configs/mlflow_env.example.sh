# MLflow server settings. Copy OUTSIDE the repository and fill in, then `source` it before running anything:
#   cp configs/mlflow_env.example.sh ~/work/mlflow_env.sh && chmod 600 ~/work/mlflow_env.sh && nano ~/work/mlflow_env.sh
#   source ~/work/mlflow_env.sh
# Never commit the filled-in copy. The password is read from the environment only and is masked in every log.
export MLFLOW_TRACKING_URI=https://mlflow.gpu.uz
export MLFLOW_TRACKING_USERNAME=<user>
export MLFLOW_TRACKING_PASSWORD=<password>
# optional
# export MLFLOW_TRACKING_SERVER_CERT_PATH=/path/to/ca-bundle.crt    # if the server uses an internal certificate
# export MLFLOW_DISABLE=1                                            # switch tracking off everywhere

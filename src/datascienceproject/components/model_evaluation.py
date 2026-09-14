import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
from src.datascienceproject.entity.config_entity import ModelEvaluationConfig
from src.datascienceproject.utils.common import save_json
from pathlib import Path

# os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/parastoof/data-science-project.mlflow"
# os.environ["MLFLOW_TRACKING_USERNAME"]="parastoof"
# os.environ["MLFLOW_TRACKING_PASSWORD"]="a55b8c0ae54a9cdd2a268cee2b91031a09e6e24a"

class ModelEvaluation:
    def __init__(self,config:ModelEvaluationConfig):
            self.config=config

    def eval_metrics(self, actual, pred):
          rmse=np.sqrt(mean_squared_error(actual, pred))
          mae=mean_absolute_error(actual, pred)
          r2=r2_score(actual, pred)
          return rmse, mae, r2

    def log_into_mlflow(self):
        test_data=pd.read_csv(self.config.test_data_path)
        model=joblib.load(self.config.model_path)
        test_x =test_data.drop([self.config.target_column], axis=1)
        test_y =test_data[[self.config.target_column]]

        mlflow.set_registry_uri(self.config.mlflow_url)
        tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_quaities = model.predict(test_x)
            (rmse, mae, r2) = self.eval_metrics(test_y, predicted_quaities)

            scores={"rmse":rmse, "mae":mae, "r2":r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)
            
            mlflow.log_params(self.config.all_params)
    
            mlflow.log_metric("rmse",rmse)
            mlflow.log_metric("mae",mae)
            mlflow.log_metric("r2",r2)
            
            if tracking_url_type_store!="file":
                mlflow.sklearn.log_model(model,"model",registered_model_name="ElasticNet Model")
            else:
                mlflow.sklearn.log_model(model,"model")
        
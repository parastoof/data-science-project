from src.datascienceproject.config.configuration import ConfiguationManager
from src.datascienceproject.components.model_evaluation import ModelEvaluation
from src.datascienceproject import logger
from pathlib import Path


STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass
    def initiate_model_evaluation(self):
        config=ConfiguationManager()
        model_evaluation_config=config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.log_into_mlflow()

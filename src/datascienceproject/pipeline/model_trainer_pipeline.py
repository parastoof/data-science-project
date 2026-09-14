from src.datascienceproject.config.configuration import ConfiguationManager
from src.datascienceproject.components.model_trainer import ModelTrainer
from src.datascienceproject import logger
from pathlib import Path


STAGE_NAME = "Model Trainer Stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass
    def initiate_model_trainer(self):
        config=ConfiguationManager()
        model_trainer_config=config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()

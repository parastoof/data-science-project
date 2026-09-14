from src.datascienceproject import logger
from src.datascienceproject.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.datascienceproject.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
from src.datascienceproject.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.datascienceproject.pipeline.model_trainer_pipeline import ModelTrainerTrainingPipeline
from src.datascienceproject.pipeline.model_evaluation_pipeline import ModelEvaluationTrainingPipeline


STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<") 
    data_ingestion= DataIngestionTrainingPipeline() 
    data_ingestion.initiate_data_ingestion() 
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx=======x")
except Exception as e: 
    logger.exception(e) 
    raise e

STAGE_NAME = "Data Validation Stage"

try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<") 
    data_validation= DataValidationTrainingPipeline() 
    data_validation.initiate_data_validation() 
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx=======x")
except Exception as e: 
    logger.exception(e) 
    raise e


STAGE_NAME = "Data Transformation Stage"

try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<") 
    obj = DataTransformationTrainingPipeline() 
    obj.initiate_data_transformation() 
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx=======x")
except Exception as e: 
    logger.exception(e) 
    raise e


STAGE_NAME = "Model Trainer Stage"

try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<") 
    obj = ModelTrainerTrainingPipeline() 
    obj.initiate_model_trainer() 
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx=======x")
except Exception as e: 
    logger.exception(e) 
    raise e


STAGE_NAME = "Model Evaluation Stage"

try:
    logger.info(f">>>> stage {STAGE_NAME} started <<<<") 
    obj = ModelEvaluationTrainingPipeline() 
    obj.initiate_model_evaluation() 
    logger.info(f">>>> stage {STAGE_NAME} completed <<<<\n\nx=======x")
except Exception as e: 
    logger.exception(e) 
    raise e

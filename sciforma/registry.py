
from mmengine.registry import Registry, build_model_from_cfg

MODELS = Registry('model', build_model_from_cfg, locations=['sciforma.models'])
DATASETS = Registry('dataset', build_model_from_cfg, locations=['sciforma.datasets'])
TRAIN_ITERATION_FUNCS = Registry('train_iteration_funcs', locations=['sciforma.train_iteration_funcs'])
VALIDATION_FUNCS = Registry('validation_funcs', locations=['sciforma.validation_funcs'])

# Pipeline registry for dynamic pipeline class resolution
PIPELINES = Registry('pipeline', locations=['sciforma.pipelines'])
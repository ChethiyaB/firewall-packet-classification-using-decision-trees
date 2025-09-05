from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

def make_training_pipeline(preprocessor, estimator, resampling_cfg: dict):
    steps = [("pre", preprocessor)]
    if resampling_cfg.get("use_smote", True):
        steps.append(("smote", SMOTE(k_neighbors=resampling_cfg.get("k_neighbors",5), random_state=0)))
    steps.append(("clf", estimator))
    return Pipeline(steps)

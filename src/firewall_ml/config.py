from dataclasses import dataclass
from pathlib import Path
import yaml, os

@dataclass
class Config:
    seed: int
    data_path: str | None
    data_url: str | None
    target: str
    test_size: float
    cv_folds: int
    scoring: str
    preprocess: dict
    resampling: dict
    models: list[dict]
    artifacts_dir: str

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    data_url = os.getenv(cfg["data"].get("url_env", ""), None)
    return Config(
        seed=cfg["seed"],
        data_path=cfg["data"].get("path"),
        data_url=data_url,
        target=cfg["target"],
        test_size=cfg["test_size"],
        cv_folds=cfg["cv_folds"],
        scoring=cfg["scoring"],
        preprocess=cfg["preprocess"],
        resampling=cfg["resampling"],
        models=cfg["models"],
        artifacts_dir=cfg["artifacts_dir"],
    )

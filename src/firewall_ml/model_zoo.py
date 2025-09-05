import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def build_preprocessor(df: pd.DataFrame, target: str, cfg: dict) -> ColumnTransformer:
    X = df.drop(columns=[target])
    num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            make_num(cfg),
            num_cols
        ))
    if cat_cols:
        transformers.append((
            "cat",
            make_cat(cfg),
            cat_cols
        ))
    return ColumnTransformer(transformers)

def make_num(cfg: dict):
    steps = [("impute", SimpleImputer(strategy="most_frequent"))]
    if cfg.get("scale_numeric", True):
        steps.append(("scale", StandardScaler(with_mean=False)))
    from sklearn.pipeline import Pipeline
    return Pipeline(steps)

def make_cat(cfg: dict):
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    from sklearn.pipeline import Pipeline
    return Pipeline([("impute", SimpleImputer(strategy=cfg.get("impute_strategy","most_frequent"))),
                     ("ohe", enc)])

import typer, joblib, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from .config import load_config
from .data import load_dataframe
from .features import build_preprocessor
from .model_zoo import make_model
from .pipeline import make_training_pipeline

app = typer.Typer()

@app.command()
def main(cfg_path: str = "configs/default.yaml"):
    cfg = load_config(cfg_path)
    df = load_dataframe(cfg.data_path, cfg.data_url)
    X, y = df.drop(columns=[cfg.target]), df[cfg.target]

    pre = build_preprocessor(df, cfg.target, cfg.preprocess)

    Path(cfg.artifacts_dir).mkdir(exist_ok=True, parents=True)

    for m in cfg.models:
        est = make_model(m["name"], m.get("params", {}))
        pipe = make_training_pipeline(pre, est, cfg.resampling)

        skf = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
        scores = cross_val_score(pipe, X, y, scoring=cfg.scoring, cv=skf, n_jobs=-1)
        print(f"{m['name']} {cfg.scoring}: {scores.mean():.4f} ± {scores.std():.4f}")

        # final fit and persist
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed)
        pipe.fit(Xtr, ytr)
        yhat = pipe.predict(Xte)
        print(classification_report(yte, yhat))

        out = Path(cfg.artifacts_dir, f"{m['name']}.joblib")
        joblib.dump(pipe, out)
        print(f"Saved: {out}")

if __name__ == "__main__":
    app()

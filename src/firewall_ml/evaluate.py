import typer, joblib, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
app = typer.Typer()

@app.command()
def main(model_path: str, data_path: str, target: str):
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    X, y = df.drop(columns=[target]), df[target]
    yhat = model.predict(X)
    print(classification_report(y, yhat))
    print(confusion_matrix(y, yhat))

if __name__ == "__main__":
    app()

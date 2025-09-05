from pathlib import Path
import pandas as pd
import requests

def load_dataframe(local_path: str | None, remote_url: str | None) -> pd.DataFrame:
    if remote_url:
        # stream & cache
        Path("data").mkdir(exist_ok=True)
        out = Path("data/remote.csv")
        with requests.get(remote_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(out, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        return pd.read_csv(out)
    if local_path:
        return pd.read_csv(local_path)
    raise ValueError("No data source provided.")

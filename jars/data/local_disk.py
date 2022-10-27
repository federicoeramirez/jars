import pandas as pd
import os

def get_local_data():
    """
    Loads local data with optimized dtypes.
    """
    dtypes = {
    "Unnamed: 0": "int16",
    "valence": "float32",
    "year": "int16",
    "acousticness": "float32",
    "artists": "O",
    "danceability": "float32",
    "duration_ms": "int32",
    "energy": "float32",
    "explicit": "int8",
    "id": "O",
    "instrumentalness": "float32",
    "key": "int8",
    "liveness": "float32",
    "loudness": "float32",
    "mode": "int8",
    "name": "O",
    "popularity": "int8",
    "release_date": "O",
    "speechiness": "float32",
    "tempo": "float32",
    "lyrics": "O"
    }

    path = os.path.join(
        os.environ.get("LOCAL_DATA_PATH"),
        "data_lyrics_10k_sorted.csv") # replace with full preprocessed dataset

    df = pd.read_csv(path, dtype=dtypes)

    if 'Unnamed: 0' in df.columns:
        # remove column if exists
        df = df.drop(columns=['Unnamed: 0'])

    df = df.drop_duplicates()

    return df

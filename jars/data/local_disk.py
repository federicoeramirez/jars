import pandas as pd

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

    path = '~/code/federicoeramirez/jars/raw_data/data_lyrics_10k_preprocessed.csv' # replace with full preprocessed dataset

    df = pd.read_csv(path, dtype=dtypes)

    if 'Unnamed: 0' in df.columns:
        # remove column if exists
        df = df.drop(columns=['Unnamed: 0'])

    if 'Unnamed: 0.1' in df.columns:
        # remove column if exists
        df = df.drop(columns=['Unnamed: 0.1'])

    df['duration_s'] = df['duration_ms'].apply(lambda x: x/1000)
    df = df[(df['duration_s'] > 60) & (df['duration_s'] < 600)]
    df['duration_m'] = df['duration_s']/60
    df['artists'] = df['artists'].apply(lambda x: x.replace("['", '').replace("'", '').replace("]", ''))
    df['index'] = df['artists'] + ' - "' + df['name'] + '"'
    df = df.set_index(df['index']).sort_index(axis=1)
    df = df.drop(columns=['artists', 'name', 'release_date', 'duration_ms', 'duration_s'])

    df = df.drop_duplicates()

    print('Data loaded ✅')

    return df

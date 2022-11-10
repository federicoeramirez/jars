import pandas as pd

def get_local_data():
    """
    Loads local data with optimized dtypes.
    """
    dtypes = {
    "valence": "float32",
    "acousticness": "float32",
    "danceability": "float32",
    "duration_m": "float32",
    "energy": "float32",
    "explicit": "int8",
    "id": "O",
    "index":"O",
    "instrumentalness": "float32",
    "liveness": "float32",
    "loudness": "float32",
    "mode": "int8",
    "popularity": "float32",
    "release_date": "O",
    "speechiness": "float32",
    "tempo": "float32",
    "key_0": "int8",
    "key_1": "int8",
    "key_2": "int8",
    "key_3": "int8",
    "key_4": "int8",
    "key_5": "int8",
    "key_6": "int8",
    "key_7": "int8",
    "key_8": "int8",
    "key_9": "int8",
    "key_10": "int8",
    "key_11": "int8",
    "anger": "float32",
    "disgust": "float32",
    "joy": "float32",
    "fear": "float32",
    "neutral": "float32",
    "surprise": "float32",
    "sadness": "float32",
    }

    path = '/processed_data/data_full_processed.csv'

    data = pd.read_csv(path, dtype=dtypes)
    data = data.rename(columns={'index':'full_name'})

    data = data.drop(columns=['duration_m'])
    data = data.drop_duplicates()
    data = data.drop_duplicates(subset='full_name')

    print('Data loaded ✅')

    return data

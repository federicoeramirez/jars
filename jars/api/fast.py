from fastapi import FastAPI
from jars.data.local_disk import get_local_data
from jars.ml_logic.utils import (load_sp,
                                 load_genius,
                                 get_song_features,
                                 add_new_song,
                                 find_similar_songs)

app = FastAPI()

@app.get("/recommendation")
def get_recommendation(song: str, amount: int = 20): # playlist: bool = False
    """
    Receives song's name (and artist name) as input and returns a recommendation of songs.
    "amount" specifies the desired number (int) of recommended songs.
    "playlist" True or False to create playlist on Spotify with recommended songs.
    """
    if song == None:
        return print('Waiting for input...')

    if song == '':
        return print('Waiting for input...')

    # load spotipy to connect to the Spotify API
    sp = load_sp()

    # load genius to connect to the Genius API
    genius = load_genius()

    # load data
    data = get_local_data()

    # get song features
    full_name, id, features = get_song_features(song, sp, genius)

    if full_name not in data['full_name']:
        # add new song to dataframe
        data = add_new_song(data, full_name, id, features)

    # get similar songs and their id
    recommendation, recommendation_id = find_similar_songs(data,
                                                           features,
                                                           amount)

    """
    # playlist creation is False by default
    if playlist == True:
        # creates a playlist
        make_playlist(recommendation_id, sp)
    """

    return dict(recommendation), dict(recommendation_id)


@app.get("/")
def root():
    return {'greeting': 'Hello'}

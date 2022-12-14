from jars.data.local_disk import get_local_data
from jars.ml_logic.utils import (load_sp,
                                 load_genius,
                                 get_song_features,
                                 add_new_song,
                                 make_playlist,
                                 find_similar_songs)

def get_recommendation(song: str, amount: int = 15, playlist: bool = False):
    """
    Receives song's name (and artist name) as input and returns a recommendation of songs.
    "amount" specifies the desired number (int) of recommended songs.
    "playlist" True or False to create playlist on Spotify with recommended songs.
    """
    if song == None:
        return print('Waiting for input...')

    # load spotipy to connect to the Spotify API
    sp = load_sp()

    # load genius to connect to the Genius API
    genius = load_genius()

    # load data
    df = get_local_data()

    # get song features
    index, id, features = get_song_features(song, sp, genius)

    if id not in df['id']:
        # add new song to dataframe
        df = add_new_song(df, index, id, features)

    # get similar songs and their id
    recommendation, recommendation_id = find_similar_songs(df, features, int(amount))

    # playlist creation is False by default
    if playlist == True:
        # creates a playlist
        make_playlist(recommendation_id, sp)

    return dict(recommendation)

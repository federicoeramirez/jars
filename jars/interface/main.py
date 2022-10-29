from jars.data.local_disk import get_local_data
from jars.ml_logic.utils import (load_sp,
                                 get_song_features,
                                 add_new_song,
                                 make_playlist,
                                 find_similar_songs)

def get_recommendation(song: str, amount: int = 15, playlist: bool = True):
    """
    Receives artist and song name as input and gives a recommendation of songs.
    """
    # load spotipy to connect to the Spotify API
    sp = load_sp()

    # get song features
    index, id, features = get_song_features(song, sp)

    # load data
    df = get_local_data()

    if id not in df['id']:
        # add new song to dataframe
        df = add_new_song(df, index, id, features)

    # get similar songs and their id
    recommendation, recommendation_id = find_similar_songs(df, features, int(amount))

    # playlist creation is False by default
    if playlist == True:
        # creates a playlist
        make_playlist(recommendation_id, sp)

    print(f'List of {amount} recommended songs:')
    print(recommendation)

    return dict(recommendation)

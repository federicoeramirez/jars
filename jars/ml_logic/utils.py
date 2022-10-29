import os
import spotipy
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_sp():
    """
    Load needed credentials and permissions for the Spotify API.
    """
    cid = os.environ.get('SPOTIFY_CLIENT_ID')
    secret = os.environ.get('SPOTIFY_CLIENT_SECRET')
    redirect_uri = os.environ.get('SPOTIPY_REDIRECT_URI')
    scope = 'playlist-modify-private'
    username = None

    # auth_manager=SpotifyOAuth(scope=scope)
    # spotipy.Spotify(auth_manager=auth_manager)

    #spotify_token = spotipy.util.prompt_for_user_token(username,
    #                                                   scope,
    #                                                   cid,
    #                                                   secret,
    #                                                   redirect_uri)
    spotify_token = os.environ.get('SPOTIFY_TOKEN')
    sp = spotipy.Spotify(auth=spotify_token)

    print('Spotify credentials loaded ✅')

    return sp


def get_song_features(song: str, sp: object):
    """
    Receives search string and spotipy class object as input.
    Outputs song's "index", "id" and all "features" sorted.
    """
    # make Spotify API call to search for song info
    search_result = sp.search(song, limit=1)

    if search_result == None:
        return print('Song not found. Try searching again!')

    # get song id for search
    id = search_result['tracks']['items'][0]['id']

    # get song features
    features = sp.audio_features(id)[0]

    # create columns with necessary features
    features['year'] = search_result['tracks']['items'][0]['album']['release_date'][:4]
    features['explicit'] = search_result['tracks']['items'][0]['explicit']
    features['popularity'] = search_result['tracks']['items'][0]['popularity']

    # convert dict to series
    features = pd.Series(features)

    # cleanup
    features['explicit'] = features['explicit'] * 1
    features['duration_m'] = (features['duration_ms'] /1000)/60
    features = features.drop(['id',
                              'uri',
                              'track_href',
                              'analysis_url',
                              'type',
                              'duration_ms',
                              'time_signature']).sort_index()

    # get artist name
    artist = search_result['tracks']['items'][0]['artists'][0]['name']

    # get track name
    name = search_result['tracks']['items'][0]['name']

    # create index
    index = artist + ' - "' + name + '"'

    print('Song found. Features loaded ✅')

    return index, id, features


def add_new_song(df: pd.DataFrame, index: str, id: str, features: pd.Series):
    """
    Receives "df" with all songs, new song "index", new song "id" and new song "features".
    Adds a new song at the end of the df with all corresponding features.
    """
    song_df = pd.DataFrame(features).transpose()
    song_df['index'] = index
    song_df['id'] = id
    df = df.append(song_df.sort_index(axis=1), ignore_index=True)

    return df


def find_similar_songs(df: pd.DataFrame, features: pd.Series, amount=15):
    """
    Receives a "df" and a song's "features" as input.
    Returns similar songs sorted and their id.
    """
    # drop columns that won't be used for vector space
    df_processed = df.drop(columns=['id', 'lyrics', 'index'])
    df_processed = df_processed.set_index(df['index'])

    # create vector with song
    v1 = np.array(features).reshape(1, -1)

    # calculate cosine similarity
    sim1 = cosine_similarity(df_processed, v1).reshape(-1)

    # create dataframe with top recommendation
    recommendation_df = pd.DataFrame(sim1, index = df_processed.index)
    recommendation_df = recommendation_df.rename(columns={0:'cosine_similarity'}).reset_index()
    recommendation_df = recommendation_df.merge(df[['id', 'index']], how='left')
    recommendation_df.sort_values('cosine_similarity', ascending=False, inplace=True)

    recommendation = recommendation_df['index'].head(amount).reset_index()['index']
    recommendation_id = recommendation_df['id'].head(amount)

    return recommendation, recommendation_id


def make_playlist(recommendation_id: list, sp: object):
    """
    Receives a list of songs' id and spotipy class object as input.
    """
    # getting user id
    user_id = sp.me()['id']

    # creating a private playlist
    playlist = sp.user_playlist_create(user_id,
                                       'JARS 1.0',
                                       public=False,
                                       collaborative=False,
                                       description='JARS 1.0 testing')

    # obtaining playlist id
    playlist_id = playlist['id']

    username = None

    # add songs to playlist
    sp.user_playlist_add_tracks(username,
                                playlist_id,
                                recommendation_id,
                                position=None)

    print('Playlist saved ✅')

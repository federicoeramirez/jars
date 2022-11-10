import os
import spotipy
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from lyricsgenius import Genius
from jars.ml_logic.lyrics import get_lyrics, preprocess_language
from jars.ml_logic.emotions import get_emotions
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load_sp():
    """
    Load needed credentials and permissions for the Spotify API.
    """
    client_credentials_manager = spotipy.oauth2.SpotifyClientCredentials()
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    sp.trace = True

    print('Spotify credentials loaded ✅')

    return sp


def load_genius():
    """
    Load needed credentials for Genius API.
    """
    genius_token = os.environ.get('LYRICSGENIUS_ACCESS_TOKEN')

    # instantiate the Genius class with some useful hyperparameters
    genius = Genius(genius_token,
                    timeout=220,
                    remove_section_headers=True,
                    skip_non_songs=True)

    return genius


def get_song_features(song: str, sp: object, genius: object):
    """
    Receives search string and spotipy class object as input.
    Outputs song's "full_name", "id" and all "features" sorted.
    """
    # make Spotify API call to search for song info
    search_result = sp.search(song, limit=1)

    if search_result == None:
        return print('Song not found. Try searching again!')

    print('Song found in Spotify API ✅')

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
    full_name = artist + ' - "' + name + '"'

    features['lyrics'] = get_lyrics(artist, name, genius)
    features = preprocess_language(features)

    features['emotions'] = get_emotions(features['lyrics'])
    features = pd.concat([features, pd.Series(features['emotions']).fillna('None')], axis=0)

    features = features.drop(['lyrics',
                              'translated_lyrics',
                              'language',
                              'emotions',
                              'year'])

    features = preprocess(features.to_frame().transpose())
    features = features.sort_index(axis=1)

    print('Features loaded ✅')

    return full_name, id, features


def preprocess(features):
    # StandarScaler normal distribution
    scaler = StandardScaler()
    features['tempo'] = scaler.fit_transform(features[['tempo']])

    # MinMaxScaler skewed data
    mm_scaler = MinMaxScaler()
    features['loudness'] = mm_scaler.fit_transform(features[['loudness']])
    features['popularity'] = mm_scaler.fit_transform(features[['popularity']])

    # OneHotEncode categorical features
    song_key = features['key']

    for key in range(12):
        if key == int(song_key):
            features[f'key_{key}'] = 1
        else:
            features[f'key_{key}'] = 0

    features = features.drop(columns=['key', 'duration_m'])

    return features


def add_new_song(df: pd.DataFrame, full_name: str, id: str, features: pd.DataFrame):
    """
    Receives "df" with all songs, new song "index", new song "id" and new song "features".
    Adds a new song at the end of the df with all corresponding features.
    """
    features['full_name'] = full_name
    features['id'] = id
    df = df.append(features, ignore_index=True)

    return df


def find_similar_songs(df: pd.DataFrame, features: pd.Series, amount=20):
    """
    Receives a "df" and a song's "features" as input.
    Returns similar songs sorted and their id.
    """
    # drop columns that won't be used for vector space
    df_processed = df.drop(columns=['id',
                                    'full_name'])

    df_processed = df_processed.set_index(df['full_name']).sort_index(axis=1)

    features_processed = features.drop(columns=['id',
                                    'full_name']).sort_index(axis=1)
    # create vector with song
    v1 = np.array(features_processed).reshape(1, -1)

    # calculate cosine similarity
    sim1 = cosine_similarity(df_processed, v1).reshape(-1)

    # create dataframe with top recommendation
    recommendation_df = pd.DataFrame(sim1, index = df_processed.index)
    recommendation_df = recommendation_df.rename(columns={0:'cosine_similarity'}).reset_index()
    recommendation_df = recommendation_df.merge(df[['id', 'full_name']], how='left')
    recommendation_df.sort_values('cosine_similarity', ascending=False, inplace=True)
    recommendation_df = recommendation_df.reset_index()
    recommendation_df.drop_duplicates(subset='full_name', inplace=True)
    recommendation = recommendation_df['full_name'].reset_index(drop=True).head(amount)
    recommendation_id = recommendation_df['id'].head(amount)

    print('Recommendation ready ✅')

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

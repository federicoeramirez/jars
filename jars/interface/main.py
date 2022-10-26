import pandas as pd
import os
import spotipy
import spotipy.util as util
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from google.cloud import bigquery


# SPOTIFY CREDENTIALS
# needed credentials and permissions (scope)
cid = os.environ.get('SPOTIFY_CLIENT_ID')
secret = os.environ.get('SPOTIFY_CLIENT_SECRET')
redirect_uri = os.environ.get('SPOTIPY_REDIRECT_URI')
scope = 'playlist-modify-private'
username = None

spotify_token = util.prompt_for_user_token(username,
                                           scope,
                                           cid,
                                           secret,
                                           redirect_uri)

sp = spotipy.Spotify(auth=spotify_token)

print('Spotify credentials loaded ✅')


# LOADING DATA
# from bigquery
def get_bq_data():

    client = bigquery.Client()

    table = f"{os.environ['DATASET']}.{os.environ.get('TABLE')}"

    rows = client.list_rows(table)

    big_query_df = rows.to_dataframe()
    big_query_df.drop(columns=['int64_field_0'], inplace=True)

    print('Data loaded from BQ ✅')

    return big_query_df


# GET SONG FEATURES
def get_song_features(input):
    """
    Gets index, id and all audio_features sorted.
    """
    search_result = sp.search(input, limit=1)

    id = search_result['tracks']['items'][0]['id']

    # get audio features
    audio_features = sp.audio_features(id)[0]
    audio_features['year'] = search_result['tracks']['items'][0]['album']['release_date'][:4]
    audio_features['explicit'] = search_result['tracks']['items'][0]['explicit']
    audio_features['popularity'] = search_result['tracks']['items'][0]['popularity']

    # convert dict to series
    audio_features = pd.Series(audio_features)

    # cleanup
    audio_features['explicit'] = audio_features['explicit'] * 1
    audio_features['duration_m'] = (audio_features['duration_ms'] /1000)/60
    audio_features = audio_features.drop(['id', 'uri', 'track_href', 'analysis_url', 'type', 'duration_ms', 'time_signature']).sort_index()

    # get artist name
    artist_name = search_result['tracks']['items'][0]['artists'][0]['name']

    # get track name
    track_name = search_result['tracks']['items'][0]['name']

    # create index
    track_index = artist_name + ' - "' + track_name + '"'

    print('Audio features loaded ✅')

    return track_index, id, audio_features



def preprocess(data):
    # data cleanup and arrangements
    data['duration_s'] = data['duration_ms'].apply(lambda x: x/1000)
    data = data[(data['duration_s'] > 60) & (data['duration_s'] < 600)]
    data['duration_m'] = data['duration_s']/60
    data['artists'] = data['artists'].apply(lambda x: x.replace("['", '').replace("'", '').replace("]", ''))
    data['index'] = data['artists'] + ' - "' + data['name'] + '"'
    data = data.set_index(data['index']).sort_index(axis=1)
    data = data.drop(columns=['artists', 'name', 'index', 'id', 'release_date', 'lyrics', 'duration_ms', 'duration_s'])

    print('Data preprocessed ✅')

    return data


def get_recommendation(song_features, data, amount=15):
    """
    Compares cosine similarity between our requested song features and our dataset.
    Results sorted
    """

    v1 = np.array(song_features).reshape(1, -1)
    sim1 = cosine_similarity(data, v1).reshape(-1)

    recommendation_df = pd.DataFrame(sim1, index = data.index)
    recommendation_df = recommendation_df.rename(columns={0:'cosine_similarity'})
    recommendation_df.sort_values('cosine_similarity', ascending=False, inplace=True)

    recommendation_id = [get_song_features(song)[1] for song in recommendation_df.head(amount).index]

    return recommendation_df.head(amount), recommendation_id

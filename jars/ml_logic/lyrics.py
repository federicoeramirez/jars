import unidecode
from difflib import SequenceMatcher
from langdetect import detect
from google.cloud import translate_v2 as translate

def clean_text(text):
    """
    Removes symbols, accents and uppercases from text.
    """
    text = text.replace("[",'')\
        .replace("]",'')\
        .replace("'",'')\
        .replace(",",'')\
        .replace(":",'')\
        .replace(")",'')\
        .replace("(",'')\
        .replace(".",'')\
        .replace('"','')\
        .replace('/','')\
        .replace("\\",'')\
        .replace("(?)",'')\
        .replace("-",'')

    return unidecode.unidecode(text.lower())


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def get_lyrics(artist, song_title, genius):
    """
    Returns lyrics from Genius API.
    """
    # cleanup the inputs
    artist = clean_text(artist)
    song_title = clean_text(song_title)

    # call the API to search for the song
    song = genius.search_song(title=song_title, artist=artist)

    if song != None:
        # get song name from the API response to compare similarity
        api_response = unidecode.unidecode(song.to_dict()['full_title']\
            .replace('\xa0', ' ')\
            .replace('\u200b', ' ')\
            .lower())
        api_response = api_response.split(' by ')
        # check similarity
        song_similar = similar(api_response[0], song_title)
        # artist_similar = similar(api_response[1], artist)

        if song_similar >= 0.9:
            # all lyrics responses come with the song's title and 'Lyrics' str
            # so we count how many characters should be removed in order
            # to delete any extra text
            characters_to_remove = len(song.to_dict()['title'] + ' Lyrics')

            # get lyrics from API
            lyrics = genius.lyrics(song.to_dict()['id'])[characters_to_remove:-5]\
                .replace('\n', ' ')\
                .replace('\u205f', ' ')\
                .replace('\u2005', ' ')\
                .replace('\\', ' ')\
                .strip()

        else:
            lyrics = 'None'

    else:
        lyrics = 'None'
    if lyrics == 'None':
        print("Lyrics not found in Genius' database ❌")
    else:
        print('Lyrics loaded ✅')

    return lyrics


def detect_language(text):
    if text != 'None':
        return detect(text)
    else:
        return 'None'


def translate_text(text, language):
    if language != ('en', 'None'):
        translate_client = translate.Client()
        result = translate_client.translate(text, target_language='en')
        return result["translatedText"].replace('&#39;',"'")
    else:
        return text


def preprocess_language(data):

    data['language'] = detect_language(data['lyrics'])

    data['translated_lyrics'] = translate_text(data['lyrics'], data['language'])

    return data

import pandas as pd
from transformers import pipeline


model_path = "j-hartmann/emotion-english-distilroberta-base"
classifier = pipeline("text-classification",
                      model=model_path,
                      tokenizer=model_path,
                      top_k=None,
                      max_length=512,
                      truncation=True)


def get_emotions(lyrics):
    """
    Analyses emotions from text.
    """
    if lyrics != 'None':
        emotions = classifier(lyrics)
        emotions = sorted(emotions[0], key=lambda x: x["label"])
        new_dict = {}
        for dictionary in emotions:
            new_dict[dictionary['label']] = dictionary['score']

        print('Emotions loaded ✅')

        return new_dict

    else:
        emotions = {
            'anger':0,
            'disgust':0,
            'fear':0,
            'joy':0,
            'neutral':0,
            'sadness':0,
            'surprise':0
        }

        print('Default emotions loaded ✅')

        return emotions


def preprocess_emotions(data):
    """
    Receives a DataFrame and generates columns for each emotion (anger, disgust, fear, etc.).
    """
    data["emotions"] = get_emotions(data)
    data = pd.concat([data, pd.Series(data['emotions']).fillna('None')], axis=1)

    print('Emotions loaded ✅')

    return data.drop(columns=['emotions'])

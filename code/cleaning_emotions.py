import pandas as pd
from transformers import pipeline
import re
import requests
from fuzzywuzzy import fuzz
from google.cloud import translate_v2 as translate
from dotenv import load_dotenv
import os
load_dotenv()

'''
Assigning emotions to the lyrics generated. Only one run needed after running
filter_data and getdata
'''

MAX_TOKENS = 512
PROJECT_ID = os.environ['PROJECT_ID']
PARENT = f"projects/{PROJECT_ID}"

#Classifications for emotions_id
EMOTION_DICT = {
    'admiration': 0, 'amusement': 0, 'anger':1, 'annoyance': 1, 'approval': 0, 
    'caring': 0, 'confusion': 1, 'curiosity': 2, 'desire': 0, 'disappointment': 1, 
    'disapproval': 1, 'disgust': 1, 'embarrassment': 1, 'excitement': 0, 'fear': 1, 
    'gratitude': 0, 'grief': 1, 'joy': 0, 'love': 0, 'nervousness': 2, 
    'optimism': 0, 'pride': 0, 'realization': 2, 'relief': 2, 'remorse': 1, 'sadness': 1, 'surprise': 2, 'neutral': 2
}

TRANSLATE_DICT = {} #to store already translated songs
PREDICTIONS_DICT = {} #to store the predictions of songs already predicted

def translate_text(text, target_language_code):
    '''Translates each line in a lyric into English'''
    client = translate.Client("en")

    if isinstance(text, bytes):
        text = text.decode("utf-8")

    response = client.translate(text, target_language=target_language_code
    )

    return response["translatedText"]

def clean_lyrics(lyrics):
    '''
    Cleaning lyrics of any punction/special characters
    Removing all duplicate lines in a lyric, and then translating each line into
    English.
    '''
    cleaned_lyrics = re.sub(r'\[.*?\]', '', lyrics)
    sentences = cleaned_lyrics.split('\n')
    unique_sentences = []
    translated_sentences = []
    target_lang = "en"
    for sentence in sentences:
        is_duplicate = False
        for unique_sentence in unique_sentences:
            similarity = fuzz.token_set_ratio(sentence, unique_sentence)  # Using fuzzywuzzy for similarity calculation
            
            # Define a similarity threshold (adjust as needed)
            similarity_threshold = 85  # Adjust the threshold as needed
            
            # If the similarity exceeds the threshold, consider it a duplicate
            if similarity > similarity_threshold:
                is_duplicate = True
                break
        
        # If the sentence is not a duplicate, add it to the list of unique sentences
        if not is_duplicate:
            unique_sentences.append(sentence)
    for sent in unique_sentences:
        if sent in TRANSLATE_DICT:
            translated_sent = TRANSLATE_DICT[sent]
        else:
            translated_sent = translate_text(sent, target_lang)
            TRANSLATE_DICT[sent] = translated_sent
        translated_sentences.append(translated_sent)
    
    # Join the unique sentences to form the deduplicated lyrics
    deduplicated_lyrics = '\n'.join(translated_sentences)

    return deduplicated_lyrics
    
def make_emotions(songs_df):
    '''
    Generate emotion values for each song's lyrics using the HuggingFace model:
    https://huggingface.co/SamLowe/roberta-base-go_emotions

    '''

    songs_df['lyrics'] = songs_df['lyrics'].apply(clean_lyrics)
    #print(songs_df['lyrics'])
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
    emotions = []
    anger_list = []
    love_list = []
    sad_list = []
    batch_size = 256
    for index, row in songs_df.iterrows():
        if row['id'] in PREDICTIONS_DICT: #Checks if the song already has had predictions generated
            emotions.append(PREDICTIONS_DICT[row['id']][0])
            anger_list.append(PREDICTIONS_DICT[row['id']][1])
            love_list.append(PREDICTIONS_DICT[row['id']][2])
            sad_list.append(PREDICTIONS_DICT[row['id']][3])
        else:
            batch_emotions = [] #Batching lyrics for easier tokenization
            lyrics = row['lyrics']
            tokens = lyrics.split()

            token_batches = []
            for i in range(0, len(tokens), batch_size):
                if (len(tokens) - i) < batch_size:
                    token_batches.append(tokens[i:])
                else:
                    token_batches.append(tokens[i:i+batch_size])

            for batch in token_batches:
                batch_lyric = ' '.join(batch)
                emo = classifier([batch_lyric])
                batch_emotions.append(emo)
                #print(batch_emotions)
            final_emotions = {emotion['label']: 0 for emotion in batch_emotions[0][0]} 
            #Aggregating values of emotions across batches
            for batch in batch_emotions:
                for sublist in  batch_emotions:
                    for subsublist in sublist:
                        for emotion in subsublist:
                            final_emotions[emotion['label']] += emotion['score'] 
            num_batches = len(batch_emotions)
            average_emotions = {emotion: score / num_batches for emotion, score in final_emotions.items()}

            anger = max(average_emotions['anger'], average_emotions['annoyance'])
            anger_list.append(anger)
            love = max(average_emotions['love'], average_emotions['desire'], average_emotions['caring'])
            love_list.append(love)
            sad = max(average_emotions['sadness'], average_emotions['grief'])
            sad_list.append(sad)
            sorted_emotions = sorted(average_emotions.items(), key=lambda x: x[1], reverse=True)
            
            top_emotion = dict(sorted_emotions[:1])
            if EMOTION_DICT[next(iter(top_emotion.keys()))] == 2:
                top_emotion = dict(sorted_emotions[1:2])
            emotions.append(next(iter(top_emotion.keys())))
            dict_vals = [next(iter(top_emotion.keys())), anger, love, sad]
            PREDICTIONS_DICT[row['id']] = dict_vals

    
    
    songs_df['emotions'] = emotions
    songs_df['anger'] = anger_list
    songs_df['love'] = love_list
    songs_df['sadness'] = sad_list
    songs_df['emotions_id'] = songs_df['emotions'].map(EMOTION_DICT)
    print(songs_df.head())
    songs_df.to_csv('../data/song_emotions.csv')


def test_model():
    '''
    Only to be run for testing the code itself, to see how it performs on a small
    subset of the data. 
    '''
    test_df = pd.read_csv('../data/top_songs.csv').head(1)
    track_id = test_df['id'].iloc[0]
    querystring = {"trackId":track_id}
    headers = {
        "X-RapidAPI-Key": "e66ab9e6c9msh0a0457e0c88540dp140b34jsnde29bcd01798",
        "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
    }
    response = requests.get('https://spotify-scraper.p.rapidapi.com/v1/track/lyrics', headers=headers, params=querystring)
    test_df.loc[0, 'lyrics'] = response.text
    print(test_df)
    make_emotions(test_df)

#test_model()
songs_df = pd.read_csv('../data/feats_lyrics.csv')
#print(songs_df['lyrics'].iloc[0])
make_emotions(songs_df)




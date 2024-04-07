import pandas as pd
from transformers import pipeline
import re
import requests
from fuzzywuzzy import fuzz
from google.cloud import translate_v2 as translate
from dotenv import load_dotenv
import os
load_dotenv()

#Assigning emotions to the lyrics generated. Hopefully only one run needed

MAX_TOKENS = 512
PROJECT_ID = os.environ['PROJECT_ID']
PARENT = f"projects/{PROJECT_ID}"

EMOTION_DICT = {
    'admiration': 0, 'amusement': 0, 'anger':1, 'annoyance': 1, 'approval': 0, 
    'caring': 0, 'confusion': 1, 'curiosity': 2, 'desire': 2, 'disappointment': 1, 
    'disapproval': 1, 'disgust': 1, 'embarrassment': 1, 'excitement': 0, 'fear': 1, 
    'gratitude': 0, 'grief': 1, 'joy': 0, 'love': 0, 'nervousness': 2, 
    'optimism': 0, 'pride': 0, 'realization': 2, 'relief': 2, 'remorse': 1, 'sadness': 1, 'surprise': 2, 'neutral': 2
}


def translate_text(text, target_language_code):
    client = translate.Client("en")

    if isinstance(text, bytes):
        text = text.decode("utf-8")

    response = client.translate(text, target_language=target_language_code
    )

    return response["translatedText"]

def clean_lyrics(lyrics):
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
        translated_sent = translate_text(sent, target_lang)
        translated_sentences.append(translated_sent)
    
    # Join the unique sentences to form the deduplicated lyrics
    deduplicated_lyrics = '\n'.join(translated_sentences)
    return deduplicated_lyrics
    
def make_emotions(songs_df):

    songs_df['lyrics'] = songs_df['lyrics'].apply(clean_lyrics)
    #print(songs_df['lyrics'])
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
    emotions = []
    batch_size = 256
    for index, row in songs_df.iterrows():
        batch_emotions = []
        lyrics = row['lyrics']
        tokens = lyrics.split()
        #print(len(tokens))
        token_batches = []
        for i in range(0, len(tokens), batch_size):
            if (len(tokens) - i) < batch_size:
                token_batches.append(tokens[i:])
            else:
                token_batches.append(tokens[i:i+batch_size])
        #print(len(token_batches))
        for batch in token_batches:
            batch_lyric = ' '.join(batch)
            emo = classifier([batch_lyric])
            batch_emotions.append(emo)
            #print(batch_emotions)
        final_emotions = {emotion['label']: 0 for emotion in batch_emotions[0][0]}
        for batch in batch_emotions:
            for sublist in  batch_emotions:
                for subsublist in sublist:
                    for emotion in subsublist:
                        final_emotions[emotion['label']] += emotion['score']
        num_batches = len(batch_emotions)
        average_emotions = {emotion: score / num_batches for emotion, score in final_emotions.items()}
        sorted_emotions = sorted(average_emotions.items(), key=lambda x: x[1], reverse=True)
        top_emotion = dict(sorted_emotions[:1])
        emotions.append(next(iter(top_emotion.keys())))
    
    #print(emotions)
    songs_df['emotions'] = emotions
    songs_df['emotions_id'] = songs_df['emotions'].map(EMOTION_DICT)
    songs_df.to_csv('../data/song_emotions.csv')


def test_model():
    test_df = pd.read_csv('../data/top_songs.csv').head(1)
    track_id = test_df['id'].iloc[0]
    querystring = {"trackId":track_id}
    headers = {
        "X-RapidAPI-Key": "e66ab9e6c9msh0a0457e0c88540dp140b34jsnde29bcd01798",
        "X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
    }
    response = requests.get('https://spotify-scraper.p.rapidapi.com/v1/track/lyrics', headers=headers, params=querystring)
    test_df.loc[0, 'lyrics'] = response.text


    make_emotions(test_df)

#test_model()
songs_df = pd.read_csv('../data/feats_lyrics.csv')
#print(songs_df['lyrics'].iloc[0])
make_emotions(songs_df)




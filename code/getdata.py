import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from bs4 import BeautifulSoup
from lyricsgenius import Genius
import time
from dotenv import load_dotenv
import os
load_dotenv()


#Fetches spotify data and lyrics for each song using API keys. Will also ideally
#run only once

LYRIC_URL = "https://spotify-scraper.p.rapidapi.com/v1/track/lyrics"
CID =os.environ['CID']
SECRET_API = os.environ['SPOTIFY_SECRET']
#GENIUS_TOKEN = 'QhX0q-edkd7ZlG2u79k1K40Y6wLA-jmpCpfVJbuxZikzvRFHs7m70IqPF5BcgMjx'



def add_features(df):
	client_credentials_manager = SpotifyClientCredentials(client_id=CID, client_secret=SECRET_API)
	sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)
	#genius = Genius(GENIUS_TOKEN)
	missed_lyrics = []
	for i in range(len(df)):
		#querystring = {"url": df['url'].iloc[i], "format": "lrc"}
		#response = requests.get(LYRIC_URL, params=querystring)
		#if response.status_code != 200:
		#	print(response.json().get('error'))
		#else:	
		#	lines = response.json().get('lines', [])
		#	lyrics = '\n'.join([line['words'] for line in lines])
		#	df.loc[i, 'lyrics'] = lyrics
		track_id = df['id'].iloc[i]
		audio_features = sp.audio_features([track_id])[0]
		df.loc[i, 'acoustic'] = audio_features['acousticness']
		df.loc[i, 'dance'] = audio_features['danceability']
		df.loc[i, 'energy'] = audio_features['energy']
		df.loc[i, 'instrumental'] = audio_features['instrumentalness']
		df.loc[i, 'loudness'] = audio_features['loudness']
		df.loc[i, 'mode'] = audio_features['mode']
		df.loc[i, 'tempo'] = audio_features['tempo']
		df.loc[i, 'valence'] = audio_features['valence']
		df.loc[i, 'explicit'] = sp.track(track_id)['explicit']
		querystring = {"trackId":track_id}
		headers = {
			"X-RapidAPI-Key": os.environ['RAPIDAPI_KEY'],
			"X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
		}
		response = requests.get(LYRIC_URL, headers=headers, params=querystring)
		if (response.status_code == 200):
			df.loc[i, 'lyrics'] = response.text
		else:
			print(df['title'].iloc[i])
			missed_lyrics.append[i]
		time.sleep(0.5)
	
	for ind in missed_lyrics:
		track_id = df['id'].iloc[ind]
		querystring = {"trackId":track_id}
		headers = {
			"X-RapidAPI-Key": "e66ab9e6c9msh0a0457e0c88540dp140b34jsnde29bcd01798",
			"X-RapidAPI-Host": "spotify-scraper.p.rapidapi.com"
		}
		response = requests.get(LYRIC_URL, headers=headers, params=querystring)
		if (response.status_code == 200):
			df.loc[ind, 'lyrics'] = response.text
		else:
			print(df['title'].iloc[ind])
			df.drop(ind, inplace=True)

		time.sleep(0.5)
	
	df.reset_index(drop=True, inplace=True)
	df.to_csv('../data/feats_lyrics.csv')


add_features(pd.read_csv('../data/top_songs.csv'))

